#[path = "commands/abilities.rs"]
mod abilities;
#[path = "training/actions.rs"]
mod actions;
#[path = "commands/combat.rs"]
mod combat;
#[path = "commands/damage.rs"]
mod damage;
#[path = "training/encoding.rs"]
mod encoding;
#[path = "core/geometry.rs"]
mod geometry;
#[path = "commands/inspection.rs"]
mod inspection;
#[path = "commands/lifecycle.rs"]
mod lifecycle;
mod model;
#[path = "commands/movement.rs"]
mod movement;
#[path = "training/policy.rs"]
mod policy;
#[path = "commands/preparation.rs"]
mod preparation;
#[path = "commands/reactions.rs"]
mod reactions;
#[path = "core/resolution.rs"]
mod resolution;
#[path = "setup/runtime.rs"]
mod runtime;
#[path = "training/sampler.rs"]
mod sampler;
#[path = "core/shared.rs"]
mod shared;
#[path = "setup/shrines.rs"]
mod shrines;
#[path = "commands/spells.rs"]
mod spells;
#[path = "core/stats.rs"]
mod stats;
#[path = "setup/synthesis.rs"]
mod synthesis;
#[path = "training/tree.rs"]
mod tree;

use model::{COMMANDS, Catalog, Command, Definition, Failure, PROTOCOL, RULESET, State};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::io::{self, BufRead, Write};
use std::time::Instant;

/// 已完成公共阶段检查；预检与正式连续运行共用同一个规则内核。
fn execute(s: &State, c: &Command, catalog: &Catalog, preview: bool) -> Result<State, Failure> {
    match c.kind.as_str() {
        "attack" | "react" => combat::apply(s, c, catalog, preview),
        "move"
            if s.unit(c.unit_id.as_deref().unwrap_or(""))
                .is_some_and(model::Unit::runner) =>
        {
            reactions::move_runner(s, c, catalog, preview)
        }
        "move" | "finish-mode" => movement::apply(s, c, catalog),
        _ if COMMANDS.contains(&c.kind.as_str()) => preparation::apply(s, c, catalog, preview),
        _ => Err(Failure::Unsupported("command-kind")),
    }
}
fn apply_runtime(s: &State, c: &Command, catalog: &Catalog) -> Result<State, Failure> {
    movement::stage(s, c, catalog)?;
    execute(s, c, catalog, false)
}
fn transition(s: &State, c: &Command, catalog: &Catalog) -> Result<State, Failure> {
    // 外部导出保持独立拥有；内部连续执行依靠 Rc 写时分离保留旧局面。
    let next = apply_runtime(s, c, catalog)?;
    // State::clone 是深复制快照出口；内部提交才保留 Rc 共享。
    Ok(next.clone())
}

#[derive(Deserialize)]
struct Job {
    state: State,
    #[serde(default)]
    probes: Vec<Command>,
    command: Option<Command>,
}
#[derive(Default)]
struct Resident {
    state: Option<State>,
    revision: u64,
}
#[derive(Serialize)]
struct ResultState {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<State>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
}
impl ResultState {
    fn from(result: Result<State, Failure>) -> Self {
        match result {
            Ok(state) => Self {
                status: "available",
                state: Some(state),
                message: None,
                reason: None,
            },
            Err(Failure::Invalid(m)) => Self {
                status: "invalid",
                state: None,
                message: Some(m.into()),
                reason: None,
            },
            Err(Failure::InvalidOwned(m)) => Self {
                status: "invalid",
                state: None,
                message: Some(m),
                reason: None,
            },
            Err(Failure::Unsupported(r)) => Self {
                status: "unsupported",
                state: None,
                message: None,
                reason: Some(r),
            },
            Err(Failure::Uncertain) => Self {
                status: "uncertain",
                state: None,
                message: None,
                reason: None,
            },
        }
    }
}
#[derive(Serialize)]
struct Response {
    inspections: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<ResultState>,
}
fn evaluate(job: &Job, catalog: &Catalog) -> Response {
    let mut queries = inspection::Queries::default();
    let inspections = job
        .probes
        .iter()
        .map(|c| match queries.inspect(&job.state, c, catalog) {
            Ok(_) => json!({"status":"available"}),
            Err(e) => e.value(),
        })
        .collect();
    let result = job
        .command
        .as_ref()
        .map(|c| ResultState::from(transition(&job.state, c, catalog)));
    Response {
        inspections,
        result,
    }
}

/// 本地实验协议只接收 TS 正式引擎重建的规范局面，不是联网存档解析器。
/// 原型拒绝超出表示范围的局面；不对任意输入做缺省补全，也不向 TS 隐式回退。
fn jobs(value: Value, catalog: &Catalog) -> Result<Vec<Job>, String> {
    let jobs: Vec<Job> = serde_json::from_value(value).map_err(|e| e.to_string())?;
    for job in &jobs {
        let s = &job.state;
        if ![1, 2].contains(&s.active) || !s.turns.contains_key("1") || !s.turns.contains_key("2") {
            return Err("unsupported-state: player/turns".into());
        }
        for u in s.units.iter().chain(s.landmarks().iter()) {
            if ![1, 2].contains(&u.owner)
                || ![1.0, 2.0].contains(&u.size)
                || !geometry::inside(u.at())
                || geometry::cells(u, u.at())
                    .iter()
                    .any(|p| !geometry::inside(*p))
            {
                return Err("unsupported-state: unit geometry/owner".into());
            }
            let mut kinds = vec![u.kind.clone()];
            if let Some(traits) = u.extra.get("traits") {
                kinds.extend(
                    serde_json::from_value::<Vec<model::Kind>>(traits.clone())
                        .map_err(|e| e.to_string())?,
                );
            }
            if kinds.iter().any(|k| !catalog.contains_key(&k.key())) {
                return Err("unknown-catalog-kind".into());
            }
        }
    }
    Ok(jobs)
}
/// 本地实验协议；初始化与重置先验证再替换，连续命令只提交成功前缀。
/// 修订号拒绝过期批次；规则失败作为结果返回，协议解析错误保持驻留状态不变。
fn handle(
    mut request: Value,
    catalog: &mut Option<Catalog>,
    loaded: &mut Vec<Job>,
    resident: &mut Resident,
) -> Result<Value, String> {
    match request["op"].as_str().unwrap_or("") {
        "init" => {
            if request["protocol"] != PROTOCOL || request["ruleset"] != RULESET {
                return Err("protocol/ruleset mismatch".into());
            }
            if catalog.is_some() {
                return Err("already initialized".into());
            }
            let printed: Vec<Value> =
                serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
            let definitions: Vec<Definition> =
                serde_json::from_value(request["catalog"].take()).map_err(|e| e.to_string())?;
            let mut entries = BTreeMap::new();
            for d in definitions {
                if entries.insert(d.id.key(), d).is_some() {
                    return Err("duplicate catalog kind".into());
                }
            }
            if entries.is_empty() {
                return Err("empty catalog".into());
            }
            let combat = request["combat"].take();
            for path in [
                "/accumulator/attack",
                "/accumulator/range",
                "/accumulator/max",
                "/goldSpellChance",
                "/sageAuraAttack",
                "/littleGoldImmunity",
                "/kingAttackImmunity",
                "/frontDamageCap",
                "/slayerReflectRate",
                "/catapultMarkDamage",
                "/charger/heavyChance",
                "/charger/criticalChance",
                "/charger/heavyBonus",
                "/charger/bonus",
                "/superCritical/lethalChance",
                "/superCritical/doubleChance",
                "/superCritical/lethalDamage",
                "/vampire/base",
                "/vampire/perKill",
                "/minerBaseDamage",
                "/sacrificeMaxHpCost",
                "/giantAreaDamage",
                "/firelord/radius",
                "/firelord/damage",
                "/firelord/splash",
                "/archmageCounterChance",
                "/archmageCounterHealth",
            ] {
                if !combat.pointer(path).is_some_and(Value::is_number) {
                    return Err(format!("missing combat rule: {path}"));
                }
            }
            let pools: BTreeMap<String, Vec<model::Kind>> =
                serde_json::from_value(request["summonPools"].take()).map_err(|e| e.to_string())?;
            for name in ["normal", "ultimate", "shrine"] {
                if !pools.get(name).is_some_and(|pool| {
                    !pool.is_empty() && pool.iter().all(|k| entries.contains_key(&k.key()))
                }) {
                    return Err(format!("invalid summon pool: {name}"));
                }
            }
            if pools["shrine"].len() < 3 {
                return Err("shrine pool requires at least 3 entries".into());
            }
            let recipes: Vec<Value> =
                serde_json::from_value(request["recipes"].take()).map_err(|e| e.to_string())?;
            if recipes.is_empty()
                || recipes.iter().any(|r| {
                    r["id"].as_str().is_none_or(str::is_empty)
                        || !r["source"]
                            .as_str()
                            .is_some_and(|s| ["board", "hand"].contains(&s))
                        || ["material", "result"].iter().any(|k| {
                            serde_json::from_value::<model::Kind>(r[k].clone())
                                .map_or(true, |k| !entries.contains_key(&k.key()))
                        })
                })
            {
                return Err("invalid synthesis recipes".into());
            }
            let encoding = request["encoding"].take();
            if !encoding.is_null() {
                encoding::validate_schema(&encoding)?;
            }
            *catalog = Some(Catalog {
                entries,
                combat,
                pools,
                recipes,
                printed,
                encoding,
            });
            Ok(
                json!({"protocol":PROTOCOL,"ruleset":RULESET,"commands":COMMANDS,"completeEngine":true}),
            )
        }
        "sample-game" => sampler::game(&request, catalog.as_ref().ok_or("initialize first")?),
        "training-nodes" => {
            let catalog = catalog.as_ref().ok_or("initialize first")?;
            let actor = request["actor"].as_u64().ok_or("actor required")? as usize;
            let cursors: Vec<Vec<usize>> =
                serde_json::from_value(request["cursors"].take()).map_err(|e| e.to_string())?;
            let mut tree = tree::Tree::new(&request["observation"], actor, catalog)?;
            let policy = policy::TinyPolicy::new(73129);
            let mut nodes = vec![];
            let mut encoded = vec![];
            let mut logits = vec![];
            for cursor in cursors {
                let node = tree.node(&cursor)?;
                if request["encode"] == true && !node.choices.is_empty() {
                    let input = encoding::encode(&tree, &node)?;
                    logits.push(json!(policy.logits(&input)));
                    encoded.push(json!(input));
                } else if request["encode"] == true {
                    logits.push(Value::Null);
                    encoded.push(Value::Null);
                }
                nodes.push(node.wire());
            }
            Ok(json!({"nodes":nodes,"encoded":encoded,"logits":logits}))
        }
        "run" | "load" => {
            let catalog = catalog.as_ref().ok_or("initialize first")?;
            let incoming = jobs(request["jobs"].take(), catalog)?;
            if request["op"] == "load" {
                *loaded = incoming;
                Ok(json!({"jobs":loaded.len()}))
            } else {
                serde_json::to_value(
                    incoming
                        .iter()
                        .map(|j| evaluate(j, catalog))
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())
            }
        }
        "create" => {
            let catalog = catalog.as_ref().ok_or("initialize first")?;
            let seed = request["seed"]
                .as_f64()
                .filter(|v| v.is_finite() && v.fract() == 0.0 && v.abs() <= 9007199254740991.0)
                .ok_or("seed must be a safe integer")?;
            let rules = request["rules"].as_str().unwrap_or("classic");
            if !["classic", "shrine"].contains(&rules) {
                return Err("unknown rules mode".into());
            }
            let mut state = runtime::create(seed as i64 as u32, rules == "shrine", catalog)?;
            if request["clearHistory"] == true {
                state.events.clear();
                state.extra.insert("log".into(), json!([]));
            }
            let observation = runtime::observe(&state, runtime::viewer(&state))?;
            resident.state = Some(state);
            resident.revision += 1;
            Ok(json!({"revision":resident.revision,"observation":observation}))
        }
        "observe" => {
            let state = resident.state.as_ref().ok_or("reset first")?;
            let viewer = request
                .get("viewer")
                .map(|v| v.as_u64().ok_or("invalid viewer"))
                .transpose()?
                .map(|v| v as usize)
                .unwrap_or_else(|| runtime::viewer(state));
            runtime::observe(state, viewer)
        }
        "bench" => {
            let catalog = catalog.as_ref().ok_or("initialize first")?;
            let repeats = request["repeats"]
                .as_u64()
                .filter(|n| (1..=1000).contains(n))
                .ok_or("invalid repeats")?;
            if loaded.is_empty() {
                return Err("load jobs first".into());
            }
            let start = Instant::now();
            for _ in 0..repeats {
                for job in loaded.iter() {
                    black_box(evaluate(job, catalog));
                }
            }
            Ok(
                json!({"elapsedMs":start.elapsed().as_secs_f64()*1000.0,"jobs":loaded.len() as u64*repeats}),
            )
        }
        "reset" => {
            let catalog = catalog.as_ref().ok_or("initialize first")?;
            let mut incoming = jobs(json!([{"state":request["state"].take()}]), catalog)?;
            resident.state = Some(incoming.remove(0).state);
            resident.revision += 1;
            Ok(json!({"revision":resident.revision}))
        }
        "export" => Ok(
            json!({"revision":resident.revision,"state":resident.state.as_ref().ok_or("reset first")?}),
        ),
        "step" => {
            let catalog = catalog.as_ref().ok_or("initialize first")?;
            if request["revision"].as_u64() != Some(resident.revision) {
                return Err("stale revision".into());
            }
            let commands: Vec<Command> =
                serde_json::from_value(request["commands"].take()).map_err(|e| e.to_string())?;
            let trace = request["trace"] == true;
            let clear = request["clearHistory"] == true;
            let mut results = vec![];
            for c in commands {
                match apply_runtime(resident.state.as_ref().ok_or("reset first")?, &c, catalog) {
                    Ok(mut next) => {
                        results.push(if trace {
                            json!({"status":"available","state":next})
                        } else {
                            json!({"status":"available"})
                        });
                        if clear {
                            next.events.clear();
                            next.extra.insert("log".into(), json!([]));
                        }
                        resident.state = Some(next);
                        resident.revision += 1;
                    }
                    Err(error) => {
                        results.push(error.value());
                        break;
                    }
                }
            }
            let mut response = json!({"revision":resident.revision,"results":results});
            if request["observe"] == true {
                let state = resident.state.as_ref().unwrap();
                response["observation"] = runtime::observe(state, runtime::viewer(state))?;
            }
            Ok(response)
        }
        _ => Err("unknown operation".into()),
    }
}
fn main() -> io::Result<()> {
    let mut catalog = None;
    let mut loaded = vec![];
    let mut resident = Resident::default();
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let answer = serde_json::from_str(&line?)
            .map_err(|e| e.to_string())
            .and_then(|request| handle(request, &mut catalog, &mut loaded, &mut resident));
        let answer = answer.unwrap_or_else(|error| json!({"error":error}));
        serde_json::to_writer(&mut output, &answer)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_rejects_mismatch_and_does_not_partially_initialize() {
        let mut catalog = None;
        let mut loaded = vec![];
        let mut resident = Resident::default();
        let mut init = json!({"op":"init","protocol":PROTOCOL,"ruleset":"old","catalog":[{"id":1,"name":"test","tier":"normal","attack":1,"health":1,"range":1,"actions":1,"move":2}],
            "summonPools":{"normal":[1],"ultimate":[1],"shrine":[1,1,1]},"recipes":[{"id":"test","material":1,"result":1,"source":"board"}],
            "combat":{"accumulator":{"attack":1,"range":1,"max":4},"goldSpellChance":0.5,"sageAuraAttack":1,"littleGoldImmunity":0.5,"kingAttackImmunity":1,"frontDamageCap":1,"slayerReflectRate":0.5,"catapultMarkDamage":1,
                "charger":{"heavyChance":0.1,"criticalChance":0.2,"heavyBonus":1,"bonus":1},"superCritical":{"lethalChance":0.1,"doubleChance":0.2,"lethalDamage":1},"vampire":{"base":0.2,"perKill":0.2},"minerBaseDamage":1,"sacrificeMaxHpCost":20,"giantAreaDamage":20,"firelord":{"radius":5,"damage":40,"splash":20},"archmageCounterChance":0.5,"archmageCounterHealth":10}});
        assert!(handle(init.clone(), &mut catalog, &mut loaded, &mut resident).is_err());
        assert!(catalog.is_none());
        init["ruleset"] = json!(RULESET);
        init["protocol"] = json!("unknown");
        assert!(handle(init.clone(), &mut catalog, &mut loaded, &mut resident).is_err());
        assert!(catalog.is_none());
        init["protocol"] = json!(PROTOCOL);
        assert_eq!(
            handle(init.clone(), &mut catalog, &mut loaded, &mut resident).unwrap()["completeEngine"],
            true
        );
        assert!(handle(init, &mut catalog, &mut loaded, &mut resident).is_err());
        assert!(
            handle(
                json!({"op":"bench","repeats":1}),
                &mut catalog,
                &mut loaded,
                &mut resident
            )
            .is_err()
        );
        assert!(
            handle(
                json!({"op":"run","jobs":[{}]}),
                &mut catalog,
                &mut loaded,
                &mut resident
            )
            .is_err()
        );
        assert!(
            handle(
                json!({"op":"step","revision":1,"commands":[]}),
                &mut catalog,
                &mut loaded,
                &mut resident
            )
            .is_err()
        );
        assert_eq!(
            handle(
                json!({"op":"run","jobs":[]}),
                &mut catalog,
                &mut loaded,
                &mut resident
            )
            .unwrap(),
            json!([])
        );
    }
}
