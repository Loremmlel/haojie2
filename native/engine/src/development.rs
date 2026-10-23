//! 可信开发协议与诊断会话；正式训练只使用共用规则包和执行内核，不经过这个入口。
#[cfg(feature = "kernel-profile")]
use crate::profile;
use crate::{
    canonical, catalog as rules_catalog,
    command::apply_runtime,
    encoding, geometry, inspection,
    model::{COMMANDS, Catalog, Command, Failure, PROTOCOL, RULESET, State},
    records, runtime, tree,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::hint::black_box;
use std::io::{self, BufRead, Write};
use std::time::Instant;

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
            if let Some(traits) = &u.traits {
                kinds.extend(traits.iter().cloned());
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
            *catalog = Some(rules_catalog::from_request(request)?);
            Ok(
                json!({"protocol":PROTOCOL,"ruleset":RULESET,"commands":COMMANDS,"completeEngine":true}),
            )
        }
        "audit-record" => records::audit(
            std::path::Path::new(request["record"].as_str().ok_or("record required")?),
            catalog.as_ref().ok_or("initialize first")?,
            |_, _| Ok(()),
        ),
        "hash-states" => {
            // 可信开发入口的差分参照；正式训练仍只接收已校验起点与命令记录。
            let states: Vec<State> =
                serde_json::from_value(request["states"].take()).map_err(|e| e.to_string())?;
            Ok(json!(
                states
                    .iter()
                    .map(|s| {
                        let value = serde_json::to_value(s).unwrap();
                        json!({"typed":canonical::hash(s),"value":records::hash(&value)})
                    })
                    .collect::<Vec<_>>()
            ))
        }
        "training-nodes" => {
            let catalog = catalog.as_ref().ok_or("initialize first")?;
            let actor = request["actor"].as_u64().ok_or("actor required")? as usize;
            let cursors: Vec<Vec<usize>> =
                serde_json::from_value(request["cursors"].take()).map_err(|e| e.to_string())?;
            let mut tree = tree::Tree::new(&request["observation"], actor, catalog)?;
            let mut nodes = vec![];
            let mut encoded = vec![];
            for cursor in cursors {
                let node = tree.node(&cursor)?;
                if request["encode"] == true && !node.choices.is_empty() {
                    let input = encoding::encode(&tree, &node)?;
                    encoded.push(json!(input));
                } else if request["encode"] == true {
                    encoded.push(Value::Null);
                }
                nodes.push(node.wire());
            }
            Ok(json!({"nodes":nodes,"encoded":encoded}))
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
pub fn serve() -> io::Result<()> {
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
        let mut init: Value = serde_json::from_slice(rules_catalog::RULES).unwrap();
        init["ruleset"] = json!("old");
        assert!(handle(init.clone(), &mut catalog, &mut loaded, &mut resident).is_err());
        assert!(catalog.is_none());
        init["ruleset"] = json!(RULESET);
        init["protocol"] = json!("unknown");
        assert!(handle(init.clone(), &mut catalog, &mut loaded, &mut resident).is_err());
        assert!(catalog.is_none());
        init["protocol"] = json!(PROTOCOL);
        let mut incomplete = init.clone();
        incomplete["recipes"] = json!([]);
        assert!(handle(incomplete, &mut catalog, &mut loaded, &mut resident).is_err());
        assert!(catalog.is_none());
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
