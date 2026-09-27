//! 公开动作说明与 TS commands/options、training/queries 对齐；不评分、不裁剪参数域。
use crate::model::{Catalog, Command, Failure, State, Unit, extra_number, number};
use crate::movement::{charge_kind, reserve};
use crate::preparation::kind;
use crate::stats::{attack_charge, stats};
use serde_json::{Value, json};

#[derive(Clone)]
pub struct Step {
    pub kind: &'static str,
    pub field: &'static str,
    pub relation: &'static str,
    pub unit_only: bool,
}
impl Step {
    pub fn new(kind: &'static str) -> Self {
        Self {
            kind,
            field: "targetId",
            relation: "any",
            unit_only: false,
        }
    }
    fn target(relation: &'static str, field: &'static str, unit_only: bool) -> Self {
        Self {
            kind: "target",
            field,
            relation,
            unit_only,
        }
    }
}
#[derive(Clone)]
pub struct Action {
    pub id: String,
    pub command: Value,
    pub steps: Vec<Step>,
    pub free: bool,
    pub materials: Vec<String>,
    pub chosen: Vec<Value>,
}
fn action(id: &str, c: Value, steps: Vec<Step>) -> Action {
    Action {
        id: id.into(),
        command: c,
        steps,
        free: false,
        materials: vec![],
        chosen: vec![],
    }
}
pub fn array(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
pub fn text(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}
pub fn extend(v: &Value, fields: Value) -> Value {
    let mut c = v.clone();
    c.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    c
}

/// 拒绝权威私有字段；预检使用无 RNG 的公开局面，首次概率分支返回 uncertain。
pub fn position(observation: &Value) -> Result<State, String> {
    for key in ["seed", "rng", "log", "events", "past", "future", "present"] {
        if observation.get(key).is_some() {
            return Err(format!("training observation contains {key}"));
        }
    }
    let mut value = observation.clone();
    value["log"] = json!([]);
    value["events"] = json!([]);
    serde_json::from_value(value).map_err(|e| e.to_string())
}
pub fn permitted(s: &State, actor: usize, c: &Value) -> bool {
    if ![1, 2].contains(&actor)
        || c.get("player").is_some_and(|v| v != actor)
        || s.extra.contains_key("winner")
    {
        return false;
    }
    if let Some(r) = s.pending.first() {
        return c["type"] == "react" && r["owner"] == actor;
    }
    if s.phase == "shrine-draft" {
        return c["type"] == "choose-shrine"
            && s.extra.get("shrineDraft").unwrap_or(&Value::Null)["committed"][actor.to_string()]
                != true;
    }
    let u = c["unitId"].as_str().and_then(|id| s.unit(id));
    if c.get("unitId").is_some() && u.is_none_or(|u| u.owner != actor) {
        return false;
    }
    actor == s.active
        || (c["type"] == "skill"
            && u.is_some_and(|u| {
                c.get("ability").map_or(u.kind.is("u7"), |a| a == "u7") && u.has("u7")
            }))
}
pub fn inspect(
    s: &State,
    actor: usize,
    c: &Value,
    catalog: &Catalog,
) -> Result<&'static str, String> {
    if !permitted(s, actor, c) {
        return Ok("invalid");
    }
    let c: Command = serde_json::from_value(c.clone()).map_err(|e| e.to_string())?;
    match crate::transition(s, &c, catalog, true) {
        Ok(_) => Ok("available"),
        Err(Failure::Uncertain) => Ok("uncertain"),
        Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => Ok("invalid"),
        Err(Failure::Unsupported(r)) => Err(format!("unsupported training inspection: {r}")),
    }
}
fn allowed(s: &State, actor: usize, a: &Action, catalog: &Catalog) -> Result<bool, String> {
    if !permitted(s, actor, &a.command) {
        return Ok(false);
    }
    if !a.materials.is_empty() {
        return Ok(true);
    }
    if a.steps.is_empty() {
        return Ok(inspect(s, actor, &a.command, catalog)? != "invalid");
    }
    let c = &a.command;
    let id = a.id.split(':').next().unwrap();
    let command = text(&c["type"]);
    if command == "react" {
        return Ok(true);
    }
    if !s.pending.is_empty() {
        return Ok(false);
    }
    if command == "synthesize" {
        return Ok(s.phase == "synthesis");
    }
    if s.phase == "shrine-setup" && ["deploy", "equip", "activate-aura"].contains(&command) {
        return Ok(true);
    }
    if s.phase != "play" && command != "reroll" && id != "giant" {
        return Ok(false);
    }
    let Some(u) = c["unitId"].as_str().and_then(|id| s.unit(id)) else {
        return Ok(true);
    };
    let stats = stats(s, u, catalog);
    let ability = c.get("ability").map(kind).unwrap_or(u.kind.clone());
    let r = reserve(u, &ability, catalog);
    let borrowed = ability != u.kind;
    let usage = &u.extra.get("abilityUsage").unwrap_or(&Value::Null)[ability.key()];
    let once = if borrowed {
        usage["once"] == true
    } else {
        u.extra.get("onceUsed") == Some(&json!(true))
    };
    let free = if borrowed {
        usage["free"].as_f64().unwrap_or(-1.0)
    } else {
        u.extra
            .get("freeUsed")
            .and_then(Value::as_f64)
            .unwrap_or(f64::NAN)
    };
    if (u.owner != s.active && id != "giant")
        || stats.frozen
        || stats.stunned
        || (stats.sleeping && id != "giant")
    {
        return Ok(false);
    }
    if command == "attack" {
        return Ok(stats.remaining != 0.0
            && !(u.has("4")
                && !u.silenced
                && number(&reserve(u, &crate::model::Kind::Number(4), catalog)["readyCharge"])
                    < 2.0));
    }
    if !a.free
        && ((u.mode != "none" && !(command == "move" && u.mode == "move"))
            || stats.operations_left <= 0.0)
    {
        return Ok(false);
    }
    if command == "move"
        && u.mode == "none"
        && charge_kind(u, catalog)
            .is_some_and(|k| number(&reserve(u, &k, catalog)["readyCharge"]) < 1.0)
    {
        return Ok(false);
    }
    if id == "dash" && (number(&r["readyCharge"]) < 2.0 || u.hp <= 10.0) {
        return Ok(false);
    }
    if id == "cross" && (number(&r["readyCharge"]) < 1.0 || once) {
        return Ok(false);
    }
    if id == "siphon"
        && (free == s.ply || s.siphons.iter().filter(|l| l["sourceId"] == u.id).count() >= 3)
    {
        return Ok(false);
    }
    if a.free && (free == s.ply + u.offset || (id == "giant" && once)) {
        return Ok(false);
    }
    if id == "superhook"
        && !(u
            .extra
            .get("hookReadyAt")
            .and_then(Value::as_f64)
            .is_some_and(|v| v <= s.ply + u.offset)
            && extra_number(u, "hookExpiresAt") > s.ply + u.offset)
    {
        return Ok(false);
    }
    Ok(true)
}
fn unit_actions(s: &State, u: &Unit, catalog: &Catalog) -> Vec<Action> {
    if catalog[&u.kind.key()].landmark.is_some() && u.hp <= 0.0 {
        return vec![];
    }
    let mut result = vec![];
    let st = stats(s, u, catalog);
    let point = || Step::new("point");
    let target = |r, only| Step::target(r, "targetId", only);
    let base = |t| json!({"type":t,"unitId":u.id});
    if st.movement > 0.0 {
        result.push(action("move", base("move"), vec![point()]));
    }
    if st.actions > 0.0 && (u.silenced || !u.has("firelord")) {
        let c = base("attack");
        if u.signed() {
            result.push(action(
                "attack",
                extend(&c, json!({"mode":"damage"})),
                vec![target("any", false)],
            ));
            result.push(action(
                "heal",
                extend(&c, json!({"mode":"heal"})),
                vec![target("any", true)],
            ));
        } else {
            result.push(action("attack", c.clone(), vec![target("any", false)]));
        }
        if crate::combat::healing_attack(u, catalog) {
            result.push(action(
                "self-heal",
                extend(&c, json!({"targetId":u.id,"mode":"heal"})),
                vec![],
            ));
        }
        if u.weapon("u28") {
            result.push(action(
                "attack-path",
                extend(&c, json!({"mode":"damage","path":[]})),
                vec![Step::new("path")],
            ));
        }
    }
    let d = &catalog[&u.kind.key()];
    let ordinal = u
        .kind
        .key()
        .trim_start_matches(['u', 's'])
        .trim_end_matches('p')
        .parse::<u32>()
        .ok();
    if d.spell.is_none()
        && d.weapon.is_none()
        && !d.aura
        && d.landmark.is_none()
        && !u.any(&["grave", "wall"])
        && ordinal.is_some_and(|n| {
            array(&s.extra.get("auras").unwrap_or(&Value::Null)[u.owner.to_string()])
                .iter()
                .any(|a| {
                    a["kind"] == "s9" && a["parity"] == if n % 2 == 1 { "odd" } else { "even" }
                })
        })
    {
        result.push(action(
            "shatter",
            base("shatter"),
            vec![target("enemy", false)],
        ));
    }
    if ["move", "attack"].contains(&u.mode.as_str()) {
        result.push(action("finish", base("finish-mode"), vec![]));
    }
    for (yes, id, mode) in [
        (
            charge_kind(u, catalog).as_ref() == Some(&u.kind),
            "charge-move",
            "move",
        ),
        (
            attack_charge(u, catalog).as_ref() == Some(&u.kind),
            "charge-attack",
            "attack",
        ),
        (
            !u.silenced && ["4", "15", "u2"].iter().any(|k| u.kind.is(k)),
            "charge-attack",
            "attack",
        ),
        (
            !u.silenced && ["21", "u6"].iter().any(|k| u.kind.is(k)),
            "charge-skill",
            "skill",
        ),
    ] {
        if yes {
            result.push(action(
                id,
                extend(&base("charge"), json!({"mode":mode})),
                vec![],
            ));
        }
    }
    if !u.silenced {
        let c = base("skill");
        let mut add = |id, c, steps| result.push(action(id, c, steps));
        match u.kind.key().as_str() {
            "5" => add("stomp", c, vec![]),
            "6" => add("buff", c, vec![]),
            "7" => add("hook", c, vec![target("enemy", true), point()]),
            "14" => {
                add(
                    "sacrifice-summon",
                    extend(&c, json!({"mode":"summon"})),
                    vec![target("friend", true)],
                );
                add(
                    "sacrifice",
                    c,
                    vec![target("friend", true), Step::new("column")],
                );
            }
            "19" => add("wall", c, vec![point()]),
            "21" => add("dash", c, vec![point(), target("enemy", false)]),
            "u6" => add("cross", c, vec![point()]),
            "u7" => {
                let mut a = action("giant", c, vec![target("any", true), point()]);
                a.free = true;
                result.push(a);
            }
            "u14" => add(
                "siphon",
                c,
                vec![target("any", false), Step::target("any", "secondId", false)],
            ),
            "u19" => add("revive", c, vec![Step::new("death"), point()]),
            "u21" => add("single-buff", c, vec![target("friend", true)]),
            "u23" => add("superhook", c, vec![target("enemy", true)]),
            "u24" => add("ice-mark", c, vec![point()]),
            _ => {}
        }
        for k in array(u.extra.get("traits").unwrap_or(&Value::Null)) {
            let k = kind(k);
            let mut borrowed = u.clone();
            borrowed.extra.remove("traits");
            borrowed
                .extra
                .extend(reserve(u, &k, catalog).as_object().unwrap().clone());
            borrowed.kind = k.clone();
            let usage = &u.extra.get("abilityUsage").unwrap_or(&Value::Null)[k.key()];
            borrowed
                .extra
                .insert("onceUsed".into(), json!(usage["once"] == true));
            borrowed.extra.insert(
                "freeUsed".into(),
                json!(usage["free"].as_f64().unwrap_or(-1.0)),
            );
            for mut a in unit_actions(s, &borrowed, catalog) {
                if ["skill", "charge"].contains(&text(&a.command["type"])) {
                    a.id = format!("{}:{}", a.id, k.key());
                    a.command["ability"] = json!(k);
                    result.push(a);
                }
            }
        }
    }
    result
}
fn card_actions(s: &State, card: &Value, catalog: &Catalog) -> Vec<Action> {
    let k = kind(&card["kind"]);
    let d = &catalog[&k.key()];
    let base = |t| json!({"type":t,"cardId":card["id"]});
    let point = || Step::new("point");
    let target = |r| Step::target(r, "targetId", true);
    let mut result = vec![];
    if d.spell.is_none() && d.weapon.is_none() && !d.aura {
        result.push(action(
            "deploy",
            extend(&base("deploy"), json!({"charge":false})),
            vec![point()],
        ));
        if k.is("1") {
            result.push(action(
                "deploy-charge",
                extend(&base("deploy"), json!({"charge":true})),
                vec![point()],
            ));
        }
    } else if d.aura {
        result.push(action("activate-aura", base("activate-aura"), vec![]));
    } else if d.weapon.is_some() {
        result.push(action("equip", base("equip"), vec![target("friend")]));
    } else if k.is("8") {
        result.push(action("blast", base("cast"), vec![point()]));
    } else if k.is("25") {
        result.push(action(
            "reforge-one",
            extend(&base("cast"), json!({"mode":"single"})),
            vec![],
        ));
        result.push(action(
            "reforge-two",
            extend(&base("cast"), json!({"mode":"double"})),
            vec![Step::target("friend", "sacrificeIds", true); 2],
        ));
    } else if k.is("u9") {
        for (id, mode) in [("storm-row", "row"), ("storm-col", "column")] {
            result.push(action(
                id,
                extend(&base("cast"), json!({"mode":mode})),
                vec![Step::new(mode)],
            ));
        }
    } else {
        result.push(action(
            "cast",
            base("cast"),
            vec![target(if k.is("u26") { "any" } else { "friend" })],
        ));
    }
    // 完整无参数命令统一走公开预检；按实际部署时刻稳定排序，保留 TS 改判顺序。
    result.push(action("self-reroll", base("reroll"), vec![]));
    let mut mages: Vec<_> = s
        .units
        .iter()
        .filter(|u| u.has("u13") && u.owner == s.active)
        .collect();
    mages.sort_by(|a, b| extra_number(a, "deployedAt").total_cmp(&extra_number(b, "deployedAt")));
    for u in mages {
        result.push(action(
            &format!("reroll-{}", u.id),
            extend(&base("reroll"), json!({"unitId":u.id})),
            vec![],
        ));
    }
    result
}
fn pool<'a>(s: &State, c: &Value, catalog: &'a Catalog) -> Option<&'a str> {
    let card = array(&s.extra["hands"][s.active.to_string()])
        .iter()
        .find(|v| v["id"] == c["cardId"]);
    match text(&c["type"]) {
        "summon" => Some(
            if s.extra.get("mode") == Some(&json!("shrine")) || c["ultimate"] == true {
                "ultimate"
            } else {
                "normal"
            },
        ),
        "extra-summon" => Some(if c["ultimate"] == false {
            "normal"
        } else {
            "ultimate"
        }),
        "cast" if card.is_some_and(|v| v["kind"] == 25) => {
            Some(if s.extra.get("mode") == Some(&json!("shrine")) {
                "ultimate"
            } else {
                "normal"
            })
        }
        "reroll" => card.and_then(|v| {
            let k = kind(&v["kind"]);
            if v["summonPool"] == "normal"
                || catalog.pools["normal"].contains(&k)
                || k.is("3p")
                || k.is("17p")
            {
                Some("normal")
            } else if v["summonPool"] == "ultimate"
                || catalog.pools["ultimate"].contains(&k)
                || k.is("u12p")
            {
                Some("ultimate")
            } else {
                None
            }
        }),
        _ => None,
    }
}
pub fn actions(s: &State, actor: usize, catalog: &Catalog) -> Result<Vec<Action>, String> {
    let mut candidates = vec![];
    let simple = |c: Value| action(text(&c["type"]), c.clone(), vec![]);
    if s.extra.contains_key("winner") {
        return Ok(vec![]);
    }
    if let Some(r) = s.pending.first() {
        let k = text(&r["kind"]);
        candidates.push(action(
            k,
            if k == "hit-pull" {
                json!({"type":"react","mode":"pull"})
            } else {
                json!({"type":"react"})
            },
            match k {
                "hit-pull" => vec![],
                "bounce" | "hut-spawn" => vec![Step::new("point")],
                _ => vec![Step::target("any", "targetId", false)],
            },
        ));
        candidates.push(simple(json!({"type":"react"})));
    } else if s.phase == "shrine-draft" {
        for k in
            array(&s.extra.get("shrineDraft").unwrap_or(&Value::Null)["offers"][actor.to_string()])
        {
            let c = json!({"type":"choose-shrine","player":actor,"shrineKind":k});
            if k == "s9" {
                for p in ["odd", "even"] {
                    candidates.push(simple(extend(&c, json!({"parity":p}))));
                }
            } else {
                candidates.push(simple(c));
            }
        }
    } else if let Some(offer) = s.extra.get("summonOffer") {
        for i in 0..array(&offer["groups"]).len() {
            for j in i + 1..array(&offer["groups"]).len() {
                candidates.push(simple(
                    json!({"type":"choose-summons","offerIndices":[i,j]}),
                ));
            }
        }
    } else {
        for t in ["begin", "end", "skip-synthesis", "finish-shrine-setup"] {
            candidates.push(simple(json!({"type":t})));
        }
        for t in ["summon", "extra-summon"] {
            for ultimate in [false, true] {
                candidates.push(simple(json!({"type":t,"ultimate":ultimate})));
            }
        }
        for u in s.pieces().filter(|u| u.owner == actor) {
            candidates.extend(unit_actions(s, u, catalog));
        }
        if actor == s.active {
            for c in array(&s.extra["hands"][actor.to_string()]) {
                candidates.extend(card_actions(s, c, catalog));
            }
            candidates.push(action(
                "clock",
                json!({"type":"clock"}),
                vec![Step::target("any", "targetId", true)],
            ));
            if s.phase == "synthesis" {
                for r in &catalog.recipes {
                    let ids = crate::synthesis::materials(s, r);
                    if ids.len() < 3 {
                        continue;
                    }
                    let mut a = action(
                        &format!("synthesize:{}", text(&r["id"])),
                        json!({"type":"synthesize","recipeId":r["id"]}),
                        if catalog[&kind(&r["result"]).key()].aura {
                            vec![]
                        } else {
                            vec![Step::new("point")]
                        },
                    );
                    a.materials = ids;
                    candidates.push(a);
                }
            }
        }
    }
    let can_choose = array(&s.extra.get("auras").unwrap_or(&Value::Null)[actor.to_string()])
        .iter()
        .any(|a| a["kind"] == "laoqian" && a["usedPly"] != s.ply);
    let mut result = vec![];
    for mut a in candidates {
        if allowed(s, actor, &a, catalog)? {
            if can_choose && let Some(p) = pool(s, &a.command, catalog) {
                a.chosen = catalog.pools[p].iter().map(|k| json!(k)).collect();
            }
            result.push(a);
        }
    }
    Ok(result)
}
