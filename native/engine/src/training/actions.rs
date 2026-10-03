//! 公开动作说明与 TS commands/options、training/queries 对齐；不评分、不裁剪参数域。
use crate::inspection::Queries;
use crate::model::{Catalog, Command, Failure, State, Unit, extra_number};
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
    pub command: Command,
    pub steps: Vec<Step>,
    pub free: bool,
    pub materials: Vec<String>,
    pub chosen: Vec<crate::model::Kind>,
}
fn action(id: &str, c: Command, steps: Vec<Step>) -> Action {
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
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Compatibility);
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
pub fn permitted_command(s: &State, actor: usize, c: &Command) -> bool {
    if ![1, 2].contains(&actor)
        || c.player.is_some_and(|v| v != actor)
        || s.extra.contains_key("winner")
    {
        return false;
    }
    if let Some(r) = s.pending.first() {
        return c.kind == "react" && r.owner == actor;
    }
    if s.phase == "shrine-draft" {
        return c.kind == "choose-shrine"
            && s.extra.get("shrineDraft").unwrap_or(&Value::Null)["committed"][actor.to_string()]
                != true;
    }
    let u = c.unit_id.as_deref().and_then(|id| s.unit(id));
    if c.unit_id.is_some() && u.is_none_or(|u| u.owner != actor) {
        return false;
    }
    actor == s.active
        || (c.kind == "skill"
            && u.is_some_and(|u| {
                c.ability.as_ref().map_or(u.kind.is("u7"), |a| a.is("u7")) && u.has("u7")
            }))
}
pub fn inspect(
    s: &State,
    actor: usize,
    c: &Command,
    catalog: &Catalog,
    queries: &mut Queries,
) -> Result<&'static str, String> {
    if !permitted_command(s, actor, c) {
        return Ok("invalid");
    }
    match queries.inspect(s, c, catalog) {
        Ok(_) => Ok("available"),
        Err(Failure::Uncertain) => Ok("uncertain"),
        Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => Ok("invalid"),
        Err(Failure::Unsupported(r)) => Err(format!("unsupported training inspection: {r}")),
    }
}
fn allowed(
    s: &State,
    actor: usize,
    a: &Action,
    catalog: &Catalog,
    queries: &mut Queries,
) -> Result<bool, String> {
    if !permitted_command(s, actor, &a.command) {
        return Ok(false);
    }
    if !a.materials.is_empty() {
        return Ok(true);
    }
    if a.steps.is_empty() {
        return Ok(inspect(s, actor, &a.command, catalog, queries)? != "invalid");
    }
    let c = &a.command;
    let id = a.id.split(':').next().unwrap();
    let command = c.kind.as_str();
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
    let Some(u) = c.unit_id.as_deref().and_then(|id| s.unit(id)) else {
        return Ok(true);
    };
    let stats = stats(s, u, catalog);
    let ability = c.ability.clone().unwrap_or(u.kind.clone());
    let r = reserve(u, &ability, catalog);
    let borrowed = ability != u.kind;
    let usage = u.ability_usage.as_ref().and_then(|m| m.get(&ability.key()));
    let once = if borrowed {
        usage.is_some_and(|v| v.once)
    } else {
        u.once_used
    };
    let free = if borrowed {
        usage.map(|v| v.free).unwrap_or(-1.0)
    } else {
        u.free_used
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
                && reserve(u, &crate::model::Kind::Number(4), catalog).ready_charge < 2.0));
    }
    if !a.free
        && ((u.mode != "none" && !(command == "move" && u.mode == "move"))
            || stats.operations_left <= 0.0)
    {
        return Ok(false);
    }
    if command == "move"
        && u.mode == "none"
        && charge_kind(u, catalog).is_some_and(|k| reserve(u, &k, catalog).ready_charge < 1.0)
    {
        return Ok(false);
    }
    if id == "dash" && (r.ready_charge < 2.0 || u.hp <= 10.0) {
        return Ok(false);
    }
    if id == "cross" && (r.ready_charge < 1.0 || once) {
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
    if catalog.by_kind(&u.kind).landmark.is_some() && u.hp <= 0.0 {
        return vec![];
    }
    let mut result = vec![];
    let st = stats(s, u, catalog);
    let point = || Step::new("point");
    let target = |r, only| Step::target(r, "targetId", only);
    let base = |t| Command {
        unit_id: Some(u.id.clone()),
        ..Command::new(t)
    };
    if st.movement > 0.0 {
        result.push(action("move", base("move"), vec![point()]));
    }
    if st.actions > 0.0 && (u.silenced || !u.has("firelord")) {
        let c = base("attack");
        if u.signed() {
            result.push(action(
                "attack",
                Command {
                    mode: Some("damage".into()),
                    ..c.clone()
                },
                vec![target("any", false)],
            ));
            result.push(action(
                "heal",
                Command {
                    mode: Some("heal".into()),
                    ..c.clone()
                },
                vec![target("any", true)],
            ));
        } else {
            result.push(action("attack", c.clone(), vec![target("any", false)]));
        }
        if crate::combat::healing_attack(u, catalog) {
            result.push(action(
                "self-heal",
                Command {
                    target_id: Some(u.id.clone()),
                    mode: Some("heal".into()),
                    ..c.clone()
                },
                vec![],
            ));
        }
        if u.weapon("u28") {
            result.push(action(
                "attack-path",
                Command {
                    mode: Some("damage".into()),
                    path: Some(vec![]),
                    ..c.clone()
                },
                vec![Step::new("path")],
            ));
        }
    }
    let d = catalog.by_kind(&u.kind);
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
                Command {
                    mode: Some(mode.into()),
                    ..base("charge")
                },
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
                    Command {
                        mode: Some("summon".into()),
                        ..c.clone()
                    },
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
        for k in u.traits.iter().flatten() {
            let mut borrowed = u.clone();
            borrowed.traits = None;
            borrowed.reserve = reserve(u, k, catalog);
            borrowed.kind = k.clone();
            let usage = u.ability_usage.as_ref().and_then(|m| m.get(&k.key()));
            borrowed.once_used = usage.is_some_and(|v| v.once);
            borrowed.free_used = usage.map(|v| v.free).unwrap_or(-1.0);
            for mut a in unit_actions(s, &borrowed, catalog) {
                if ["skill", "charge"].contains(&a.command.kind.as_str()) {
                    a.id = format!("{}:{}", a.id, k.key());
                    a.command.ability = Some(k.clone());
                    result.push(a);
                }
            }
        }
    }
    result
}
fn card_actions(s: &State, card: &Value, catalog: &Catalog) -> Vec<Action> {
    let k = kind(&card["kind"]);
    let d = catalog.by_kind(&k);
    let base = |t| Command {
        card_id: Some(text(&card["id"]).into()),
        ..Command::new(t)
    };
    let point = || Step::new("point");
    let target = |r| Step::target(r, "targetId", true);
    let mut result = vec![];
    if d.spell.is_none() && d.weapon.is_none() && !d.aura {
        result.push(action(
            "deploy",
            Command {
                charge: Some(false),
                ..base("deploy")
            },
            vec![point()],
        ));
        if k.is("1") {
            result.push(action(
                "deploy-charge",
                Command {
                    charge: Some(true),
                    ..base("deploy")
                },
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
            Command {
                mode: Some("single".into()),
                ..base("cast")
            },
            vec![],
        ));
        result.push(action(
            "reforge-two",
            Command {
                mode: Some("double".into()),
                ..base("cast")
            },
            vec![Step::target("friend", "sacrificeIds", true); 2],
        ));
    } else if k.is("u9") {
        for (id, mode) in [("storm-row", "row"), ("storm-col", "column")] {
            result.push(action(
                id,
                Command {
                    mode: Some(mode.into()),
                    ..base("cast")
                },
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
            Command {
                unit_id: Some(u.id.clone()),
                ..base("reroll")
            },
            vec![],
        ));
    }
    result
}
fn pool<'a>(s: &State, c: &Command, catalog: &'a Catalog) -> Option<&'a str> {
    let card = array(&s.extra["hands"][s.active.to_string()])
        .iter()
        .find(|v| v["id"].as_str() == c.card_id.as_deref());
    match c.kind.as_str() {
        "summon" => Some(
            if s.extra.get("mode") == Some(&json!("shrine")) || c.ultimate == Some(true) {
                "ultimate"
            } else {
                "normal"
            },
        ),
        "extra-summon" => Some(if c.ultimate == Some(false) {
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
pub fn actions(
    s: &State,
    actor: usize,
    catalog: &Catalog,
    queries: &mut Queries,
) -> Result<Vec<Action>, String> {
    let mut candidates = vec![];
    let simple = |c: Command| action(&c.kind, c.clone(), vec![]);
    if s.extra.contains_key("winner") {
        return Ok(vec![]);
    }
    if let Some(r) = s.pending.first() {
        let k = r.kind.as_str();
        candidates.push(action(
            k,
            if k == "hit-pull" {
                Command {
                    mode: Some("pull".into()),
                    ..Command::new("react")
                }
            } else {
                Command::new("react")
            },
            match k {
                "hit-pull" => vec![],
                "bounce" | "hut-spawn" => vec![Step::new("point")],
                _ => vec![Step::target("any", "targetId", false)],
            },
        ));
        candidates.push(simple(Command::new("react")));
    } else if s.phase == "shrine-draft" {
        for k in
            array(&s.extra.get("shrineDraft").unwrap_or(&Value::Null)["offers"][actor.to_string()])
        {
            let c = Command {
                player: Some(actor),
                shrine_kind: Some(kind(k)),
                ..Command::new("choose-shrine")
            };
            if k == "s9" {
                for p in ["odd", "even"] {
                    candidates.push(simple(Command {
                        parity: Some(p.into()),
                        ..c.clone()
                    }));
                }
            } else {
                candidates.push(simple(c));
            }
        }
    } else if let Some(offer) = s.extra.get("summonOffer") {
        for i in 0..array(&offer["groups"]).len() {
            for j in i + 1..array(&offer["groups"]).len() {
                candidates.push(simple(Command {
                    offer_indices: Some(vec![i as f64, j as f64]),
                    ..Command::new("choose-summons")
                }));
            }
        }
    } else {
        for t in ["begin", "end", "skip-synthesis", "finish-shrine-setup"] {
            candidates.push(simple(Command::new(t)));
        }
        for t in ["summon", "extra-summon"] {
            for ultimate in [false, true] {
                candidates.push(simple(Command {
                    ultimate: Some(ultimate),
                    ..Command::new(t)
                }));
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
                Command::new("clock"),
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
                        Command {
                            recipe_id: Some(text(&r["id"]).into()),
                            ..Command::new("synthesize")
                        },
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
        if allowed(s, actor, &a, catalog, queries)? {
            if can_choose && let Some(p) = pool(s, &a.command, catalog) {
                a.chosen = catalog.pools[p].clone();
            }
            result.push(a);
        }
    }
    Ok(result)
}
