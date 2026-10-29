//! 引擎命令说明，对应 TS commands/options；只表达规则资格与参数域。
use crate::inspection::Queries;
use crate::model::{Catalog, Command, Failure, Point, State, Unit, extra_number};
use crate::movement::{charge_kind, reserve};
use crate::preparation::kind;
use crate::stats::Stats;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub struct Step {
    pub kind: &'static str,
    pub field: &'static str,
    pub relation: &'static str,
    pub unit_only: bool,
}
impl Step {
    pub const fn new(kind: &'static str) -> Self {
        Self {
            kind,
            field: "targetId",
            relation: "any",
            unit_only: false,
        }
    }
    const fn target(relation: &'static str, field: &'static str, unit_only: bool) -> Self {
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
    pub id: Cow<'static, str>,
    pub command: Command,
    pub steps: Cow<'static, [Step]>,
    pub free: bool,
    pub materials: Vec<String>,
    pub chosen: Vec<crate::model::Kind>,
    pub initial_inspection: Option<&'static str>,
}
fn action(id: impl Into<Cow<'static, str>>, c: Command, steps: &'static [Step]) -> Action {
    Action {
        id: id.into(),
        command: c,
        steps: Cow::Borrowed(steps),
        free: false,
        materials: vec![],
        chosen: vec![],
        initial_inspection: None,
    }
}
pub fn array(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
pub fn text(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}
/// 拒绝权威私有字段；预检使用无 RNG 的公开局面，首次概率分支返回 uncertain。
pub fn position(observation: &Value) -> Result<State, String> {
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Compatibility);
    for key in ["seed", "rng", "log", "events", "past", "future", "present"] {
        if observation.get(key).is_some() {
            return Err(format!("public observation contains {key}"));
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
    inspection_status(queries.inspect(s, c, catalog))
}
fn inspection_status(result: Result<(), Failure>) -> Result<&'static str, String> {
    match result {
        Ok(_) => Ok("available"),
        Err(Failure::Uncertain) => Ok("uncertain"),
        Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => Ok("invalid"),
        Err(Failure::Unsupported(r)) => Err(format!("unsupported rule inspection: {r}")),
    }
}
pub fn points(
    s: &State,
    actor: usize,
    c: &Command,
    catalog: &Catalog,
    queries: &mut Queries,
) -> Result<Option<Vec<(Point, &'static str)>>, String> {
    if c.kind != "move" && c.kind != "deploy" && c.kind != "skill" && c.kind != "cast" {
        return Ok(None);
    }
    if !permitted_command(s, actor, c) {
        return Ok(Some(vec![]));
    }
    let domain = match queries.points(s, c, catalog) {
        Ok(Some(domain)) => domain,
        Ok(None) => return Ok(None),
        Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => return Ok(Some(vec![])),
        Err(error) => return Err(format!("point preparation failed: {error:?}")),
    };
    let mut points = vec![];
    for (at, complete) in domain {
        let status = if complete {
            "available"
        } else {
            inspection_status(queries.inspect(
                s,
                &Command {
                    x: Some(at.x),
                    y: Some(at.y),
                    ..c.clone()
                },
                catalog,
            ))?
        };
        if status != "invalid" {
            points.push((at, status));
        }
    }
    Ok(Some(points))
}
pub(crate) fn allowed(
    s: &State,
    actor: usize,
    a: &mut Action,
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
        let status = inspect(s, actor, &a.command, catalog, queries)?;
        a.initial_inspection = Some(status);
        return Ok(status != "invalid");
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
    let stats = queries.attributes(s, u, catalog);
    let ability = c.ability.clone().unwrap_or(u.kind.clone());
    let r = reserve(u, &ability, catalog);
    let borrowed = ability != u.kind;
    let usage = u
        .ability_usage
        .as_ref()
        .and_then(|m| m.get(ability.key_ref().as_ref()));
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
pub(crate) fn unit_actions(s: &State, u: &Unit, catalog: &Catalog, st: &Stats) -> Vec<Action> {
    if catalog.by_kind(&u.kind).landmark.is_some() && u.hp <= 0.0 {
        return vec![];
    }
    let mut result = vec![];
    let unit_id: Rc<str> = Rc::from(u.id.as_str());
    let base = |t| Command {
        unit_id: Some(Rc::clone(&unit_id)),
        ..Command::new(t)
    };
    if st.movement > 0.0 {
        result.push(action(
            "move",
            base("move"),
            const { &[Step::new("point")] },
        ));
    }
    if st.actions > 0.0 && (u.silenced || !u.has("firelord")) {
        let c = base("attack");
        if u.signed() {
            result.push(action(
                "attack",
                {
                    let mut command = c.clone();
                    command.mode = Some("damage".into());
                    command
                },
                const { &[Step::target("any", "targetId", false)] },
            ));
            result.push(action(
                "heal",
                {
                    let mut command = c.clone();
                    command.mode = Some("heal".into());
                    command
                },
                const { &[Step::target("any", "targetId", true)] },
            ));
        } else {
            result.push(action(
                "attack",
                c.clone(),
                const { &[Step::target("any", "targetId", false)] },
            ));
        }
        if crate::combat::healing_attack(u, catalog) {
            result.push(action(
                "self-heal",
                {
                    let mut command = c.clone();
                    command.target_id = Some(Rc::clone(&unit_id));
                    command.mode = Some("heal".into());
                    command
                },
                &[],
            ));
        }
        if u.weapon("u28") {
            result.push(action(
                "attack-path",
                {
                    let mut command = c.clone();
                    command.mode = Some("damage".into());
                    command.path = Some(vec![]);
                    command
                },
                const { &[Step::new("path")] },
            ));
        }
    }
    let d = catalog.by_kind(&u.kind);
    let ordinal = u
        .kind
        .key_ref()
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
            const { &[Step::target("enemy", "targetId", false)] },
        ));
    }
    if ["move", "attack"].contains(&u.mode.as_str()) {
        result.push(action("finish", base("finish-mode"), &[]));
    }
    ability_actions(&mut result, u, &u.kind, &unit_id, catalog);
    if !u.silenced {
        for k in u.traits.iter().flatten() {
            let start = result.len();
            // 继承只追加技能与蓄力，不构造随后丢弃的通用动作和完整实体投影。
            ability_actions(&mut result, u, k, &unit_id, catalog);
            for a in &mut result[start..] {
                a.id = format!("{}:{}", a.id, k.key()).into();
                a.command.ability = Some(k.clone());
            }
        }
    }
    result
}
fn ability_actions(
    result: &mut Vec<Action>,
    u: &Unit,
    kind: &crate::model::Kind,
    unit_id: &Rc<str>,
    catalog: &Catalog,
) {
    let d = catalog.by_kind(kind);
    if d.landmark.is_some() && u.hp <= 0.0 {
        return;
    }
    let base = |t| Command {
        unit_id: Some(Rc::clone(unit_id)),
        ..Command::new(t)
    };
    for (yes, id, mode) in [
        (
            d.movement > 0.0 && d.movement.fract() != 0.0,
            "charge-move",
            "move",
        ),
        (d.actions == 0.5, "charge-attack", "attack"),
        (
            !u.silenced && ["4", "15", "u2"].iter().any(|k| kind.is(k)),
            "charge-attack",
            "attack",
        ),
        (
            !u.silenced && ["21", "u6"].iter().any(|k| kind.is(k)),
            "charge-skill",
            "skill",
        ),
    ] {
        if yes {
            result.push(action(
                id,
                {
                    let mut command = base("charge");
                    command.mode = Some(mode.into());
                    command
                },
                &[],
            ));
        }
    }
    if !u.silenced {
        let c = base("skill");
        let mut add =
            |id: &'static str, c, steps: &'static [Step]| result.push(action(id, c, steps));
        match kind.key_ref().as_ref() {
            "5" => add("stomp", c, &[]),
            "6" => add("buff", c, &[]),
            "7" => add(
                "hook",
                c,
                const { &[Step::target("enemy", "targetId", true), Step::new("point")] },
            ),
            "14" => {
                add(
                    "sacrifice-summon",
                    {
                        let mut command = c.clone();
                        command.mode = Some("summon".into());
                        command
                    },
                    const { &[Step::target("friend", "targetId", true)] },
                );
                add(
                    "sacrifice",
                    c,
                    const {
                        &[
                            Step::target("friend", "targetId", true),
                            Step::new("column"),
                        ]
                    },
                );
            }
            "19" => add("wall", c, const { &[Step::new("point")] }),
            "21" => add(
                "dash",
                c,
                const { &[Step::new("point"), Step::target("enemy", "targetId", false)] },
            ),
            "u6" => add("cross", c, const { &[Step::new("point")] }),
            "u7" => {
                let mut a = action(
                    "giant",
                    c,
                    const { &[Step::target("any", "targetId", true), Step::new("point")] },
                );
                a.free = true;
                result.push(a);
            }
            "u14" => add(
                "siphon",
                c,
                const {
                    &[
                        Step::target("any", "targetId", false),
                        Step::target("any", "secondId", false),
                    ]
                },
            ),
            "u19" => add(
                "revive",
                c,
                const { &[Step::new("death"), Step::new("point")] },
            ),
            "u21" => add(
                "single-buff",
                c,
                const { &[Step::target("friend", "targetId", true)] },
            ),
            "u23" => add(
                "superhook",
                c,
                const { &[Step::target("enemy", "targetId", true)] },
            ),
            "u24" => add("ice-mark", c, const { &[Step::new("point")] }),
            _ => {}
        }
    }
}
pub(crate) fn card_actions(s: &State, card: &Value, catalog: &Catalog) -> Vec<Action> {
    let k = kind(&card["kind"]);
    let d = catalog.by_kind(&k);
    let card_id: Rc<str> = Rc::from(text(&card["id"]));
    let base = |t| Command {
        card_id: Some(Rc::clone(&card_id)),
        ..Command::new(t)
    };
    let mut result = vec![];
    if d.spell.is_none() && d.weapon.is_none() && !d.aura {
        result.push(action(
            "deploy",
            {
                let mut command = base("deploy");
                command.charge = Some(false);
                command
            },
            const { &[Step::new("point")] },
        ));
        if k.is("1") {
            result.push(action(
                "deploy-charge",
                {
                    let mut command = base("deploy");
                    command.charge = Some(true);
                    command
                },
                const { &[Step::new("point")] },
            ));
        }
    } else if d.aura {
        result.push(action("activate-aura", base("activate-aura"), &[]));
    } else if d.weapon.is_some() {
        result.push(action(
            "equip",
            base("equip"),
            const { &[Step::target("friend", "targetId", true)] },
        ));
    } else if k.is("8") {
        result.push(action(
            "blast",
            base("cast"),
            const { &[Step::new("point")] },
        ));
    } else if k.is("25") {
        result.push(action(
            "reforge-one",
            {
                let mut command = base("cast");
                command.mode = Some("single".into());
                command
            },
            &[],
        ));
        result.push(action(
            "reforge-two",
            {
                let mut command = base("cast");
                command.mode = Some("double".into());
                command
            },
            const { &[Step::target("friend", "sacrificeIds", true); 2] },
        ));
    } else if k.is("u9") {
        for (id, mode) in [("storm-row", "row"), ("storm-col", "column")] {
            result.push(action(
                id,
                {
                    let mut command = base("cast");
                    command.mode = Some(mode.into());
                    command
                },
                if mode == "row" {
                    const { &[Step::new("row")] }
                } else {
                    const { &[Step::new("column")] }
                },
            ));
        }
    } else {
        result.push(action(
            "cast",
            base("cast"),
            if k.is("u26") {
                const { &[Step::target("any", "targetId", true)] }
            } else {
                const { &[Step::target("friend", "targetId", true)] }
            },
        ));
    }
    for command in crate::preparation::reroll_commands(s, card, catalog) {
        let id: Cow<'static, str> = command
            .unit_id
            .as_deref()
            .map(|id| Cow::Owned(format!("reroll-{id}")))
            .unwrap_or_else(|| "self-reroll".into());
        result.push(action(id, command, &[]));
    }
    result
}
pub(crate) fn pool<'a>(s: &State, c: &Command, catalog: &'a Catalog) -> Option<&'a str> {
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
        "reroll" => card
            .and_then(|v| crate::preparation::summon_pool(v, catalog))
            .map(|pool| {
                if pool == "ultimate" {
                    "ultimate"
                } else {
                    "normal"
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
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Candidates);
    let mut candidates = vec![];
    let simple = |id: &'static str, c: Command| action(id, c, &[]);
    if s.extra.contains_key("winner") {
        return Ok(vec![]);
    }
    if let Some(r) = s.pending.first() {
        let k = r.kind.as_str();
        candidates.push(action(
            k.to_owned(),
            if k == "hit-pull" {
                let mut command = Command::new("react");
                command.mode = Some("pull".into());
                command
            } else {
                Command::new("react")
            },
            match k {
                "hit-pull" => &[],
                "bounce" | "hut-spawn" => const { &[Step::new("point")] },
                _ => const { &[Step::target("any", "targetId", false)] },
            },
        ));
        candidates.push(simple("react", Command::new("react")));
    } else if s.phase == "shrine-draft" {
        for k in
            array(&s.extra.get("shrineDraft").unwrap_or(&Value::Null)["offers"][actor.to_string()])
        {
            let mut c = Command::new("choose-shrine");
            c.player = Some(actor);
            c.shrine_kind = Some(kind(k));
            if k == "s9" {
                for p in ["odd", "even"] {
                    let mut command = c.clone();
                    command.parity = Some(p.into());
                    candidates.push(simple("choose-shrine", command));
                }
            } else {
                candidates.push(simple("choose-shrine", c));
            }
        }
    } else if let Some(offer) = s.extra.get("summonOffer") {
        for i in 0..array(&offer["groups"]).len() {
            for j in i + 1..array(&offer["groups"]).len() {
                let mut command = Command::new("choose-summons");
                command.offer_indices = Some(vec![i as f64, j as f64]);
                candidates.push(simple("choose-summons", command));
            }
        }
    } else {
        for t in ["begin", "end", "skip-synthesis", "finish-shrine-setup"] {
            candidates.push(simple(t, Command::new(t)));
        }
        for t in ["summon", "extra-summon"] {
            for ultimate in [false, true] {
                let mut command = Command::new(t);
                command.ultimate = Some(ultimate);
                candidates.push(simple(t, command));
            }
        }
        for u in s.pieces().filter(|u| u.owner == actor) {
            let giant = u.has("u7");
            if !giant && (actor != s.active || s.phase != "play") {
                continue;
            }
            let stats = queries.attributes(s, u, catalog);
            // 巨大化免操作预算；手牌改判独立生成，连续操作仍保留结束入口。
            if !giant && u.mode == "none" && stats.operations_left <= 0.0 && stats.remaining <= 0.0
            {
                continue;
            }
            candidates.extend(unit_actions(s, u, catalog, &stats));
        }
        if actor == s.active {
            for c in array(&s.extra["hands"][actor.to_string()]) {
                candidates.extend(card_actions(s, c, catalog));
            }
            candidates.push(action(
                "clock",
                Command::new("clock"),
                const { &[Step::target("any", "targetId", true)] },
            ));
            if s.phase == "synthesis" {
                for r in &catalog.recipes {
                    let ids = crate::synthesis::materials(s, r);
                    if ids.len() < 3 {
                        continue;
                    }
                    let mut a = action(
                        format!("synthesize:{}", text(&r["id"])),
                        {
                            let mut command = Command::new("synthesize");
                            command.recipe_id = Some(text(&r["id"]).into());
                            command
                        },
                        if catalog[&kind(&r["result"]).key()].aura {
                            &[]
                        } else {
                            const { &[Step::new("point")] }
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
        if allowed(s, actor, &mut a, catalog, queries)? {
            if can_choose && let Some(p) = pool(s, &a.command, catalog) {
                a.chosen = catalog.pools[p].clone();
            }
            result.push(a);
        }
    }
    Ok(result)
}
