//! 主动技能复用移动、攻击、伤害与部署入口。继承能力只投影自己的次数和蓄力。
use crate::damage::{Source, area_damage, damage, kill, lower_max, protected};
use crate::geometry::{
    Target, attack_path, can_place, find_target, movement_path, point_target, ring, square,
    targets, top_target,
};
use crate::model::{Catalog, Command, Failure, Point, State, Unit, ensure, extra_number, number};
use crate::movement::{actor, choose_skill, finish, reserve};
use crate::resolution::{Resolution, add_effect, deploy_unit, template};
use crate::shrines::point;
use crate::stats::{prune_siphons, stats};
use serde_json::{Value, json};
use std::collections::HashMap;

pub fn unit(s: &State, id: Option<&str>) -> Result<Unit, Failure> {
    s.unit(id.unwrap_or(""))
        .map(Unit::fork)
        .ok_or(Failure::Invalid("请选择仍在场上的随从。"))
}
pub fn target(s: &State, id: Option<&str>) -> Result<Target, Failure> {
    find_target(s, id.unwrap_or("")).ok_or(Failure::Invalid("请选择有效的目标。"))
}
fn friend(s: &State, u: &Unit, id: Option<&str>, range: f64) -> Result<Unit, Failure> {
    let v = unit(s, id)?;
    ensure(v.side() == u.owner, "请选择友方随从。")?;
    ensure(
        attack_path(s, u, &Target::from(&v), range, None, false).is_some(),
        "目标不在技能范围内。",
    )?;
    Ok(v)
}
fn enemy(
    s: &State,
    u: &Unit,
    id: Option<&str>,
    range: f64,
    global: bool,
    catalog: &Catalog,
) -> Result<Target, Failure> {
    let t = target(s, id)?;
    ensure(
        t.unit
            .as_ref()
            .is_some_and(|v| catalog.by_kind(&v.kind).landmark.is_none() && v.side() != u.owner)
            && top_target(s, &t, catalog),
        "请选择敌方或中立的栈顶随从。",
    )?;
    ensure(
        global || attack_path(s, u, &t, range, None, false).is_some(),
        "目标不在技能射程内。",
    )?;
    Ok(t)
}
fn move_to(s: &mut State, id: &str, to: Point) {
    if let Some(u) = s.unit_mut(id) {
        u.x = to.x;
        u.y = to.y;
    }
}
fn extra(s: &mut State, id: &str, key: &str, value: Value) {
    if let Some(u) = s.unit_mut(id) {
        match key {
            "charge" => u.reserve.charge = number(&value),
            "readyCharge" => u.reserve.ready_charge = number(&value),
            "onceUsed" => u.once_used = value == true,
            "freeUsed" => u.free_used = number(&value),
            _ => {
                u.extra.insert(key.into(), value);
            }
        }
    }
}
fn delayed_attack(s: &mut State, v: &Unit, owner: usize, source: &str) {
    add_effect(
        s,
        &v.id,
        "attack",
        owner,
        2.0,
        Some(10.0),
        Some(source),
        false,
    );
    let e = s.unit_mut(&v.id).unwrap().effects.last_mut().unwrap();
    e.from += 2.0;
    e.until += 2.0;
}
fn prepare_hook(
    s: &State,
    u: &Unit,
    c: &Command,
    range: f64,
    catalog: &Catalog,
) -> Result<(Target, Point), Failure> {
    let t = enemy(s, u, c.target_id.as_deref(), range, false, catalog)?;
    let to = hook_destination(s, u, c, range, &t, catalog)?;
    Ok((t, to))
}
fn hook_destination(
    s: &State,
    u: &Unit,
    c: &Command,
    range: f64,
    t: &Target,
    catalog: &Catalog,
) -> Result<Point, Failure> {
    let to = point(c)?;
    let v = t.unit.as_ref().unwrap();
    ensure(!v.has("5"), "大肉比不能被钩子牵引。")?;
    let mut moved = v.fork();
    moved.x = to.x;
    moved.y = to.y;
    ensure(
        t.at != to
            && can_place(s, v, to, catalog)
            && attack_path(s, u, &Target::from(&moved), range, None, false).is_some(),
        "牵引落点必须合法且在钩子射程内。",
    )?;
    Ok(to)
}
struct Sacrifice {
    victim: Unit,
    same: bool,
    summon: bool,
    target: Option<Target>,
    amount: f64,
}
fn prepare_sacrifice(
    s: &State,
    u: &Unit,
    c: &Command,
    range: f64,
    catalog: &Catalog,
) -> Result<Sacrifice, Failure> {
    let v = friend(s, u, c.target_id.as_deref(), range)?;
    ensure(
        v.id != u.id && !v.kind.is("u25"),
        "不可献祭自身或克隆军团。",
    )?;
    if u.max_hp < catalog.rule("/sacrificeMaxHpCost") {
        return Err(Failure::InvalidOwned(format!(
            "生命上限不足{}。",
            catalog.rule("/sacrificeMaxHpCost")
        )));
    }
    let same = v.kind.is("14");
    let summon = c.mode.as_deref() == Some("summon");
    ensure(!summon || same, "只有献祭另一枚献祭炮才能直接换取召唤。")?;
    ensure(
        summon
            || c.column
                .is_some_and(|x| x.fract() == 0.0 && (1.0..=9.0).contains(&x)),
        "请选择一列。",
    )?;
    let mut choices: Vec<_> = targets(s)
        .into_iter()
        .filter(|t| {
            t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != u.owner
                && t.footprint().iter().any(|p| {
                    Some(p.x) == c.column && if u.owner == 1 { p.y >= u.y } else { p.y <= u.y }
                })
                && attack_path(s, u, t, range, None, false).is_some()
                && top_target(s, t, catalog)
        })
        .collect();
    choices.sort_by(|a, b| {
        if u.owner == 1 {
            a.at.y.total_cmp(&b.at.y)
        } else {
            b.at.y.total_cmp(&a.at.y)
        }
    });
    ensure(!choices.is_empty() || same, "这一列没有射程内的敌方目标。")?;
    Ok(Sacrifice {
        amount: stats(s, &v, catalog).attack,
        victim: v,
        same,
        summon,
        target: choices.into_iter().next(),
    })
}
fn prepare_revival(
    s: &State,
    u: &Unit,
    c: &Command,
    range: f64,
    catalog: &Catalog,
) -> Result<(Value, Point), Failure> {
    let record = revival_record(s, u, c.death_id.as_deref())?;
    let to = revival_destination(s, u, c, range, &record, catalog)?;
    Ok((record, to))
}
fn revival_record(s: &State, u: &Unit, id: Option<&str>) -> Result<Value, Failure> {
    let record = s.extra["deaths"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"].as_str() == id);
    ensure(
        record.is_some_and(|r| {
            r["kind"] != "grave"
                && number(&r["owner"]) == u.owner as f64
                && r["revived"] != true
                && number(&r["ply"]) < s.ply
                && number(&r["ply"]) >= s.ply - 4.0
        }),
        "只能选择前两个己方回合窗口内尚未复活的友方阵亡记录。",
    )?;
    Ok(record.unwrap().clone())
}
fn revival_destination(
    s: &State,
    u: &Unit,
    c: &Command,
    range: f64,
    record: &Value,
    catalog: &Catalog,
) -> Result<Point, Failure> {
    let to = point(c)?;
    let key = crate::preparation::kind(&record["kind"]).key();
    let ghost = template(s, &key, u.owner, to, "preview", catalog);
    ensure(
        can_place(s, &ghost, to, catalog)
            && attack_path(s, u, &point_target(to), range, None, false).is_some(),
        "复活位置须在射程内合法空格。",
    )?;
    Ok(to)
}
pub struct SkillInspection {
    view: Option<State>,
    unit: Unit,
    key: String,
    range: f64,
    records: HashMap<Option<String>, Result<Value, Failure>>,
    targets: HashMap<Option<String>, Result<Target, Failure>>,
}
impl SkillInspection {
    pub fn inspect(&mut self, c: &Command, catalog: &Catalog) -> Result<(), Failure> {
        let Some(s) = &self.view else {
            return Ok(());
        };
        match self.key.as_str() {
            "u19" => {
                let record = self
                    .records
                    .entry(c.death_id.clone())
                    .or_insert_with(|| revival_record(s, &self.unit, c.death_id.as_deref()))
                    .as_ref()
                    .map_err(Clone::clone)?;
                revival_destination(s, &self.unit, c, self.range, record, catalog)?;
            }
            "7" => {
                let target = self
                    .targets
                    .entry(c.target_id.clone())
                    .or_insert_with(|| {
                        enemy(
                            s,
                            &self.unit,
                            c.target_id.as_deref(),
                            self.range,
                            false,
                            catalog,
                        )
                    })
                    .as_ref()
                    .map_err(Clone::clone)?;
                hook_destination(s, &self.unit, c, self.range, target, catalog)?;
            }
            "14" => {
                prepare_sacrifice(s, &self.unit, c, self.range, catalog)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}
/// 冰痕正式入口在公共资格之前读取坐标，预检须保持错误顺序。
pub fn skill_coordinate(s: &State, c: &Command) -> Result<(), Failure> {
    let raw = s.unit(c.unit_id.as_deref().unwrap_or(""));
    let kind = c.ability.as_ref().or_else(|| raw.map(|u| &u.kind));
    if kind.is_some_and(|k| k.is("u24")) {
        let u = unit(s, c.unit_id.as_deref())?;
        ensure(u.has("u24"), "该棋子没有选定的技能。")?;
        point(c)?;
    }
    Ok(())
}
/// 只投影当前施法者，保留冷字段共享；所有技能复用正式 begin_skill 公共资格。
pub fn prepare_inspection(
    s: &State,
    c: &Command,
    catalog: &Catalog,
) -> Result<SkillInspection, Failure> {
    let raw = unit(s, c.unit_id.as_deref())?;
    let k = c.ability.as_ref().unwrap_or(&raw.kind);
    let key = k.key();
    ensure(raw.has(&key), "该棋子没有选定的技能。")?;
    let mut view = s.fork();
    project_resources(&mut view, &raw, k, catalog);
    let (u, range) = begin_skill(&mut view, c, &key, catalog)?;
    Ok(SkillInspection {
        view: ["u19", "7", "14"].contains(&key.as_str()).then_some(view),
        unit: u,
        key,
        range,
        records: HashMap::new(),
        targets: HashMap::new(),
    })
}
fn project_resources(s: &mut State, raw: &Unit, k: &crate::model::Kind, catalog: &Catalog) {
    if *k == raw.kind {
        return;
    }
    let r = reserve(raw, k, catalog);
    let key = k.key();
    let u = s.unit_mut(&raw.id).unwrap();
    u.reserve = r;
    let usage = raw.ability_usage.as_ref().and_then(|v| v.get(&key));
    u.once_used = usage.is_some_and(|v| v.once);
    u.free_used = usage.map(|v| v.free).unwrap_or(-1.0);
}
pub fn skill(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let raw = unit(s, c.unit_id.as_deref())?;
    let k = c.ability.as_ref().unwrap_or(&raw.kind).clone();
    let key = k.key();
    ensure(raw.has(&key), "该棋子没有选定的技能。")?;
    let inherited = k != raw.kind;
    let native = reserve(&raw, &raw.kind, catalog);
    project_resources(s, &raw, &k, catalog);
    let mut facts = json!({"action":match key.as_str(){"5"=>"quake","7"|"u23"=>"pull","21"=>"rush","u6"=>"cross","u14"=>"siphon","u24"=>"ice-mark",_=>"buff"},"ability":k,"actor":raw.actor_event()});
    if key == "5" {
        facts["area"] = json!(
            crate::lifecycle::all_cells()
                .filter(|p| ring(&raw, &[*p]))
                .collect::<Vec<_>>()
        );
    }
    if key == "u6" {
        facts["area"] = json!(
            crate::lifecycle::all_cells()
                .filter(|p| square(&raw, *p) && (Some(p.x) == c.x || Some(p.y) == c.y))
                .collect::<Vec<_>>()
        );
    }
    if key == "u24" {
        facts["area"] = json!([point(c)?]);
    }
    let previous = ctx.facts.replace(facts);
    let mut a = c.target_id.as_deref().and_then(|id| find_target(s, id));
    let b = c.second_id.as_deref().and_then(|id| find_target(s, id));
    let links: Vec<_> = s.siphons.iter().map(|l| l["id"].clone()).collect();
    resolve_skill(s, c, &key, catalog, ctx)?;
    if inherited && let Some(u) = s.unit_mut(&raw.id) {
        let usage = crate::model::AbilityUsage {
            once: u.once_used,
            free: u.free_used,
        };
        let charge = u.reserve;
        u.ability_usage
            .get_or_insert_with(Default::default)
            .insert(key.clone(), usage);
        u.ability_charges
            .get_or_insert_with(Default::default)
            .insert(key.clone(), charge);
        u.reserve = native;
        u.once_used = raw.once_used;
        u.free_used = raw.free_used;
    }
    if let Some(t) = a.as_mut()
        && let Some(u) = s.unit(&t.id)
    {
        t.unit = Some(u.clone());
    }
    let link_created = s.siphons.iter().any(|l| !links.contains(&l["id"]));
    if let Some(e) = s.events.last_mut().filter(|e| e["type"] == "skill") {
        if key == "u14" {
            e["stage"] = json!(if link_created { "apply" } else { "blocked" });
            if let Some(a) = a {
                e["actor"] = a.actor();
                e["from"] = json!(a.at);
            }
            if let Some(b) = b {
                e["subject"] = b.actor();
                e["to"] = json!(b.at);
            }
            if !link_created {
                for field in ["to", "actor", "subject"] {
                    e.as_object_mut().unwrap().remove(field);
                }
            }
        } else if key == "u24" {
            e["to"] = json!(point(c)?);
            e.as_object_mut().unwrap().remove("subject");
        } else if (key == "u7" || key == "u21")
            && let Some(a) = a
        {
            if key == "u7" && a.unit.as_ref().is_some_and(|u| u.size == 1.0) {
                e["stage"] = json!("blocked");
            }
            e["to"] = json!(a.at);
            e["subject"] = a.actor();
        }
    }
    ctx.facts = previous;
    Ok(())
}
fn begin_skill(
    s: &mut State,
    c: &Command,
    k: &str,
    catalog: &Catalog,
) -> Result<(Unit, f64), Failure> {
    let u = if k == "u7" {
        unit(s, c.unit_id.as_deref())?
    } else {
        actor(s, c, catalog)?.fork()
    };
    ensure(!u.silenced, "沉默已移除此随从的技能。")?;
    ensure(
        !s.effect(&u, "freeze") && !s.effect(&u, "stun"),
        "冻结或眩晕中不能施放技能。",
    )?;
    if k == "u14" {
        ensure(
            extra_number(&u, "freeUsed") != s.ply,
            "本回合已经使用虹吸。",
        )?;
    }
    if k != "u7" {
        choose_skill(s, &u.id, catalog)?;
    } else {
        ensure(
            extra_number(&u, "freeUsed") != s.ply + u.offset,
            "本回合免费技能已使用。",
        )?;
    }
    let u = s.unit(&u.id).unwrap().fork();
    let range = stats(s, &u, catalog).range;
    Ok((u, range))
}
fn resolve_skill(
    s: &mut State,
    c: &Command,
    k: &str,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let (u, range) = begin_skill(s, c, k, catalog)?;
    let source = Source::new(&u, "skill");
    match k {
        "5" => {
            let victims: Vec<_> = targets(s)
                .into_iter()
                .filter(|t| {
                    t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != u.owner
                        && ring(&u, &t.footprint())
                        && top_target(s, t, catalog)
                })
                .collect();
            area_damage(
                s,
                &victims,
                |p| {
                    if ring(&u, &[p]) {
                        catalog.rule("/giantAreaDamage")
                    } else {
                        0.0
                    }
                },
                &source,
                catalog,
                ctx,
            )?;
        }
        "6" => {
            for v in s.units.clone() {
                if v.side() == u.owner
                    && !v.any(&["10", "s7"])
                    && attack_path(s, &u, &Target::from(&v), range, None, false).is_some()
                {
                    delayed_attack(s, &v, u.owner, &u.id);
                }
            }
        }
        "7" => {
            let (t, to) = prepare_hook(s, &u, c, range, catalog)?;
            if !protected(s, &t, &source, catalog, ctx)? {
                ctx.emit(s,json!({"type":"move","stage":"trigger","from":t.actor(),"to":to,"unitId":t.id,"owner":t.owner}),None);
                move_to(s, &t.id, to);
            }
        }
        "14" => {
            let Sacrifice {
                victim: v,
                same,
                summon,
                target,
                amount,
            } = prepare_sacrifice(s, &u, c, range, catalog)?;
            lower_max(s, &u.id, catalog.rule("/sacrificeMaxHpCost"), catalog, ctx)?;
            let mut sacrifice = source.clone();
            sacrifice.kind = "sacrifice";
            kill(s, &v, &sacrifice, catalog, ctx)?;
            if !summon && let Some(t) = target {
                damage(s, &t, amount, &source, catalog, ctx)?;
            }
            if same {
                s.summon_slots += 1.0;
                ctx.emit(s,json!({"type":"skill","to":u.actor_event(),"owner":u.owner,"text":"献祭同类 · 本回合额外召唤+1"}),Some("献祭炮献祭同类，获得一次仅本回合可用的召唤机会（可付2人头升级）。".into()));
            }
        }
        "19" => {
            let to = point(c)?;
            let ghost = template(s, "wall", u.owner, to, "preview", catalog);
            ensure(
                can_place(s, &ghost, to, catalog)
                    && attack_path(s, &u, &point_target(to), range, None, false).is_some(),
                "路障须放在射程内合法空格。",
            )?;
            let id = format!("u{}", s.serial);
            deploy_unit(s, "wall", u.owner, to, None, false, catalog, ctx);
            extra(s, &id, "expiresAt", json!(s.ply + 2.0));
        }
        "21" => {
            ensure(
                extra_number(&u, "readyCharge") >= 2.0 && u.reserve.charge_type.as_str() == "skill",
                "回合开始需要已经持有两层技能蓄力。",
            )?;
            ensure(u.hp > 10.0, "突袭扣10血后必须存活。")?;
            let to = point(c)?;
            let t = target(s, c.target_id.as_deref())?;
            let route = movement_path(s, &u, to, 6.0, false, catalog)
                .ok_or(Failure::Invalid("落点必须在6格可达范围内。"))?;
            let mut moved = u.clone();
            moved.x = to.x;
            moved.y = to.y;
            ensure(
                t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != u.owner
                    && attack_path(s, &moved, &t, range, None, false).is_some(),
                "落点无法攻击所选敌方。",
            )?;
            s.unit_mut(&u.id).unwrap().hp -= 10.0;
            ctx.emit(s,json!({"type":"move","from":u.actor_event(),"to":to,"path":route,"unitId":u.id,"owner":u.owner,"text":"神行突袭"}),None);
            move_to(s, &u.id, to);
            extra(s, &u.id, "charge", json!(0));
            extra(s, &u.id, "readyCharge", json!(0));
            let moved = s.unit(&u.id).unwrap().clone();
            crate::combat::perform(
                s,
                &moved,
                &t,
                &crate::combat::Options {
                    reactive: true,
                    ..Default::default()
                },
                catalog,
                ctx,
            )?;
        }
        "u6" => {
            ensure(
                !u.once_used
                    && extra_number(&u, "readyCharge") >= 1.0
                    && u.reserve.charge_type.as_str() == "skill",
                "十字浩劫需要前一回合蓄力，且一生只能发动一次。",
            )?;
            let p = point(c)?;
            ensure(square(&u, p), "交点必须位于自身11×11区域。")?;
            let hit = |v: Point| square(&u, v) && (v.x == p.x || v.y == p.y);
            let victims: Vec<_> = targets(s)
                .into_iter()
                .filter(|t| {
                    t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != u.owner
                        && top_target(s, t, catalog)
                        && t.footprint().into_iter().any(&hit)
                })
                .collect();
            ctx.emit(
                s,
                json!({"type":"skill","to":p,"owner":u.owner,"text":"十字浩劫","ultimate":true}),
                None,
            );
            area_damage(
                s,
                &victims,
                |v| {
                    if hit(v) {
                        if v == p { 40.0 } else { 20.0 }
                    } else {
                        0.0
                    }
                },
                &source,
                catalog,
                ctx,
            )?;
            extra(s, &u.id, "onceUsed", json!(true));
            extra(s, &u.id, "charge", json!(0));
            extra(s, &u.id, "readyCharge", json!(0));
        }
        "u7" => {
            ensure(!u.once_used, "巨大化一生只能用一次。")?;
            let v = unit(s, c.target_id.as_deref())?;
            let at = if c.x.is_none() && c.y.is_none() {
                v.at()
            } else {
                point(c)?
            };
            let mut giant = v.clone();
            giant.size = 2.0;
            ensure(
                v.id != u.id
                    && v.size == 1.0
                    && catalog.by_kind(&v.kind).landmark.is_none()
                    && [
                        v.at(),
                        Point {
                            x: v.x - 1.0,
                            y: v.y,
                        },
                        Point {
                            x: v.x,
                            y: v.y - 1.0,
                        },
                        Point {
                            x: v.x - 1.0,
                            y: v.y - 1.0,
                        },
                    ]
                    .contains(&at)
                    && can_place(s, &giant, at, catalog),
                "请选择有合法扩展方向的单格棋子；2×2必须包含原格，且不能与棋子、地标或基地重合。",
            )?;
            ensure(
                u.hp > 10.0 && u.max_hp > 10.0 && stats(s, &u, catalog).attack >= 10.0,
                "发动需要支付10攻击和10生命并存活。",
            )?;
            let caster = s.unit_mut(&u.id).unwrap();
            caster.hp -= 10.0;
            caster.attack_bonus = extra_number(&u, "attackBonus") - 10.0;
            if !protected(s, &Target::from(&v), &source, catalog, ctx)? {
                let v = s.unit_mut(&v.id).unwrap();
                v.x = at.x;
                v.y = at.y;
                v.size = 2.0;
                v.max_hp += 5.0;
                v.hp += 5.0;
            }
            extra(s, &u.id, "onceUsed", json!(true));
        }
        "u14" => {
            let a = target(s, c.target_id.as_deref())?;
            let b = target(s, c.second_id.as_deref())?;
            ensure(a.id != b.id, "虹吸的两个端点不能相同。")?;
            ensure(
                attack_path(s, &u, &a, range, None, false).is_some()
                    && attack_path(s, &u, &b, range, None, false).is_some(),
                "虹吸两个目标均须在射程内。",
            )?;
            ensure(
                s.siphons.iter().filter(|l| l["sourceId"] == u.id).count() < 3,
                "最多存在3条虹吸。",
            )?;
            ensure(
                (a.owner == u.owner || top_target(s, &a, catalog))
                    && (b.owner == u.owner || top_target(s, &b, catalog)),
                "敌方叠放目标只能选栈顶。",
            )?;
            if !protected(s, &a, &source, catalog, ctx)?
                && !protected(s, &b, &source, catalog, ctx)?
            {
                s.siphons.push(json!({"id":format!("link{}",s.serial),"sourceId":u.id,"owner":u.owner,"fromId":a.id,"toId":b.id}));
                s.serial += 1;
            }
        }
        "u19" => {
            let (record, to) = prepare_revival(s, &u, c, range, catalog)?;
            let key = crate::preparation::kind(&record["kind"]).key();
            let group = if key == "u25" {
                let id = format!("revived-group{}", s.serial);
                s.serial += 1;
                Some(id)
            } else {
                None
            };
            deploy_unit(s, &key, u.owner, to, group.as_deref(), false, catalog, ctx);
            s.extra
                .get_mut("deaths")
                .unwrap()
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|r| r["id"] == record["id"])
                .unwrap()["revived"] = json!(true);
            lower_max(s, &u.id, 20.0, catalog, ctx)?;
        }
        "u21" => {
            let v = friend(s, &u, c.target_id.as_deref(), range)?;
            delayed_attack(s, &v, u.owner, &u.id);
        }
        "u23" => {
            ensure(
                u.extra.contains_key("hookReadyAt")
                    && extra_number(&u, "hookReadyAt") <= s.ply + u.offset
                    && extra_number(&u, "hookExpiresAt") > s.ply + u.offset,
                "仅击杀后的下个己方回合可使用超级钩子。",
            )?;
            let t = enemy(s, &u, c.target_id.as_deref(), range, true, catalog)?;
            let v = t.unit.as_ref().unwrap();
            let to = Point {
                x: u.x,
                y: if u.owner == 1 {
                    u.y + u.size
                } else {
                    u.y - v.size
                },
            };
            ensure(!v.has("5"), "大肉比不能被钩子牵引。")?;
            ensure(can_place(s, v, to, catalog), "身前没有合法落位。")?;
            if !protected(s, &t, &source, catalog, ctx)? {
                ctx.emit(s,json!({"type":"move","stage":"trigger","from":v.actor_event(),"to":to,"unitId":v.id,"owner":v.owner,"text":"超级牵引"}),None);
                move_to(s, &v.id, to);
            }
            let u = s.unit_mut(&u.id).unwrap();
            u.extra.remove("hookReadyAt");
            u.extra.remove("hookExpiresAt");
        }
        "u24" => {
            let to = point(c)?;
            ensure(square(&u, to), "标记须位于自身11×11区域。")?;
            let mark = json!({"id":format!("ice{}",s.serial),"sourceId":u.id,"owner":u.owner,"x":to.x,"y":to.y,"due":s.ply+u.offset+2.0});
            s.serial += 1;
            s.extra
                .get_mut("iceMarks")
                .unwrap()
                .as_array_mut()
                .unwrap()
                .push(mark);
        }
        _ => {
            return Err(Failure::Invalid(
                "此棋子没有可主动使用的技能；被动能力由引擎自动结算。",
            ));
        }
    }
    if k == "u14" {
        extra(s, &u.id, "freeUsed", json!(s.ply));
    }
    if k == "u7" {
        extra(s, &u.id, "freeUsed", json!(s.ply + u.offset));
    } else if let Some(v) = s.unit_mut(&u.id) {
        finish(v);
    }
    let current = s.unit(&u.id).unwrap_or(&u);
    ctx.emit(s,json!({"type":"skill","to":current.actor_event(),"owner":u.owner,"text":catalog[k].skill.as_deref().unwrap_or("技能"),"ultimate":catalog.by_kind(&u.kind).tier!="normal"}),Some(format!("{}施放技能",catalog.by_kind(&u.kind).name)));
    prune_siphons(s, catalog);
    Ok(())
}
