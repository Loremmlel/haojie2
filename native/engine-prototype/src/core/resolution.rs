use crate::geometry::{Target, attack_path};
use crate::model::{Catalog, Failure, Point, State, Unit, number};
use crate::stats::stats;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
/// 一条命令内的临时结算上下文；事实范围服务事件快照，反击集合阻止递归循环。
/// 不写入存档，不持有真实时钟；preview 到随机边界后由调用栈返回 uncertain。
pub struct Resolution {
    pub preview: bool,
    pub retaliations: HashSet<String>,
    pub facts: Option<Value>,
    pub token: usize,
    pub protection: HashMap<String, bool>,
    pub attack_hits: Vec<(String, String, f64)>,
}
impl Resolution {
    pub fn enter(&mut self, actor: Value, subject: Value, action: &str) -> Option<Value> {
        let previous = self.facts.take();
        let mut facts = json!({"actor":actor,"subject":subject,"action":action});
        if let Some(id) = previous.as_ref().and_then(|v| v.get("causeId")) {
            facts["parentId"] = id.clone();
        }
        self.facts = Some(facts);
        previous
    }
    /// 与 TS enrichEvent 共用字段契约；仅捕获坐标/身份事实，不增加额外 serial 或随机消耗。
    pub fn emit(&mut self, s: &mut State, mut event: Value, message: Option<String>) {
        let id = format!("e{}", s.serial);
        s.serial += 1;
        event["id"] = json!(id);
        if let Some(facts) = self.facts.as_mut()
            && facts.get("causeId").is_none()
        {
            facts["causeId"] = json!(id);
        }
        let inferred = event
            .get("to")
            .filter(|p| p.get("id").is_some())
            .cloned()
            .or_else(|| {
                if event["type"] == "move" {
                    event.get("from").filter(|p| p.get("id").is_some()).cloned()
                } else {
                    None
                }
            });
        let mut result = self.facts.clone().unwrap_or_else(|| json!({}));
        result
            .as_object_mut()
            .unwrap()
            .extend(event.as_object().unwrap().clone());
        let actor = event
            .get("actor")
            .cloned()
            .or_else(|| self.facts.as_ref().and_then(|f| f.get("actor").cloned()))
            .or_else(|| {
                if event["type"] == "attack" {
                    event.get("from").filter(|p| p.get("id").is_some()).cloned()
                } else {
                    None
                }
            });
        let subject = event
            .get("subject")
            .cloned()
            .or(inferred)
            .or_else(|| self.facts.as_ref().and_then(|f| f.get("subject").cloned()));
        if let Some(actor) = actor {
            result["actor"] = actor;
        } else {
            result.as_object_mut().unwrap().remove("actor");
        }
        if let Some(subject) = subject {
            result["subject"] = subject;
        } else {
            result.as_object_mut().unwrap().remove("subject");
        }
        for key in ["from", "to"] {
            if let Some(p) = event.get(key) {
                result[key] = json!({"x":p["x"],"y":p["y"]});
            }
        }
        if !["attack", "skill", "move"].contains(&event["type"].as_str().unwrap_or(""))
            && event.get("area").is_none()
        {
            result.as_object_mut().unwrap().remove("area");
        }
        s.events.push(result);
        if let Some(message) = message {
            let log = s.extra.get_mut("log").unwrap().as_array_mut().unwrap();
            log.push(json!(format!("{} · {message}", s.ply)));
            if log.len() > 180 {
                log.remove(0);
            }
        }
    }
}
pub fn faction(owner: usize) -> &'static str {
    if owner == 1 { "苍穹方" } else { "赤焰方" }
}
pub fn reset_unit(ply: f64, u: &mut Unit) {
    u.operations = 0.0;
    u.mode = "none".into();
    u.shots = 0.0;
    u.moves = 0.0;
    u.bonus_attacks = 0.0;
    u.bonus_sequence = false;
    u.attacked.clear();
    u.weapon_first_used = false;
    u.reserve.ready_charge = u.reserve.charge;
    if u.extra.contains_key("extraOperations") {
        u.extra.insert("extraOperations".into(), json!(0));
    }
    if let Some(reserves) = &mut u.ability_charges {
        for reserve in reserves.values_mut() {
            reserve.ready_charge = reserve.charge;
        }
    }
    let offset = u.offset;
    u.effects
        .retain(|e| e.until > ply + if e.global == Some(true) { 0.0 } else { offset });
}
pub fn active_target(s: &State, t: &Target) -> Target {
    s.unit(&t.id).map(Target::from).unwrap_or_else(|| t.clone())
}
// 保持与 TS addEffect 的规则参数一一对应，避免另建仅在本语言使用的配置协议。
#[allow(clippy::too_many_arguments)]
pub fn add_effect(
    s: &mut State,
    id: &str,
    kind: &str,
    owner: usize,
    duration: f64,
    amount: Option<f64>,
    source: Option<&str>,
    global: bool,
) {
    let ply = s.ply;
    if let Some(u) = s.unit_mut(id) {
        let start = ply + if global { 0.0 } else { u.offset };
        let e = crate::model::Effect {
            kind: kind.into(),
            owner,
            from: start,
            until: start + duration,
            amount,
            source_id: source.map(str::to_owned),
            global: global.then_some(true),
        };
        u.effects.push(e);
    }
}
pub fn normalize_guards(s: &mut State) {
    let guards: Vec<_> = s
        .units
        .iter()
        .filter(|u| u.has("3"))
        .map(|u| (u.owner, u.id.clone()))
        .collect();
    for u in &mut s.units {
        if u.guard_used && !u.extra.contains_key("guardSourceIds") {
            let owner = u.owner;
            u.extra.insert(
                "guardSourceIds".into(),
                json!(
                    guards
                        .iter()
                        .filter(|(o, _)| *o == owner)
                        .map(|(_, id)| id)
                        .collect::<Vec<_>>()
                ),
            );
        }
    }
}
pub fn guardian(s: &State, u: &Unit, catalog: &Catalog) -> Option<String> {
    if u.side() == 0 {
        return None;
    }
    let used = u.extra.get("guardSourceIds").and_then(Value::as_array);
    let mut sources: Vec<_> = s
        .units
        .iter()
        .filter(|v| {
            v.has("3")
                && v.side() == u.owner
                && !v.silenced
                && !used.is_some_and(|a| a.contains(&json!(v.id)))
                && attack_path(
                    s,
                    v,
                    &Target::from(u),
                    stats(s, v, catalog).range,
                    None,
                    false,
                )
                .is_some()
        })
        .collect();
    sources.sort_by(|a, b| a.deployed_at.total_cmp(&b.deployed_at));
    sources.first().map(|u| u.id.clone())
}
pub fn template(
    s: &State,
    kind: &str,
    owner: usize,
    at: Point,
    id: &str,
    catalog: &Catalog,
) -> Unit {
    let d = &catalog[kind];
    use crate::model::{Charge, ChargeMode, UnitData};
    Unit::new(UnitData {
        id: id.into(),
        kind: d.id.clone(),
        owner,
        x: at.x,
        y: at.y,
        hp: d.health,
        max_hp: d.health,
        size: d.size.unwrap_or(1.0),
        born: s.turns[&owner.to_string()],
        offset: 0.0,
        deployed_at: 0.0,
        charged_on_deploy: ["u1", "u12", "u12p"].contains(&kind),
        mode: "none".into(),
        operations: 0.0,
        shots: 0.0,
        moves: 0.0,
        attacked: vec![],
        bonus_attacks: 0.0,
        bonus_sequence: false,
        weapon_first_used: false,
        reserve: Charge {
            charge: 0.0,
            ready_charge: 0.0,
            charge_type: if d.movement.fract() != 0.0 {
                ChargeMode::Move
            } else if kind == "21" {
                ChargeMode::Skill
            } else {
                ChargeMode::Attack
            },
            last_charge: -1.0,
        },
        upgrades: 0.0,
        kills: 0.0,
        attack_bonus: 0.0,
        range_bonus: 0.0,
        guard_used: false,
        effects: vec![],
        equipment: vec![],
        silenced: false,
        free_used: -1.0,
        once_used: false,
        traits: None,
        ability_usage: None,
        ability_charges: None,
        extra: Default::default(),
    })
}
pub fn add_unit(
    s: &mut State,
    kind: &str,
    owner: usize,
    at: Point,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    deploy_unit(s, kind, owner, at, None, false, catalog, ctx);
    Ok(())
}
/// 与 TS addUnit 相同的入场顺序：新身份、冲锋、地标时钟、举旗、事件。
/// 输入已由调用方验证占位；死亡衍生物也走此入口，不能复制旧棋子的身份/效果。
#[allow(clippy::too_many_arguments)]
pub fn deploy_unit(
    s: &mut State,
    kind: &str,
    owner: usize,
    at: Point,
    group: Option<&str>,
    charge: bool,
    catalog: &Catalog,
    ctx: &mut Resolution,
) {
    let id = format!("u{}", s.serial);
    s.serial += 1;
    let mut u = template(s, kind, owner, at, &id, catalog);
    u.deployed_at = s.ply;
    if let Some(group) = group {
        u.extra.insert("group".into(), json!(group));
    }
    if ["u1", "u12", "u12p"].contains(&kind) {
        u.born -= 1.0;
    }
    if kind == "1" && charge {
        u.hp -= 10.0;
        u.max_hp -= 10.0;
        u.born -= 1.0;
        u.charged_on_deploy = true;
    }
    if catalog[kind].landmark.is_some() {
        s.landmarks.get_or_insert_default().push(u);
    } else {
        // 与 TS 相同，先找第一处有效金晔覆盖，再检查该来源的沉默状态。
        let landmark = crate::geometry::cells(&u, at).into_iter().find_map(|p| {
            s.landmarks()
                .iter()
                .find(|l| l.at() == p)
                .filter(|l| l.live() && l.owner == owner && l.has("s1"))
        });
        let accelerated = landmark.is_some_and(|l| !l.silenced);
        if accelerated {
            let was_charge = u.charged_on_deploy;
            u.offset += 2.0;
            // 新模板的操作、蓄力和效果均为空；resetUnit 在这里不会产生其他变化。
            u.charged_on_deploy = true;
            if was_charge {
                u.extra.insert("extraOperations".into(), json!(1));
            }
            ctx.emit(s, json!({"type":"skill","to":u.actor_event(),"owner":owner,"ability":"u17","action":"clock","text":if was_charge {"金晔 · 冲锋与双操作"} else {"金晔 · 冲锋号令"}}), None);
        }
        s.units.push(u);
    }
    crate::movement::sync_banners(s, catalog);
    ctx.emit(s,json!({"type":"spawn","to":at,"unitId":id,"owner":owner,"ultimate":catalog[kind].tier!="normal"}),Some(format!("{}部署{}",faction(owner),catalog[kind].name)));
}
pub fn terminal(s: &mut State, ctx: &mut Resolution) {
    if number(&s.bases["1"]) > 0.0 && number(&s.bases["2"]) > 0.0 {
        return;
    }
    let winner = if number(&s.bases["1"]) <= 0.0 && number(&s.bases["2"]) <= 0.0 {
        json!("draw")
    } else {
        json!(if number(&s.bases["1"]) <= 0.0 { 2 } else { 1 })
    };
    s.extra.insert("winner".into(), winner.clone());
    s.pending.clear();
    let message = if winner == "draw" {
        "双方基地失守，平局".into()
    } else {
        format!("{}获胜", faction(winner.as_u64().unwrap() as usize))
    };
    ctx.emit(s, json!({"type":"turn","text":"对局结束"}), Some(message));
}
