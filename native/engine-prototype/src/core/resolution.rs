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
    u.extra.insert("attacked".into(), json!([]));
    u.extra.insert("weaponFirstUsed".into(), json!(false));
    u.extra
        .insert("readyCharge".into(), u.extra["charge"].clone());
    if u.extra.contains_key("extraOperations") {
        u.extra.insert("extraOperations".into(), json!(0));
    }
    if let Some(reserves) = u
        .extra
        .get_mut("abilityCharges")
        .and_then(Value::as_object_mut)
    {
        for reserve in reserves.values_mut().filter(|v| v.is_object()) {
            reserve["readyCharge"] = reserve["charge"].clone();
        }
    }
    u.effects
        .retain(|e| number(&e["until"]) > ply + if e["global"] == true { 0.0 } else { u.offset });
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
        let mut e = json!({"type":kind,"owner":owner,"from":start,"until":start+duration});
        if let Some(amount) = amount {
            e["amount"] = json!(amount);
        }
        if let Some(source) = source {
            e["sourceId"] = json!(source);
        }
        if global {
            e["global"] = json!(true);
        }
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
        if u.extra.get("guardUsed") == Some(&json!(true)) && !u.extra.contains_key("guardSourceIds")
        {
            u.extra.insert(
                "guardSourceIds".into(),
                json!(
                    guards
                        .iter()
                        .filter(|(o, _)| *o == u.owner)
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
    sources
        .sort_by(|a, b| number(&a.extra["deployedAt"]).total_cmp(&number(&b.extra["deployedAt"])));
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
    serde_json::from_value(json!({"id":id,"kind":d.id,"owner":owner,"x":at.x,"y":at.y,"hp":d.health,"maxHp":d.health,"size":d.size.unwrap_or(1.0),
        "born":s.turns[&owner.to_string()],"offset":0,"deployedAt":0,"chargedOnDeploy":(["u1","u12","u12p"].contains(&kind)),"mode":"none","operations":0,"shots":0,"moves":0,"attacked":[],
        "bonusAttacks":0,"bonusSequence":false,"weaponFirstUsed":false,"charge":0,"readyCharge":0,"chargeType":if d.movement.fract()!=0.0{"move"}else if kind=="21"{"skill"}else{"attack"},"lastCharge":-1,
        "upgrades":0,"kills":0,"attackBonus":0,"rangeBonus":0,"guardUsed":false,"effects":[],"equipment":[],"silenced":false,"freeUsed":-1,"onceUsed":false})).unwrap()
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
    u.extra.insert("deployedAt".into(), json!(s.ply));
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
        u.extra.insert("chargedOnDeploy".into(), json!(true));
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
            let was_charge = u.extra["chargedOnDeploy"] == true;
            u.offset += 2.0;
            // 新模板的操作、蓄力和效果均为空；resetUnit 在这里不会产生其他变化。
            u.extra.insert("chargedOnDeploy".into(), json!(true));
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
