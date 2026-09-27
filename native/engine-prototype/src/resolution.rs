use crate::geometry::{Target, attack_path};
use crate::model::{Catalog, Failure, Point, State, Unit, number};
use crate::stats::stats;
use serde_json::{Value, json};
use std::collections::HashSet;

#[derive(Default)]
/// 一条命令内的临时结算上下文；事实范围服务事件快照，反击集合阻止递归循环。
/// 不写入存档，不持有真实时钟；preview 到随机边界后由调用栈返回 uncertain。
pub struct Resolution {
    pub preview: bool,
    pub retaliations: HashSet<String>,
    pub facts: Option<Value>,
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
    serde_json::from_value(json!({"id":id,"kind":d.id,"owner":owner,"x":at.x,"y":at.y,"hp":d.health,"maxHp":d.health,"size":1,
        "born":s.turns[&owner.to_string()],"offset":0,"deployedAt":s.ply,"chargedOnDeploy":false,"mode":"none","operations":0,"shots":0,"moves":0,"attacked":[],
        "bonusAttacks":0,"bonusSequence":false,"weaponFirstUsed":false,"charge":0,"readyCharge":0,"chargeType":if d.movement.fract()!=0.0{"move"}else{"attack"},"lastCharge":-1,
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
    // 目前只用于墓地/死亡小屋衍生物；金晔入场时钟另立显式能力缺口。
    if s.landmarks()
        .iter()
        .any(|l| l.at() == at && l.live() && l.owner == owner && !l.silenced && l.has("s1"))
    {
        return Err(Failure::Unsupported("landmark-deployment"));
    }
    let id = format!("u{}", s.serial);
    s.serial += 1;
    let u = template(s, kind, owner, at, &id, catalog);
    s.units.push(u.clone());
    crate::movement::sync_banners(s, catalog);
    ctx.emit(s,json!({"type":"spawn","to":at,"unitId":id,"owner":owner,"ultimate":catalog[kind].tier!="normal"}),Some(format!("{}部署{}",faction(owner),catalog[kind].name)));
    Ok(())
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
