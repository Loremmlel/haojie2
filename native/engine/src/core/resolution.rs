use crate::geometry::{Target, attack_path};
use crate::model::{Catalog, Failure, Point, State, Unit, number};
use crate::stats::stats;
use serde_json::{Value, json};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};

thread_local! {
    static PRESENTATION_SUPPRESSED: Cell<bool> = const { Cell::new(false) };
}
struct PresentationScope(bool);
impl Drop for PresentationScope {
    fn drop(&mut self) {
        PRESENTATION_SUPPRESSED.set(self.0);
    }
}
/// 同步训练步只关闭展示，原始事件、正式随机与完整结算不变；嵌套和异常退出恢复旧范围。
pub fn without_presentation<T>(run: impl FnOnce() -> T) -> T {
    let _scope = PresentationScope(PRESENTATION_SUPPRESSED.replace(true));
    run()
}

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
    pub fn presentation_enabled(&self) -> bool {
        !self.preview && !PRESENTATION_SUPPRESSED.get()
    }
    /// 预检只保留原始事件，展示事实延迟到正式结算才构造；恢复范围仍由原调用点负责。
    pub fn replace_facts(&mut self, make: impl FnOnce() -> Value) -> Option<Value> {
        let previous = self.facts.take();
        if self.presentation_enabled() {
            self.facts = Some(make());
        }
        previous
    }
    pub fn enter(&mut self, make: impl FnOnce() -> Value) -> Option<Value> {
        let previous = self.replace_facts(make);
        if let Some(facts) = self.facts.as_mut()
            && let Some(id) = previous.as_ref().and_then(|v| v.get("causeId"))
        {
            facts["parentId"] = id.clone();
        }
        previous
    }
    /// 与 TS enrichEvent 共用字段契约；仅捕获坐标/身份事实，不增加额外 serial 或随机消耗。
    pub fn emit(&mut self, s: &mut State, mut event: Value, message: Option<String>) {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Events);
        let id = format!("e{}", s.serial);
        s.serial += 1;
        event["id"] = json!(id);
        // 与 TS 展示范围一致：保留规则查找所需的原始事件，省去不会导出的展示与日志。
        if !self.presentation_enabled() {
            s.events.push(event);
            return;
        }
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
        let actor = event
            .as_object_mut()
            .unwrap()
            .remove("actor")
            .or_else(|| self.facts.as_ref().and_then(|f| f.get("actor").cloned()))
            .or_else(|| {
                if event["type"] == "attack" {
                    event.get("from").filter(|p| p.get("id").is_some()).cloned()
                } else {
                    None
                }
            });
        let subject = event
            .as_object_mut()
            .unwrap()
            .remove("subject")
            .or(inferred)
            .or_else(|| self.facts.as_ref().and_then(|f| f.get("subject").cloned()));
        let keep_area = ["attack", "skill", "move"].contains(&event["type"].as_str().unwrap_or(""))
            || event.get("area").is_some();
        for key in ["from", "to"] {
            if let Some(p) = event.get(key) {
                event[key] = json!({"x":p["x"],"y":p["y"]});
            }
        }
        // event 已由本次 emit 独占；只复制未被覆盖的上下文字段，优先级与 TS 展开相同。
        let mut result = event;
        if let Some(facts) = self.facts.as_ref().and_then(Value::as_object) {
            let fields = result.as_object_mut().unwrap();
            for (key, value) in facts {
                if key != "actor" && key != "subject" && !fields.contains_key(key) {
                    fields.insert(key.clone(), value.clone());
                }
            }
        }
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
        if !keep_area {
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
    if !s
        .units
        .iter()
        .any(|u| u.guard_used && !u.extra.contains_key("guardSourceIds"))
    {
        return;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_presentation_scope_restores_nested_errors_and_unwind() {
        let mut context = Resolution::default();
        assert!(context.presentation_enabled());
        let result = without_presentation(|| {
            assert!(!context.presentation_enabled());
            assert!(!context.preview);
            context.replace_facts(|| panic!("训练步不得构造展示事实"));
            let nested = without_presentation(|| Err::<(), _>("内层拒绝"));
            assert_eq!(nested, Err("内层拒绝"));
            assert!(!context.presentation_enabled());
            Err::<(), _>("外层拒绝")
        });
        assert_eq!(result, Err("外层拒绝"));
        assert!(context.presentation_enabled());
        assert!(std::panic::catch_unwind(|| without_presentation(|| panic!("范围退出"))).is_err());
        assert!(context.presentation_enabled());
        context.replace_facts(|| json!({"action":"charge"}));
        assert_eq!(context.facts.unwrap()["action"], "charge");
    }

    #[test]
    fn guard_normalization_keeps_missing_and_present_fields_and_source_owners() {
        let catalog = crate::catalog::embedded().unwrap();
        let mut state = crate::runtime::create(73127, false, &catalog).unwrap();
        state.units = vec![
            template(&state, "26", 1, Point { x: 3.0, y: 5.0 }, "first", &catalog),
            template(&state, "3", 1, Point { x: 4.0, y: 5.0 }, "spent", &catalog),
            template(&state, "3", 2, Point { x: 5.0, y: 5.0 }, "enemy", &catalog),
            template(
                &state,
                "26",
                2,
                Point { x: 6.0, y: 5.0 },
                "second",
                &catalog,
            ),
        ];
        state.units[1].hp = 0.0;
        state.units[1].silenced = true;
        let unchanged = serde_json::to_value(&state).unwrap();
        normalize_guards(&mut state);
        assert_eq!(serde_json::to_value(&state).unwrap(), unchanged);

        state.units[0].guard_used = true;
        state.units[0]
            .extra
            .insert("guardSourceIds".into(), Value::Null);
        normalize_guards(&mut state);
        assert_eq!(state.units[0].extra["guardSourceIds"], Value::Null);

        state.units[0].extra.remove("guardSourceIds");
        state.units[3].guard_used = true;
        normalize_guards(&mut state);
        assert_eq!(state.units[0].extra["guardSourceIds"], json!(["spent"]));
        assert_eq!(state.units[3].extra["guardSourceIds"], json!(["enemy"]));
        let normalized = serde_json::to_value(&state).unwrap();
        normalize_guards(&mut state);
        assert_eq!(serde_json::to_value(&state).unwrap(), normalized);
    }

    #[test]
    fn lazy_facts_preserve_formal_scope_and_preview_raw_events() {
        let mut context = Resolution {
            facts: Some(json!({"causeId":"parent","action":"buff"})),
            ..Default::default()
        };
        let previous = context.replace_facts(|| json!({"action":"charge"}));
        assert!(context.facts.as_ref().unwrap().get("parentId").is_none());
        context.facts = previous;
        let previous = context.enter(|| json!({"action":"attack","actor":{"id":"source"}}));
        assert_eq!(context.facts.as_ref().unwrap()["parentId"], "parent");
        context.facts = previous;

        let catalog = crate::catalog::embedded().unwrap();
        let mut state = crate::runtime::create(73127, false, &catalog).unwrap();
        state.events.clear();
        let serial = state.serial;
        let log = state.extra["log"].clone();
        context.preview = true;
        let previous = context.enter(|| panic!("预检不得构造展示事实"));
        assert!(context.facts.is_none());
        assert_eq!(previous.as_ref().unwrap()["causeId"], "parent");
        let mut raw = json!({"type":"skill","owner":1,"action":"mark","text":"原始事件"});
        context.emit(&mut state, raw.clone(), Some("展示日志".into()));
        raw["id"] = json!(format!("e{serial}"));
        assert_eq!(state.events, vec![raw]);
        assert_eq!(state.serial, serial + 1);
        assert_eq!(state.extra["log"], log);
        context.facts = previous;
    }
}
