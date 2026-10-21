//! 原生训练的外部输入边界；局面只能由新局加已验证命令重建，禁止任意内部状态注入。
use crate::model::{COMMANDS, Catalog, Command, Kind, State};
use serde_json::Value;

fn integer(v: &Value, low: f64, high: f64) -> bool {
    v.as_f64()
        .is_some_and(|n| n.is_finite() && n.fract() == 0.0 && n >= low && n <= high)
}
fn id(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| !s.is_empty() && s.encode_utf16().count() <= 128)
}
pub fn command(v: &Value, catalog: &Catalog) -> Result<Command, String> {
    let object = v.as_object().ok_or("command expects object")?;
    if !v["type"].as_str().is_some_and(|s| COMMANDS.contains(&s)) {
        return Err("unknown command".into());
    }
    for (k, value) in object {
        let valid = match k.as_str() {
            "type" => true,
            "unitId" | "cardId" | "targetId" | "secondId" | "deathId" | "recipeId" => id(value),
            "ability" | "chosenKind" | "shrineKind" => {
                serde_json::from_value::<Kind>(value.clone())
                    .is_ok_and(|k| catalog.get(&k.key()).is_some_and(|d| d.id == k))
            }
            "ultimate" | "charge" => value.is_boolean(),
            "x" | "column" => integer(value, 1.0, 9.0),
            "y" | "row" => integer(value, 1.0, 13.0),
            "player" => integer(value, 1.0, 2.0),
            "parity" => value.as_str().is_some_and(|s| ["odd", "even"].contains(&s)),
            "direction" => value
                .as_str()
                .is_some_and(|s| ["up", "down", "left", "right"].contains(&s)),
            "mode" => value
                .as_str()
                .is_some_and(|s| !s.is_empty() && s.encode_utf16().count() <= 32),
            "materialIds" | "cardIds" | "sacrificeIds" | "offerIndices" => {
                value.as_array().is_some_and(|a| {
                    a.len() <= if k == "offerIndices" { 16 } else { 128 }
                        && a.iter().enumerate().all(|(i, v)| {
                            (if k == "offerIndices" {
                                integer(v, 0.0, 15.0)
                            } else {
                                id(v)
                            }) && !a[..i].contains(v)
                        })
                })
            }
            "path" => value.as_array().is_some_and(|a| {
                a.len() <= 256
                    && a.iter().all(|p| {
                        p.as_object().is_some_and(|o| {
                            o.len() == 2
                                && integer(&p["x"], 1.0, 9.0)
                                && integer(&p["y"], 1.0, 13.0)
                        })
                    })
            }),
            _ => false,
        };
        if !valid {
            return Err(format!("invalid command field {k}"));
        }
    }
    serde_json::from_value(v.clone()).map_err(|e| e.to_string())
}
pub fn step(
    state: &State,
    actor: usize,
    value: &Value,
    catalog: &Catalog,
) -> Result<State, String> {
    let command = command(value, catalog)?;
    step_typed(state, actor, &command, catalog)
}

/// 已由内部动作树构造的类型化命令；权限与正式结算仍在宿主执行，不回到 JSON 解析。
pub fn step_typed(
    state: &State,
    actor: usize,
    command: &Command,
    catalog: &Catalog,
) -> Result<State, String> {
    if !crate::actions::permitted_command(state, actor, command) {
        return Err("unauthorized actor".into());
    }
    let mut next = crate::resolution::without_presentation(|| {
        crate::command::apply_runtime(state, command, catalog)
    })
    .map_err(|e| format!("rule rejected command: {e:?}"))?;
    next.events.clear();
    next.extra.insert("log".into(), serde_json::json!([]));
    Ok(next)
}

/// 仅公开树标为 uncertain 的正式规则拒绝可回溯；权限和未实现错误仍为致命错误。
pub fn attempt(
    state: &State,
    actor: usize,
    command: &Command,
    status: &str,
    catalog: &Catalog,
) -> Result<Result<State, String>, String> {
    if !crate::actions::permitted_command(state, actor, command) {
        return Err("unauthorized actor".into());
    }
    match crate::resolution::without_presentation(|| {
        crate::command::apply_runtime(state, command, catalog)
    }) {
        Ok(mut next) => {
            next.events.clear();
            next.extra.insert("log".into(), serde_json::json!([]));
            Ok(Ok(next))
        }
        Err(e) => {
            let message = format!("rule rejected command: {e:?}");
            if status == "uncertain"
                && matches!(
                    e,
                    crate::model::Failure::Invalid(_) | crate::model::Failure::InvalidOwned(_)
                )
            {
                Ok(Err(message))
            } else {
                Err(message)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChargeMode, Point};
    use serde_json::json;

    #[test]
    fn training_presentation_preserves_formal_state_rng_serial_and_rejection() {
        let catalog = crate::catalog::embedded().unwrap();
        for action in ["attack", "charge", "equip", "rush", "end", "summon"] {
            let mut state = crate::runtime::create(73127, false, &catalog).unwrap();
            state.phase = "play".into();
            state.summon_slots = 0.0;
            state.turns.insert("1".into(), 3.0);
            state.turns.insert("2".into(), 3.0);
            let kind = match action {
                "attack" => "1",
                "charge" => "15",
                "rush" => "u12",
                _ => "26",
            };
            let mut source = crate::resolution::template(
                &state,
                kind,
                1,
                Point { x: 4.0, y: 4.0 },
                "source",
                &catalog,
            );
            source.born = 0.0;
            source.reserve.charge = if action == "charge" { 0.0 } else { 2.0 };
            source.reserve.ready_charge = source.reserve.charge;
            source.reserve.charge_type = if action == "rush" {
                ChargeMode::Move
            } else {
                ChargeMode::Attack
            };
            let mut target = crate::resolution::template(
                &state,
                if action == "attack" { "17p" } else { "26" },
                2,
                Point { x: 4.0, y: 5.0 },
                "target",
                &catalog,
            );
            target.born = 0.0;
            target.hp = 1000.0;
            target.max_hp = 1000.0;
            state.units = vec![source, target];
            let mut command = Command::new(if action == "rush" { "move" } else { action });
            match action {
                "attack" => {
                    command.unit_id = Some("source".into());
                    command.target_id = Some("target".into());
                }
                "charge" => {
                    command.unit_id = Some("source".into());
                    command.mode = Some("attack".into());
                }
                "equip" => {
                    state.extra.get_mut("hands").unwrap()["1"] =
                        json!([{"id":"weapon","kind":"s2","drawnAt":0,"summonedPly":1}]);
                    command.card_id = Some("weapon".into());
                    command.target_id = Some("source".into());
                }
                "rush" => {
                    command.unit_id = Some("source".into());
                    command.x = Some(4.0);
                    command.y = Some(5.0);
                }
                "summon" => {
                    state.phase = "summon".into();
                    state.summon_slots = 1.0;
                    command.ultimate = Some(false);
                }
                _ => {}
            }
            let frozen = serde_json::to_value(&state).unwrap();
            let mut expected = crate::command::apply_runtime(&state, &command, &catalog).unwrap();
            let raw = crate::resolution::without_presentation(|| {
                crate::command::apply_runtime(&state, &command, &catalog)
            })
            .unwrap();
            assert_eq!(raw.events.len(), expected.events.len(), "{action}");
            assert_eq!(raw.serial, expected.serial, "{action}");
            assert_eq!(raw.extra["rng"], expected.extra["rng"], "{action}");
            expected.events.clear();
            expected.extra.insert("log".into(), json!([]));
            let actual = step_typed(&state, 1, &command, &catalog).unwrap();
            let attempted = attempt(&state, 1, &command, "uncertain", &catalog)
                .unwrap()
                .unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(&expected).unwrap(),
                "{action}"
            );
            assert_eq!(
                serde_json::to_value(attempted).unwrap(),
                serde_json::to_value(&expected).unwrap(),
                "{action}"
            );
            assert_eq!(serde_json::to_value(&state).unwrap(), frozen, "{action}");
            if action == "attack" {
                command.target_id = Some("missing".into());
                let error = format!(
                    "rule rejected command: {:?}",
                    crate::command::apply_runtime(&state, &command, &catalog)
                        .err()
                        .unwrap()
                );
                assert_eq!(
                    step_typed(&state, 1, &command, &catalog).err().unwrap(),
                    error
                );
                assert_eq!(
                    attempt(&state, 1, &command, "uncertain", &catalog)
                        .unwrap()
                        .err()
                        .unwrap(),
                    error
                );
                assert_eq!(serde_json::to_value(&state).unwrap(), frozen);
            }
        }
    }
}
