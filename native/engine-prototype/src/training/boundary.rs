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
    if !crate::actions::permitted(state, actor, value) {
        return Err("unauthorized actor".into());
    }
    let mut next = crate::apply_runtime(state, &command, catalog)
        .map_err(|e| format!("rule rejected command: {e:?}"))?;
    next.events.clear();
    next.extra.insert("log".into(), serde_json::json!([]));
    Ok(next)
}
