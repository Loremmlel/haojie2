//! TS JSON 键顺序与数字格式用于原手工策略的场景种子；不替代存档规范哈希。
use crate::{
    model::{Command, State},
    runtime,
};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub struct Keys {
    unit_order: HashMap<String, Vec<String>>,
    effect_order: HashMap<String, Vec<String>>,
    reaction_order: HashMap<String, Vec<String>>,
}
const UNIT: &[&str] = &[
    "id",
    "kind",
    "owner",
    "x",
    "y",
    "hp",
    "maxHp",
    "size",
    "born",
    "offset",
    "deployedAt",
    "chargedOnDeploy",
    "mode",
    "operations",
    "shots",
    "moves",
    "attacked",
    "bonusAttacks",
    "bonusSequence",
    "weaponFirstUsed",
    "charge",
    "readyCharge",
    "chargeType",
    "lastCharge",
    "upgrades",
    "kills",
    "attackBonus",
    "rangeBonus",
    "guardUsed",
    "effects",
    "equipment",
    "silenced",
    "freeUsed",
    "onceUsed",
];
fn reorder(value: Value, order: impl IntoIterator<Item = impl AsRef<str>>) -> Value {
    let Value::Object(mut old) = value else {
        return value;
    };
    let mut result = Map::new();
    for key in order {
        let key = key.as_ref();
        if let Some(value) = old.shift_remove(key) {
            result.insert(key.into(), value);
        }
    }
    result.extend(old);
    Value::Object(result)
}
pub fn stringify(value: &Value) -> String {
    fn numbers(v: &Value) -> Value {
        match v {
            Value::Number(n) => {
                let f = n.as_f64().unwrap();
                if f.fract() == 0. && f.abs() < 9007199254740992. {
                    Value::from(f as i64)
                } else {
                    v.clone()
                }
            }
            Value::Array(a) => Value::Array(a.iter().map(numbers).collect()),
            Value::Object(o) => {
                Value::Object(o.iter().map(|(k, v)| (k.clone(), numbers(v))).collect())
            }
            _ => v.clone(),
        }
    }
    serde_json::to_string(&numbers(value)).unwrap()
}
impl Keys {
    pub fn new(observation: &Value) -> Self {
        let mut unit_order = HashMap::new();
        let mut effect_order = HashMap::new();
        for unit in observation["units"].as_array().into_iter().flatten().chain(
            observation["pending"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| &r["source"]),
        ) {
            if let (Some(id), Some(object)) = (unit["id"].as_str(), unit.as_object()) {
                unit_order.insert(id.into(), object.keys().cloned().collect());
            }
            for effect in unit["effects"].as_array().into_iter().flatten() {
                if let Some(object) = effect.as_object() {
                    effect_order.insert(
                        effect_identity(unit["id"].as_str().unwrap_or(""), effect),
                        object.keys().cloned().collect(),
                    );
                }
            }
        }
        for side in ["1", "2"] {
            for effect in observation["baseEffects"][side]
                .as_array()
                .into_iter()
                .flatten()
            {
                if let Some(object) = effect.as_object() {
                    effect_order.insert(
                        effect_identity(side, effect),
                        object.keys().cloned().collect(),
                    );
                }
            }
        }
        let reaction_order = observation["pending"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| {
                r.as_object()
                    .map(|o| (reaction_identity(r), o.keys().cloned().collect()))
            })
            .collect();
        Self {
            unit_order,
            effect_order,
            reaction_order,
        }
    }
    fn unit(&self, mut value: Value) -> Value {
        let id = value["id"].as_str().unwrap_or("").to_owned();
        if let Some(effects) = value["effects"].as_array_mut() {
            for e in effects {
                *e = self.effect(&id, e.take());
            }
        }
        let order = value["id"].as_str().and_then(|id| self.unit_order.get(id));
        if let Some(order) = order {
            reorder(value, order)
        } else {
            reorder(value, UNIT)
        }
    }
    fn effect(&self, parent: &str, value: Value) -> Value {
        if let Some(order) = self.effect_order.get(&effect_identity(parent, &value)) {
            reorder(value, order)
        } else if value["type"] == "mark" {
            reorder(
                value,
                ["type", "owner", "sourceId", "from", "until", "global"],
            )
        } else {
            reorder(
                value,
                [
                    "type", "owner", "from", "until", "amount", "sourceId", "global",
                ],
            )
        }
    }
    pub fn observation(&self, s: &State) -> Value {
        let mut o = runtime::observe(s, runtime::viewer(s)).unwrap();
        for u in o["units"].as_array_mut().unwrap() {
            *u = self.unit(u.take());
        }
        for r in o["pending"].as_array_mut().unwrap() {
            r["source"] = self.unit(r["source"].take());
            *r = if let Some(order) = self.reaction_order.get(&reaction_identity(r)) {
                reorder(r.take(), order)
            } else {
                reorder(r.take(), ["kind", "owner", "source", "amount", "targetId"])
            };
        }
        for p in ["1", "2"] {
            for e in o["baseEffects"][p].as_array_mut().unwrap() {
                *e = self.effect(p, e.take());
            }
        }
        reorder(
            o,
            [
                "version",
                "serial",
                "ply",
                "active",
                "phase",
                "summonSlots",
                "transitUnitId",
                "turns",
                "bases",
                "baseEffects",
                "heads",
                "hands",
                "bonus",
                "deployRows",
                "units",
                "pending",
                "deaths",
                "hazards",
                "siphons",
                "iceMarks",
                "auras",
                "winner",
            ],
        )
    }
    pub fn position(&self, s: &State) -> String {
        let mut o = self.observation(s);
        o["serial"] = Value::from(0);
        stringify(&o)
    }
    pub fn fingerprint(&self, s: &State) -> String {
        let text = stringify(&self.observation(s));
        format!(
            "{}:{}",
            radix36(crate::chance::hash(&text)),
            text.encode_utf16().count()
        )
    }
}
fn radix36(mut n: u32) -> String {
    if n == 0 {
        return "0".into();
    }
    let mut chars = Vec::new();
    while n > 0 {
        chars.push(char::from_digit(n % 36, 36).unwrap());
        n /= 36;
    }
    chars.into_iter().rev().collect()
}
pub fn public_fingerprint(observation: &Value) -> String {
    let text = stringify(observation);
    format!(
        "{}:{}",
        radix36(crate::chance::hash(&text)),
        text.encode_utf16().count()
    )
}
fn effect_identity(parent: &str, v: &Value) -> String {
    stringify(&serde_json::json!([
        parent,
        v["type"],
        v["owner"],
        v["from"],
        v["until"],
        v["sourceId"]
    ]))
}
fn reaction_identity(v: &Value) -> String {
    stringify(&serde_json::json!([
        v["kind"],
        v["owner"],
        v["source"]["id"],
        v["amount"],
        v["targetId"]
    ]))
}
pub fn command_value(c: &Command) -> Value {
    // 候选按 TS ActionSpec 的字段构造顺序展开；坐标/目标由参数选择追加。
    let value = serde_json::to_value(c).unwrap();
    if let Some(order) = &c.ai_order {
        return reorder(value, order.iter());
    }
    let order = match c.kind.as_str() {
        "synthesize" => vec!["type", "recipeId", "materialIds", "x", "y"],
        "deploy" => vec!["type", "cardId", "charge", "x", "y"],
        "attack"
            if c.target_id.as_deref() == c.unit_id.as_deref()
                && c.mode.as_deref() == Some("heal") =>
        {
            vec!["type", "unitId", "targetId", "mode", "ability", "direction"]
        }
        "skill" if c.x.is_some() && c.target_id.is_some() && c.unit_id.is_some() => vec![
            "type", "unitId", "mode", "ability", "targetId", "x", "y", "column", "secondId",
            "deathId",
        ],
        _ => vec![
            "type",
            "unitId",
            "cardId",
            "mode",
            "ability",
            "ultimate",
            "targetId",
            "secondId",
            "deathId",
            "sacrificeIds",
            "x",
            "y",
            "row",
            "column",
            "direction",
            "path",
        ],
    };
    reorder(value, order)
}
pub fn command(c: &Command) -> String {
    stringify(&command_value(c))
}
pub fn tag(c: &mut Command) {
    if c.ai_order.is_none() {
        c.ai_order = Some(std::rc::Rc::new(
            command_value(c)
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect(),
        ));
    }
}
pub fn append(c: &mut Command, fields: &[&str]) {
    tag(c);
    let order = std::rc::Rc::make_mut(c.ai_order.as_mut().unwrap());
    for key in fields {
        if !order.iter().any(|k| k == key) {
            order.push((*key).into());
        }
    }
}
