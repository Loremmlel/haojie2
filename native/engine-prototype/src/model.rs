use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::ops::Deref;

pub const RULESET: &str = "3.0-feedback5-live-deployment-2026-09-23";
pub const PROTOCOL: &str = "haojie-native-engine-v4";
pub const COMMANDS: &[&str] = &[
    "move",
    "finish-mode",
    "attack",
    "react",
    "summon",
    "extra-summon",
    "choose-summons",
    "deploy",
    "charge",
    "equip",
    "activate-aura",
    "reroll",
    "begin",
    "skip-synthesis",
    "end",
    "synthesize",
    "craft",
    "choose-shrine",
    "finish-shrine-setup",
    "skill",
    "cast",
    "clock",
    "shatter",
];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Kind {
    Number(i32),
    Text(String),
}
impl Kind {
    pub fn key(&self) -> String {
        match self {
            Self::Number(n) => n.to_string(),
            Self::Text(s) => s.clone(),
        }
    }
    pub fn is(&self, key: &str) -> bool {
        match self {
            Self::Number(n) => key.parse::<i32>() == Ok(*n),
            Self::Text(s) => s == key,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Deserialize)]
pub struct Definition {
    pub id: Kind,
    pub name: String,
    pub tier: String,
    pub attack: f64,
    pub health: f64,
    pub range: f64,
    pub actions: f64,
    #[serde(default)]
    pub mage: bool,
    #[serde(rename = "move")]
    pub movement: f64,
    pub landmark: Option<Value>,
    pub size: Option<f64>,
    pub spell: Option<f64>,
    pub skill: Option<String>,
    pub weapon: Option<f64>,
    #[serde(default)]
    pub aura: bool,
}
pub struct Catalog {
    pub entries: BTreeMap<String, Definition>,
    pub combat: Value,
    pub pools: BTreeMap<String, Vec<Kind>>,
    pub recipes: Vec<Value>,
}
impl Deref for Catalog {
    type Target = BTreeMap<String, Definition>;
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}
impl Catalog {
    pub fn rule(&self, path: &str) -> f64 {
        self.combat
            .pointer(path)
            .and_then(Value::as_f64)
            .expect("初始化已校验战斗数值")
    }
}

/// 已移植规则的核心字段建类型，其余字段无损保留；不能把此类型当作完整规则实现。
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Unit {
    pub id: String,
    pub kind: Kind,
    pub owner: usize,
    pub x: f64,
    pub y: f64,
    pub size: f64,
    pub hp: f64,
    pub max_hp: f64,
    pub mode: String,
    pub operations: f64,
    pub shots: f64,
    pub moves: f64,
    pub bonus_attacks: f64,
    pub bonus_sequence: bool,
    pub silenced: bool,
    pub offset: f64,
    pub born: f64,
    pub effects: Vec<Value>,
    pub equipment: Vec<Kind>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
impl Unit {
    pub fn kinds(&self) -> Vec<Kind> {
        let mut result = vec![self.kind.clone()];
        if let Some(traits) = self.extra.get("traits").and_then(Value::as_array) {
            for value in traits {
                let kind: Kind = serde_json::from_value(value.clone()).expect("入口已校验能力编号");
                if !result.contains(&kind) {
                    result.push(kind);
                }
            }
        }
        result
    }
    pub fn any(&self, kinds: &[&str]) -> bool {
        kinds.iter().any(|k| self.has(k))
    }
    pub fn signed(&self) -> bool {
        self.any(&["u21", "sage", "s4", "s6"])
    }
    pub fn piercing(&self) -> bool {
        self.weapon("u28") || (!self.silenced && self.has("slayer"))
    }
    pub fn at(&self) -> Point {
        Point {
            x: self.x,
            y: self.y,
        }
    }
    pub fn has(&self, kind: &str) -> bool {
        self.kind.is(kind)
            || self
                .extra
                .get("traits")
                .and_then(Value::as_array)
                .is_some_and(|a| {
                    a.iter().any(|k| match k {
                        Value::String(s) => s == kind,
                        Value::Number(n) => n.to_string() == kind,
                        _ => false,
                    })
                })
    }
    pub fn weapon(&self, kind: &str) -> bool {
        self.equipment.iter().any(|k| k.is(kind))
    }
    pub fn side(&self) -> usize {
        if self.kind.is("grave") { 0 } else { self.owner }
    }
    pub fn live(&self) -> bool {
        self.hp > 0.0 && !self.extra.contains_key("dormantSince")
    }
    pub fn runner(&self) -> bool {
        !self.silenced && (self.has("u12") || self.has("u12p"))
    }
    pub fn actor_event(&self) -> Value {
        json!({"id":self.id,"owner":self.owner,"kind":self.kind,"size":self.size,"x":self.x,"y":self.y})
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct State {
    pub version: u8,
    pub ply: f64,
    pub active: usize,
    pub phase: String,
    pub summon_slots: f64,
    pub turns: BTreeMap<String, f64>,
    pub units: Vec<Unit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub landmarks: Option<Vec<Unit>>,
    pub pending: Vec<Value>,
    pub siphons: Vec<Value>,
    pub events: Vec<Value>,
    pub deploy_rows: Value,
    pub bases: Value,
    pub serial: u64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
impl State {
    pub fn unit(&self, id: &str) -> Option<&Unit> {
        self.pieces().find(|u| u.id == id)
    }
    pub fn unit_mut(&mut self, id: &str) -> Option<&mut Unit> {
        self.units
            .iter_mut()
            .chain(self.landmarks.iter_mut().flatten().filter(|l| l.live()))
            .find(|u| u.id == id)
    }
    pub fn active_effect(&self, e: &Value, u: Option<&Unit>) -> bool {
        let clock = self.ply
            + if e["global"] == true {
                0.0
            } else {
                u.map(|u| u.offset).unwrap_or(0.0)
            };
        number(&e["from"]) <= clock && number(&e["until"]) > clock
    }
    pub fn aura(&self, owner: usize, kind: &str) -> bool {
        self.extra
            .get("auras")
            .and_then(|a| a.get(owner.to_string()))
            .and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(|v| v["kind"] == kind))
    }
    pub fn random(&mut self, preview: bool) -> Result<f64, Failure> {
        if preview {
            return Err(Failure::Uncertain);
        }
        let mut x = self
            .extra
            .get("rng")
            .and_then(Value::as_u64)
            .ok_or(Failure::Unsupported("missing-authoritative-rng"))? as u32;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.extra.insert("rng".into(), json!(x));
        Ok(x as f64 / 4294967296.0)
    }
    pub fn landmarks(&self) -> &[Unit] {
        self.landmarks.as_deref().unwrap_or(&[])
    }
    pub fn pieces(&self) -> impl Iterator<Item = &Unit> {
        self.units
            .iter()
            .chain(self.landmarks().iter().filter(|l| l.live()))
    }
    pub fn effect(&self, u: &Unit, name: &str) -> bool {
        u.effects.iter().any(|e| {
            let clock = self.ply + if e["global"] == true { 0.0 } else { u.offset };
            e["type"] == name && number(&e["from"]) <= clock && number(&e["until"]) > clock
        })
    }
}
pub fn number(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}
pub fn extra_number(u: &Unit, key: &str) -> f64 {
    u.extra.get(key).map(number).unwrap_or(0.0)
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Command {
    #[serde(rename = "type")]
    pub kind: String,
    pub unit_id: Option<String>,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub target_id: Option<String>,
    pub direction: Option<String>,
    pub mode: Option<String>,
    pub path: Option<Vec<Point>>,
    pub card_id: Option<String>,
    pub ability: Option<Kind>,
    pub chosen_kind: Option<Kind>,
    pub ultimate: Option<bool>,
    pub charge: Option<bool>,
    pub offer_indices: Option<Vec<f64>>,
    pub player: Option<usize>,
    pub shrine_kind: Option<Kind>,
    pub parity: Option<String>,
    pub recipe_id: Option<String>,
    pub material_ids: Option<Vec<String>>,
    pub second_id: Option<String>,
    pub death_id: Option<String>,
    pub column: Option<f64>,
    pub row: Option<f64>,
    pub card_ids: Option<Vec<String>>,
    pub sacrifice_ids: Option<Vec<String>>,
}

#[derive(Debug)]
pub enum Failure {
    Invalid(&'static str),
    InvalidOwned(String),
    Unsupported(&'static str),
    Uncertain,
}
impl Failure {
    pub fn value(&self) -> Value {
        match self {
            Self::Invalid(message) => json!({"status":"invalid", "message":message}),
            Self::InvalidOwned(message) => json!({"status":"invalid", "message":message}),
            Self::Unsupported(reason) => json!({"status":"unsupported", "reason":reason}),
            Self::Uncertain => json!({"status":"uncertain"}),
        }
    }
}
pub fn ensure(value: bool, message: &'static str) -> Result<(), Failure> {
    if value {
        Ok(())
    } else {
        Err(Failure::Invalid(message))
    }
}
