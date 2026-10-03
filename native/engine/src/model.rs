use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

pub const RULESET: &str = "3.0-feedback5-live-deployment-2026-09-23";
pub const PROTOCOL: &str = "haojie-native-engine-v4";
pub use crate::command::{COMMANDS, CommandKind};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
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
    #[serde(skip)]
    pub encoded_printed: std::cell::OnceCell<[Option<f64>; 10]>,
    #[serde(skip)]
    pub printed_mage: Option<bool>,
    #[serde(skip)]
    pub printed_aura: Option<bool>,
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
    pub entries: BTreeMap<String, Rc<Definition>>,
    pub numbers: [Option<Rc<Definition>>; 32],
    pub kind_codes: std::collections::HashMap<Kind, usize>,
    pub vocab_codes: std::collections::HashMap<String, std::collections::HashMap<String, usize>>,
    pub combat: Value,
    pub pools: BTreeMap<String, Vec<Kind>>,
    pub recipes: Vec<Value>,
    pub encoding: Value,
}
impl Deref for Catalog {
    type Target = BTreeMap<String, Rc<Definition>>;
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}
impl Catalog {
    /// 普通编号直接索引；文本编号借用查找，不再为每次属性查询生成 String。
    pub fn by_kind(&self, k: &Kind) -> &Definition {
        self.get_kind(k).expect("规则包编号")
    }
    pub fn get_kind(&self, k: &Kind) -> Option<&Definition> {
        match k {
            Kind::Number(n) if *n >= 0 && (*n as usize) < self.numbers.len() => {
                self.numbers[*n as usize].as_deref()
            }
            Kind::Text(s) => self.entries.get(s).map(Rc::as_ref),
            _ => self.entries.get(&k.key()).map(Rc::as_ref),
        }
    }
    pub fn rule(&self, path: &str) -> f64 {
        self.combat
            .pointer(path)
            .and_then(Value::as_f64)
            .expect("初始化已校验战斗数值")
    }
}

/// 单个实体是分支复制单元；可变借用隔离完整实体，普通 Clone 保持规则快照独立。
#[derive(Deserialize, Serialize)]
#[serde(transparent)]
pub struct Unit(Rc<UnitData>);
impl Clone for Unit {
    fn clone(&self) -> Self {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::UnitCopy);
        Self(Rc::new((*self.0).clone()))
    }
}
impl Deref for Unit {
    type Target = UnitData;
    fn deref(&self) -> &UnitData {
        &self.0
    }
}
impl DerefMut for Unit {
    fn deref_mut(&mut self) -> &mut UnitData {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::UnitCopy);
        Rc::make_mut(&mut self.0)
    }
}

/// 核心字段建类型，其余字段无损保留；只接受入口已校验的规范实体。
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChargeMode {
    Move,
    Attack,
    Skill,
}
impl ChargeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Move => "move",
            Self::Attack => "attack",
            Self::Skill => "skill",
        }
    }
    pub fn parse(value: &str) -> Self {
        match value {
            "move" => Self::Move,
            "skill" => Self::Skill,
            "attack" => Self::Attack,
            _ => unreachable!("准备阶段已校验蓄力模式"),
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Charge {
    pub charge: f64,
    pub ready_charge: f64,
    pub charge_type: ChargeMode,
    pub last_charge: f64,
}

/// 能力附表在协议入口严格读取；运行期间只使用小型类型化记录。
fn read_charges<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<indexmap::IndexMap<String, Charge>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Entry {
        charge: f64,
        ready_charge: f64,
        charge_type: ChargeMode,
        last_charge: f64,
    }
    let values = Option::<indexmap::IndexMap<String, Entry>>::deserialize(d)?;
    Ok(values.map(|m| {
        m.into_iter()
            .map(|(key, e)| {
                (
                    key,
                    Charge {
                        charge: e.charge,
                        ready_charge: e.ready_charge,
                        charge_type: e.charge_type,
                        last_charge: e.last_charge,
                    },
                )
            })
            .collect()
    }))
}
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AbilityUsage {
    pub once: bool,
    pub free: f64,
}

#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Effect {
    #[serde(rename = "type")]
    pub kind: String,
    pub from: f64,
    pub until: f64,
    pub owner: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amount: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global: Option<bool>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UnitData {
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
    pub deployed_at: f64,
    pub charged_on_deploy: bool,
    pub attacked: Vec<String>,
    pub weapon_first_used: bool,
    pub upgrades: f64,
    pub kills: f64,
    pub attack_bonus: f64,
    pub range_bonus: f64,
    pub guard_used: bool,
    pub free_used: f64,
    pub once_used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traits: Option<Vec<Kind>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ability_usage: Option<indexmap::IndexMap<String, AbilityUsage>>,
    pub effects: Vec<Effect>,
    pub equipment: Vec<Kind>,
    #[serde(flatten)]
    pub reserve: Charge,
    #[serde(
        default,
        deserialize_with = "read_charges",
        skip_serializing_if = "Option::is_none"
    )]
    pub ability_charges: Option<indexmap::IndexMap<String, Charge>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
impl Unit {
    pub fn new(data: UnitData) -> Self {
        Self(Rc::new(data))
    }
    // 内部局面与只读查询共享实体；写入自动分离，规则快照继续使用拥有型复制。
    pub fn fork(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
    pub fn kinds(&self) -> Vec<Kind> {
        let mut result = vec![self.kind.clone()];
        if let Some(traits) = &self.traits {
            for kind in traits {
                if !result.contains(kind) {
                    result.push(kind.clone());
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
                .traits
                .as_ref()
                .is_some_and(|a| a.iter().any(|k| k.is(kind)))
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

/// 反应持有当时的实体快照；后续活实体修改由 Rc 写入隔离，不重新解析 JSON。
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Reaction {
    pub kind: String,
    pub owner: usize,
    pub source: Unit,
    pub amount: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
}
impl Reaction {
    pub fn new(kind: &str, source: &Unit, amount: f64) -> Self {
        Self {
            kind: kind.into(),
            owner: source.owner,
            source: source.fork(),
            amount,
            target_id: None,
        }
    }
    pub fn fork(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            owner: self.owner,
            source: self.source.fork(),
            amount: self.amount,
            target_id: self.target_id.clone(),
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClockFrame {
    pub ply: f64,
    pub turns: BTreeMap<String, f64>,
    pub units: Vec<Unit>,
}
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClockFrames {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<Rc<ClockFrame>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<Rc<ClockFrame>>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct State {
    #[serde(skip)]
    pub entities: crate::entities::EntityIndex,
    pub version: u8,
    pub ply: f64,
    pub active: usize,
    pub phase: String,
    pub summon_slots: f64,
    pub turns: BTreeMap<String, f64>,
    pub units: Vec<Unit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub landmarks: Option<Vec<Unit>>,
    pub pending: Vec<Reaction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_frames: Option<BTreeMap<String, ClockFrames>>,
    pub siphons: Vec<Value>,
    pub events: Vec<Value>,
    pub deploy_rows: Value,
    pub bases: Value,
    pub base_effects: BTreeMap<String, Vec<Effect>>,
    pub serial: u64,
    #[serde(flatten)]
    pub extra: crate::shared::ValueMap,
}
impl State {
    /// 与 TS forkPosition 相同：实体和扩展字段按需复制，小型核心容器立即隔离。
    /// 预检/失败丢弃；内部成功直接保留，外部 transition 另行导出独立快照。
    pub fn fork(&self) -> Self {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Branch);
        Self {
            entities: self.entities.clone(),
            version: self.version,
            ply: self.ply,
            active: self.active,
            phase: self.phase.clone(),
            summon_slots: self.summon_slots,
            turns: self.turns.clone(),
            units: self.units.iter().map(Unit::fork).collect(),
            landmarks: self
                .landmarks
                .as_ref()
                .map(|values| values.iter().map(Unit::fork).collect()),
            pending: self.pending.iter().map(Reaction::fork).collect(),
            clock_frames: self.clock_frames.clone(),
            siphons: self.siphons.clone(),
            events: self.events.clone(),
            deploy_rows: self.deploy_rows.clone(),
            bases: self.bases.clone(),
            base_effects: self.base_effects.clone(),
            serial: self.serial,
            extra: self.extra.fork(),
        }
    }
    pub fn unit(&self, id: &str) -> Option<&Unit> {
        self.entities.handle(self, id).and_then(|h| self.entity(h))
    }
    pub fn entity(&self, h: crate::entities::EntityHandle) -> Option<&Unit> {
        let u = self.entities.at(self, h)?;
        let (landmark, _) = self.entities.location(self, h)?;
        (!landmark || u.live()).then_some(u)
    }
    pub fn unit_mut(&mut self, id: &str) -> Option<&mut Unit> {
        let h = self.entities.handle(self, id)?;
        let (landmark, index) = self.entities.location(self, h)?;
        if landmark {
            self.landmarks.as_mut()?.get_mut(index).filter(|u| u.live())
        } else {
            self.units.get_mut(index)
        }
    }
    pub fn active_effect(&self, e: &Effect, u: Option<&Unit>) -> bool {
        let clock = self.ply
            + if e.global == Some(true) {
                0.0
            } else {
                u.map(|u| u.offset).unwrap_or(0.0)
            };
        e.from <= clock && e.until > clock
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
            let clock = self.ply
                + if e.global == Some(true) {
                    0.0
                } else {
                    u.offset
                };
            e.kind == name && e.from <= clock && e.until > clock
        })
    }
}
pub fn number(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}
pub fn extra_number(u: &Unit, key: &str) -> f64 {
    match key {
        "charge" => return u.reserve.charge,
        "readyCharge" => return u.reserve.ready_charge,
        "lastCharge" => return u.reserve.last_charge,
        "deployedAt" => return u.deployed_at,
        "upgrades" => return u.upgrades,
        "kills" => return u.kills,
        "attackBonus" => return u.attack_bonus,
        "rangeBonus" => return u.range_bonus,
        "freeUsed" => return u.free_used,
        _ => {}
    }
    u.extra.get(key).map(number).unwrap_or(0.0)
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Command {
    #[serde(rename = "type")]
    pub kind: CommandKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<Vec<Point>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ability: Option<Kind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chosen_kind: Option<Kind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ultimate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub charge: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offer_indices: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shrine_kind: Option<Kind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub death_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sacrifice_ids: Option<Vec<String>>,
}

impl Command {
    /// 内部构造不经过 JSON；可选参数保持缺失状态，外部边界仍独立校验。
    pub fn new(kind: &str) -> Self {
        Self {
            kind: kind.into(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
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
