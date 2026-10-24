use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::borrow::Cow;
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
    Text(Rc<str>),
}
impl Kind {
    pub fn key(&self) -> String {
        self.key_ref().into_owned()
    }
    /// 文本编号直接借用不可变内容；数值编号仍按原方式生成十进制键。
    pub fn key_ref(&self) -> Cow<'_, str> {
        match self {
            Self::Number(n) => Cow::Owned(n.to_string()),
            Self::Text(s) => Cow::Borrowed(s.as_ref()),
        }
    }
    pub fn is(&self, key: &str) -> bool {
        match self {
            Self::Number(n) => key.parse::<i32>() == Ok(*n),
            Self::Text(s) => s.as_ref() == key,
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
    pub printed_features: std::cell::OnceCell<[Option<f64>; 10]>,
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
    pub rule_values: rustc_hash::FxHashMap<&'static str, f64>,
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
            Kind::Text(s) => self.entries.get(s.as_ref()).map(Rc::as_ref),
            _ => self.entries.get(&k.key()).map(Rc::as_ref),
        }
    }
    pub fn rule(&self, path: &str) -> f64 {
        self.rule_values
            .get(path)
            .copied()
            .or_else(|| self.combat.pointer(path).and_then(Value::as_f64))
            .expect("初始化已校验战斗数值")
    }
}

/// 单个实体是分支复制单元；快照共享只读数据，任一副本写入前分离完整实体。
#[derive(Deserialize, Serialize)]
#[serde(transparent)]
pub struct Unit(Rc<UnitData>);
impl Clone for Unit {
    fn clone(&self) -> Self {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::UnitCopy);
        Self(Rc::clone(&self.0))
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
    // 内部查询与规则快照都由类型控制写入分离，不能获得共享数据的可变引用。
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
    /// 纯谓词查询保留本体优先和继承顺序；可能含重复，不替代规则触发用的去重 kinds。
    pub fn native_and_traits(&self) -> impl Iterator<Item = &Kind> {
        std::iter::once(&self.kind).chain(self.traits.iter().flatten())
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
    /// 历史数组和条目分别写前分离；追加记录无需复制既有 JSON 条目。
    pub deaths: Rc<Vec<Rc<Value>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock_frames: Option<BTreeMap<String, ClockFrames>>,
    pub siphons: Vec<Value>,
    pub events: Vec<Value>,
    pub deploy_rows: crate::shared::SharedValue,
    pub bases: crate::shared::SharedValue,
    pub base_effects: BTreeMap<String, Vec<Effect>>,
    pub serial: u64,
    #[serde(flatten)]
    pub extra: crate::shared::ValueMap,
}
impl State {
    /// 与 TS forkPosition 相同：实体和扩展字段按需复制，小型核心容器立即隔离。
    /// 预检/失败丢弃；内部成功直接保留，外部 transition 另行导出独立快照。
    pub fn fork(&self) -> Self {
        self.fork_with_events(true)
    }
    /// 命令总会重置事件；对应 TS 浅分支后直接替换 events，不复制马上丢弃的旧事件。
    pub fn fork_command(&self) -> Self {
        self.fork_with_events(false)
    }
    fn fork_with_events(&self, copy_events: bool) -> Self {
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
            deaths: Rc::clone(&self.deaths),
            clock_frames: self.clock_frames.clone(),
            siphons: self.siphons.clone(),
            events: if copy_events {
                self.events.clone()
            } else {
                vec![]
            },
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
    pub fn turn(&self, owner: usize) -> f64 {
        match owner {
            1 => self.turns["1"],
            2 => self.turns["2"],
            _ => self.turns[&owner.to_string()],
        }
    }
    pub fn entity(&self, h: crate::entities::EntityHandle) -> Option<&Unit> {
        let (landmark, index) = self.entities.location(self, h)?;
        let u = if landmark {
            self.landmarks().get(index)
        } else {
            self.units.get(index)
        }?;
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

/// 对应 TS 稀疏命令对象；候选扩展常用字段内联，较少变动的参数共享并在写入前分离。
/// 序列化仍使用原平面字段与顺序，外部输入不接受内部 extra 字段作为规则参数。
#[derive(Clone, Default)]
pub struct Command {
    pub kind: CommandKind,
    pub unit_id: Option<Rc<str>>,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub target_id: Option<Rc<str>>,
    pub direction: Option<Cow<'static, str>>,
    pub card_id: Option<Rc<str>>,
    pub extra: Option<Rc<CommandExtras>>,
}

#[derive(Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandExtras {
    pub mode: Option<String>,
    pub path: Option<Vec<Point>>,
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

const EMPTY_COMMAND_EXTRAS: CommandExtras = CommandExtras {
    mode: None,
    path: None,
    ability: None,
    chosen_kind: None,
    ultimate: None,
    charge: None,
    offer_indices: None,
    player: None,
    shrine_kind: None,
    parity: None,
    recipe_id: None,
    material_ids: None,
    second_id: None,
    death_id: None,
    column: None,
    row: None,
    card_ids: None,
    sacrifice_ids: None,
};
impl std::ops::Deref for Command {
    type Target = CommandExtras;
    fn deref(&self) -> &Self::Target {
        self.extra.as_deref().unwrap_or(&EMPTY_COMMAND_EXTRAS)
    }
}
impl std::ops::DerefMut for Command {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Rc::make_mut(
            self.extra
                .get_or_insert_with(|| Rc::new(CommandExtras::default())),
        )
    }
}

impl Serialize for Command {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", &self.kind)?;
        macro_rules! optional {
            ($field:ident, $name:literal) => {
                if let Some(value) = &self.$field {
                    map.serialize_entry($name, value)?;
                }
            };
        }
        optional!(unit_id, "unitId");
        optional!(x, "x");
        optional!(y, "y");
        optional!(target_id, "targetId");
        optional!(direction, "direction");
        optional!(mode, "mode");
        optional!(path, "path");
        optional!(card_id, "cardId");
        optional!(ability, "ability");
        optional!(chosen_kind, "chosenKind");
        optional!(ultimate, "ultimate");
        optional!(charge, "charge");
        optional!(offer_indices, "offerIndices");
        optional!(player, "player");
        optional!(shrine_kind, "shrineKind");
        optional!(parity, "parity");
        optional!(recipe_id, "recipeId");
        optional!(material_ids, "materialIds");
        optional!(second_id, "secondId");
        optional!(death_id, "deathId");
        optional!(column, "column");
        optional!(row, "row");
        optional!(card_ids, "cardIds");
        optional!(sacrifice_ids, "sacrificeIds");
        map.end()
    }
}
impl<'de> Deserialize<'de> for Command {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Wire {
            #[serde(rename = "type")]
            kind: CommandKind,
            unit_id: Option<Rc<str>>,
            x: Option<f64>,
            y: Option<f64>,
            target_id: Option<Rc<str>>,
            direction: Option<String>,
            card_id: Option<Rc<str>>,
            #[serde(flatten)]
            extra: CommandExtras,
        }
        let wire = Wire::deserialize(deserializer)?;
        let e = &wire.extra;
        let empty = e.mode.is_none()
            && e.path.is_none()
            && e.ability.is_none()
            && e.chosen_kind.is_none()
            && e.ultimate.is_none()
            && e.charge.is_none()
            && e.offer_indices.is_none()
            && e.player.is_none()
            && e.shrine_kind.is_none()
            && e.parity.is_none()
            && e.recipe_id.is_none()
            && e.material_ids.is_none()
            && e.second_id.is_none()
            && e.death_id.is_none()
            && e.column.is_none()
            && e.row.is_none()
            && e.card_ids.is_none()
            && e.sacrifice_ids.is_none();
        Ok(Self {
            kind: wire.kind,
            unit_id: wire.unit_id,
            x: wire.x,
            y: wire.y,
            target_id: wire.target_id,
            direction: wire.direction.map(Self::shared_direction),
            card_id: wire.card_id,
            extra: (!empty).then(|| Rc::new(wire.extra)),
        })
    }
}

impl Command {
    /// 方向候选与协议输入共用静态已知值；未知方向原样保留拥有型字符串。
    pub fn shared_direction(value: String) -> Cow<'static, str> {
        match value.as_str() {
            "up" => Cow::Borrowed("up"),
            "down" => Cow::Borrowed("down"),
            "left" => Cow::Borrowed("left"),
            "right" => Cow::Borrowed("right"),
            _ => Cow::Owned(value),
        }
    }
    /// 内部构造不经过 JSON；可选参数保持缺失状态，外部边界仍独立校验。
    pub fn new(kind: &str) -> Self {
        Self {
            kind: kind.into(),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod command_tests {
    use super::*;

    #[test]
    fn command_fork_resets_events_and_preserves_snapshot_isolation_and_serial() {
        let catalog = crate::catalog::embedded().unwrap();
        let mut source = crate::runtime::create(73127, false, &catalog).unwrap();
        source.units.push(crate::resolution::template(
            &source,
            "26",
            1,
            Point { x: 3.0, y: 5.0 },
            "source",
            &catalog,
        ));
        source.events = vec![json!({"id":"previous","nested":{"value":"旧事件"}})];
        let frozen = serde_json::to_value(&source).unwrap();
        let mut regular = source.fork();
        assert_eq!(regular.events, source.events);
        regular.events[0]["nested"]["value"] = json!("普通分支事件");

        let mut expected = source.fork();
        expected.events.clear();
        let mut command = source.fork_command();
        assert_eq!(
            serde_json::to_value(&command).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        let serial = source.serial;
        crate::resolution::Resolution::default().emit(
            &mut command,
            json!({"type":"skill","text":"新命令事件"}),
            None,
        );
        assert_eq!(command.events.len(), 1);
        assert_eq!(command.events[0]["id"], format!("e{serial}"));
        assert_eq!(command.serial, serial + 1);
        command.units[0].hp -= 1.0;
        command.extra.insert("log".into(), json!(["命令分支日志"]));
        assert_eq!(serde_json::to_value(&source).unwrap(), frozen);
    }

    #[test]
    fn kind_text_sharing_preserves_serde_keys_hashes_and_entity_cow() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        #[derive(Deserialize, Serialize, Hash)]
        #[serde(untagged)]
        enum PreviousKind {
            Number(i32),
            Text(String),
        }
        let hash = |value: &dyn Fn(&mut DefaultHasher)| {
            let mut hasher = DefaultHasher::new();
            value(&mut hasher);
            hasher.finish()
        };
        for wire in ["1", "-4", r#""u13""#, r#""3p""#, r#""编号🦋""#, r#""12""#] {
            let actual: Kind = serde_json::from_str(wire).unwrap();
            let previous: PreviousKind = serde_json::from_str(wire).unwrap();
            assert_eq!(
                serde_json::to_string(&actual).unwrap(),
                serde_json::to_string(&previous).unwrap()
            );
            assert_eq!(hash(&|h| actual.hash(h)), hash(&|h| previous.hash(h)));
            assert_eq!(
                crate::canonical::hash(&actual),
                crate::canonical::hash(&serde_json::from_str::<Value>(wire).unwrap())
            );
            assert!(actual.is(actual.key_ref().as_ref()));
            assert_eq!(actual.key(), actual.key_ref());
            let sibling = actual.clone();
            assert_eq!(sibling, actual);
            if let Kind::Text(text) = &actual {
                let Kind::Text(shared) = &sibling else {
                    unreachable!()
                };
                assert!(Rc::ptr_eq(text, shared));
                assert!(matches!(actual.key_ref(), Cow::Borrowed(_)));
            } else {
                assert!(matches!(actual.key_ref(), Cow::Owned(_)));
            }
        }
        assert_ne!(Kind::Number(12), Kind::Text("12".into()));
        let catalog = crate::catalog::embedded().unwrap();
        let state = crate::runtime::create(73127, false, &catalog).unwrap();
        let source = crate::resolution::template(
            &state,
            "u13",
            1,
            Point { x: 3.0, y: 4.0 },
            "source",
            &catalog,
        );
        let mut branch = source.fork();
        branch.hp -= 1.0;
        let (Kind::Text(original), Kind::Text(shared)) = (&source.kind, &branch.kind) else {
            unreachable!()
        };
        assert!(Rc::ptr_eq(original, shared));
        branch.kind = Kind::Text("u19".into());
        assert!(source.kind.is("u13"));
        assert!(branch.kind.is("u19"));
        assert_ne!(branch.hp, source.hp);
    }

    #[test]
    fn sparse_commands_preserve_wire_fields_and_shared_write_boundaries() {
        let full = concat!(
            r#"{"type":"move","unitId":"u1","x":2.0,"y":3.0,"targetId":"u2","direction":"down","mode":"damage","path":[{"x":1.0,"y":2.0}],"#,
            r#""cardId":"c1","ability":"u7","chosenKind":25,"ultimate":false,"charge":false,"offerIndices":[0.0,1.0],"player":1,"#,
            r#""shrineKind":"s9","parity":"odd","recipeId":"firelord","materialIds":["u3"],"secondId":"u4","deathId":"d1","column":2.0,"row":3.0,"cardIds":["c2"],"sacrificeIds":["u5","u6"]}"#,
        );
        let source: Command = serde_json::from_str(full).unwrap();
        assert_eq!(serde_json::to_string(&source).unwrap(), full);
        let mut sibling = source.clone();
        assert!(Rc::ptr_eq(
            source.unit_id.as_ref().unwrap(),
            sibling.unit_id.as_ref().unwrap()
        ));
        assert!(Rc::ptr_eq(
            source.target_id.as_ref().unwrap(),
            sibling.target_id.as_ref().unwrap()
        ));
        assert!(Rc::ptr_eq(
            source.card_id.as_ref().unwrap(),
            sibling.card_id.as_ref().unwrap()
        ));
        sibling.x = Some(4.0);
        assert!(Rc::ptr_eq(
            source.extra.as_ref().unwrap(),
            sibling.extra.as_ref().unwrap()
        ));
        sibling.direction = Some("left".into());
        assert!(Rc::ptr_eq(
            source.extra.as_ref().unwrap(),
            sibling.extra.as_ref().unwrap()
        ));
        assert!(matches!(source.direction, Some(Cow::Borrowed("down"))));
        assert!(matches!(sibling.direction, Some(Cow::Borrowed("left"))));
        assert_eq!(source.direction.as_deref(), Some("down"));
        assert_eq!(sibling.direction.as_deref(), Some("left"));
        sibling.mode = Some("heal".into());
        assert!(!Rc::ptr_eq(
            source.extra.as_ref().unwrap(),
            sibling.extra.as_ref().unwrap()
        ));
        assert_eq!(source.mode.as_deref(), Some("damage"));
        sibling.target_id = Some(Rc::from("other-target"));
        assert_eq!(source.target_id.as_deref(), Some("u2"));
        assert_eq!(sibling.target_id.as_deref(), Some("other-target"));
        assert_eq!(serde_json::to_string(&source).unwrap(), full);
        for text in [
            r#"{"type":"move","unitId":"u1","x":1,"y":2}"#,
            r#"{"type":"move","unitId":"u1","x":1,"y":2,"direction":null,"path":null,"ability":null,"unknown":3}"#,
        ] {
            let command: Command = serde_json::from_str(text).unwrap();
            assert!(command.extra.is_none());
            assert_eq!(
                serde_json::to_value(command).unwrap(),
                json!({"type":"move","unitId":"u1","x":1.0,"y":2.0})
            );
        }
        for direction in ["up", "down", "left", "right", "未知方向🌸"] {
            let wire = json!({"type":"attack","direction":direction});
            let command: Command = serde_json::from_value(wire.clone()).unwrap();
            assert!(command.extra.is_none());
            assert_eq!(command.direction.as_deref(), Some(direction));
            assert_eq!(
                matches!(command.direction, Some(Cow::Owned(_))),
                direction == "未知方向🌸"
            );
            assert_eq!(serde_json::to_value(command).unwrap(), wire);
        }
        let mut hot_direction = Command::new("attack");
        hot_direction.direction = Some("up".into());
        assert!(hot_direction.extra.is_none());
        hot_direction.direction = None;
        assert_eq!(
            serde_json::to_value(hot_direction).unwrap(),
            json!({"type":"attack"})
        );
        let empty_arrays: Command =
            serde_json::from_value(json!({"type":"attack","path":[],"charge":false})).unwrap();
        assert_eq!(
            serde_json::to_value(empty_arrays).unwrap(),
            json!({"type":"attack","path":[],"charge":false})
        );
        let unknown: Command = serde_json::from_str(r#"{"type":"unknown-command"}"#).unwrap();
        assert_eq!(unknown.kind.as_str(), "unknown-command");
        assert_eq!(
            serde_json::to_value(unknown).unwrap(),
            json!({"type":"unknown-command"})
        );
        for value in [
            json!({}),
            json!({"type":"move","x":"1"}),
            json!({"type":"move","path":1}),
            json!({"type":"attack","direction":1}),
            json!({"type":"cast","ultimate":1}),
        ] {
            assert!(serde_json::from_value::<Command>(value).is_err());
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
