//! 既有规范字节流的类型化出口。固定字段按 UTF-8 次序列出；仅动态键排序。
//! flatten 扩展的同名键仍覆盖固定字段，缺失与 null、-0 与 0 的旧语义不变。
use crate::model::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, rc::Rc};

pub trait Canonical {
    fn write(&self, out: &mut Sha256);
}
pub fn hash(value: &impl Canonical) -> String {
    let mut out = Sha256::new();
    value.write(&mut out);
    format!("{:x}", out.finalize())
}
fn size(out: &mut Sha256, tag: u8, size: usize) {
    out.update([tag]);
    out.update((size as u64).to_le_bytes());
}
impl Canonical for str {
    fn write(&self, out: &mut Sha256) {
        size(out, 3, self.len());
        out.update(self.as_bytes());
    }
}
impl Canonical for String {
    fn write(&self, out: &mut Sha256) {
        self.as_str().write(out);
    }
}
impl Canonical for bool {
    fn write(&self, out: &mut Sha256) {
        out.update([1, u8::from(*self)]);
    }
}
impl Canonical for f64 {
    fn write(&self, out: &mut Sha256) {
        // serde_json 对非有限数值输出 null；入口规则仍负责拒绝非法状态。
        if !self.is_finite() {
            out.update([0]);
            return;
        }
        out.update([2]);
        out.update((if *self == 0.0 { 0.0 } else { *self }).to_le_bytes());
    }
}
macro_rules! numbers {
    ($($t:ty),*) => { $(impl Canonical for $t {
        fn write(&self, out: &mut Sha256) { (*self as f64).write(out); }
    })* };
}
numbers!(u8, u64, usize, i32);
impl<T: Canonical> Canonical for Rc<T> {
    fn write(&self, out: &mut Sha256) {
        self.as_ref().write(out);
    }
}
impl<T: Canonical> Canonical for Vec<T> {
    fn write(&self, out: &mut Sha256) {
        size(out, 4, self.len());
        for v in self {
            v.write(out);
        }
    }
}
type Field<'a> = (&'a str, Option<&'a dyn Canonical>);
fn field<'a, T: Canonical>(name: &'a str, value: &'a T) -> Field<'a> {
    (name, Some(value))
}
fn optional<'a, T: Canonical>(name: &'a str, value: &'a Option<T>) -> Field<'a> {
    (name, value.as_ref().map(|v| v as &dyn Canonical))
}
fn object<'a>(
    out: &mut Sha256,
    fixed: &[Field<'a>],
    extra: impl Iterator<Item = (&'a str, &'a dyn Canonical)>,
) {
    debug_assert!(fixed.windows(2).all(|p| p[0].0 < p[1].0));
    let mut extra: Vec<_> = extra.collect();
    extra.sort_unstable_by_key(|v| v.0);
    let count = extra.len()
        + fixed
            .iter()
            .filter(|(k, v)| v.is_some() && extra.binary_search_by_key(k, |e| e.0).is_err())
            .count();
    size(out, 5, count);
    let mut at = 0;
    for &(key, value) in fixed {
        while at < extra.len() && extra[at].0 < key {
            extra[at].0.write(out);
            extra[at].1.write(out);
            at += 1;
        }
        if at < extra.len() && extra[at].0 == key {
            extra[at].0.write(out);
            extra[at].1.write(out);
            at += 1;
        } else if let Some(value) = value {
            key.write(out);
            value.write(out);
        }
    }
    for &(key, value) in &extra[at..] {
        key.write(out);
        value.write(out);
    }
}
fn fields(out: &mut Sha256, values: &[Field<'_>]) {
    object(out, values, std::iter::empty());
}
impl<T: Canonical> Canonical for BTreeMap<String, T> {
    fn write(&self, out: &mut Sha256) {
        size(out, 5, self.len());
        for (key, value) in self {
            key.write(out);
            value.write(out);
        }
    }
}
impl<T: Canonical> Canonical for indexmap::IndexMap<String, T> {
    fn write(&self, out: &mut Sha256) {
        object(
            out,
            &[],
            self.iter().map(|(k, v)| (k.as_str(), v as &dyn Canonical)),
        );
    }
}
impl Canonical for Value {
    fn write(&self, out: &mut Sha256) {
        match self {
            Value::Null => out.update([0]),
            Value::Bool(v) => v.write(out),
            Value::Number(v) => v.as_f64().unwrap().write(out),
            Value::String(v) => v.write(out),
            Value::Array(v) => v.write(out),
            Value::Object(v) => object(
                out,
                &[],
                v.iter().map(|(k, v)| (k.as_str(), v as &dyn Canonical)),
            ),
        }
    }
}
impl Canonical for Kind {
    fn write(&self, out: &mut Sha256) {
        match self {
            Self::Number(n) => n.write(out),
            Self::Text(s) => s.write(out),
        }
    }
}
impl Canonical for ChargeMode {
    fn write(&self, out: &mut Sha256) {
        self.as_str().write(out);
    }
}
impl Canonical for Charge {
    fn write(&self, out: &mut Sha256) {
        fields(
            out,
            &[
                field("charge", &self.charge),
                field("chargeType", &self.charge_type),
                field("lastCharge", &self.last_charge),
                field("readyCharge", &self.ready_charge),
            ],
        );
    }
}
impl Canonical for AbilityUsage {
    fn write(&self, out: &mut Sha256) {
        fields(out, &[field("free", &self.free), field("once", &self.once)]);
    }
}
impl Canonical for Effect {
    fn write(&self, out: &mut Sha256) {
        fields(
            out,
            &[
                optional("amount", &self.amount),
                field("from", &self.from),
                optional("global", &self.global),
                field("owner", &self.owner),
                optional("sourceId", &self.source_id),
                field("type", &self.kind),
                field("until", &self.until),
            ],
        );
    }
}
impl Canonical for Unit {
    fn write(&self, out: &mut Sha256) {
        (**self).write(out);
    }
}
impl Canonical for UnitData {
    fn write(&self, out: &mut Sha256) {
        object(
            out,
            &[
                optional("abilityCharges", &self.ability_charges),
                optional("abilityUsage", &self.ability_usage),
                field("attackBonus", &self.attack_bonus),
                field("attacked", &self.attacked),
                field("bonusAttacks", &self.bonus_attacks),
                field("bonusSequence", &self.bonus_sequence),
                field("born", &self.born),
                field("charge", &self.reserve.charge),
                field("chargeType", &self.reserve.charge_type),
                field("chargedOnDeploy", &self.charged_on_deploy),
                field("deployedAt", &self.deployed_at),
                field("effects", &self.effects),
                field("equipment", &self.equipment),
                field("freeUsed", &self.free_used),
                field("guardUsed", &self.guard_used),
                field("hp", &self.hp),
                field("id", &self.id),
                field("kills", &self.kills),
                field("kind", &self.kind),
                field("lastCharge", &self.reserve.last_charge),
                field("maxHp", &self.max_hp),
                field("mode", &self.mode),
                field("moves", &self.moves),
                field("offset", &self.offset),
                field("onceUsed", &self.once_used),
                field("operations", &self.operations),
                field("owner", &self.owner),
                field("rangeBonus", &self.range_bonus),
                field("readyCharge", &self.reserve.ready_charge),
                field("shots", &self.shots),
                field("silenced", &self.silenced),
                field("size", &self.size),
                optional("traits", &self.traits),
                field("upgrades", &self.upgrades),
                field("weaponFirstUsed", &self.weapon_first_used),
                field("x", &self.x),
                field("y", &self.y),
            ],
            self.extra
                .iter()
                .map(|(k, v)| (k.as_str(), v as &dyn Canonical)),
        );
    }
}
impl Canonical for Reaction {
    fn write(&self, out: &mut Sha256) {
        fields(
            out,
            &[
                field("amount", &self.amount),
                field("kind", &self.kind),
                field("owner", &self.owner),
                field("source", &self.source),
                optional("targetId", &self.target_id),
            ],
        );
    }
}
impl Canonical for ClockFrame {
    fn write(&self, out: &mut Sha256) {
        fields(
            out,
            &[
                field("ply", &self.ply),
                field("turns", &self.turns),
                field("units", &self.units),
            ],
        );
    }
}
impl Canonical for ClockFrames {
    fn write(&self, out: &mut Sha256) {
        fields(
            out,
            &[
                optional("current", &self.current),
                optional("previous", &self.previous),
            ],
        );
    }
}
impl Canonical for State {
    fn write(&self, out: &mut Sha256) {
        object(
            out,
            &[
                field("active", &self.active),
                field("baseEffects", &self.base_effects),
                field("bases", &self.bases),
                optional("clockFrames", &self.clock_frames),
                field("deployRows", &self.deploy_rows),
                field("events", &self.events),
                optional("landmarks", &self.landmarks),
                field("pending", &self.pending),
                field("phase", &self.phase),
                field("ply", &self.ply),
                field("serial", &self.serial),
                field("siphons", &self.siphons),
                field("summonSlots", &self.summon_slots),
                field("turns", &self.turns),
                field("units", &self.units),
                field("version", &self.version),
            ],
            self.extra.iter().map(|(k, v)| (k, v as &dyn Canonical)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn same(state: &State) {
        assert_eq!(
            hash(state),
            crate::records::hash(&serde_json::to_value(state).unwrap())
        );
    }

    #[test]
    fn typed_stream_matches_value_tree_for_presence_extensions_snapshots_and_rollback() {
        let mut catalog = None;
        crate::handle(
            serde_json::from_slice(crate::records::RULES).unwrap(),
            &mut catalog,
            &mut vec![],
            &mut crate::Resident::default(),
        )
        .unwrap();
        let catalog = catalog.unwrap();
        for shrine in [false, true] {
            let mut s = crate::runtime::create(8137, shrine, &catalog).unwrap();
            for (kind, owner, x) in [("u17", 1, 3.0), ("18", 2, 5.0)] {
                crate::resolution::add_unit(
                    &mut s,
                    kind,
                    owner,
                    Point { x, y: 6.0 },
                    &catalog,
                    &mut crate::resolution::Resolution::default(),
                )
                .unwrap();
            }
            let u = &mut s.units[0];
            u.x = -0.0;
            u.traits = Some(vec![Kind::Number(15), Kind::Text("u13".into())]);
            u.ability_usage = Some(indexmap::IndexMap::from_iter([(
                "u13".into(),
                AbilityUsage {
                    once: false,
                    free: 0.0,
                },
            )]));
            u.ability_charges = Some(indexmap::IndexMap::from_iter([(
                "15".into(),
                Charge {
                    charge: 1.5,
                    ready_charge: -0.0,
                    charge_type: ChargeMode::Skill,
                    last_charge: 13.0,
                },
            )]));
            u.effects.push(Effect {
                kind: "attack".into(),
                from: -0.0,
                until: 17.0,
                owner: 1,
                amount: Some(0.0),
                source_id: Some("来源🎮".into()),
                global: Some(false),
            });
            for key in ["", "0", "10", "2", "é", "\u{e000}", "😀"] {
                u.extra.insert(
                    key.into(),
                    json!({"null":null,"false":false,"zero":-0.0,"list":[1,"中",{}]}),
                );
                s.extra.insert(key.into(), json!([true, false, null, -0.0]));
            }
            s.pending.push(Reaction::new("reflect", &s.units[0], 5.0));
            s.pending[0].target_id = Some("外部目标".into());
            let frame = Rc::new(ClockFrame {
                ply: s.ply,
                turns: s.turns.clone(),
                units: s.units.clone(),
            });
            s.clock_frames = Some(BTreeMap::from([(
                "1".into(),
                ClockFrames {
                    current: Some(frame.clone()),
                    previous: Some(frame),
                },
            )]));
            same(&s);
            let saved = hash(&s);
            let mut snapshot = s.units[0].clone();
            let frozen = serde_json::to_value(&s).unwrap();
            snapshot.effects[0].amount = Some(99.0);
            snapshot.traits.as_mut().unwrap().clear();
            snapshot.ability_usage.as_mut().unwrap()["u13"].once = true;
            snapshot.ability_charges.as_mut().unwrap()["15"].charge = 99.0;
            snapshot.extra.get_mut("é").unwrap()["list"][1] = json!("快照独立写入");
            snapshot.equipment.push(Kind::Text("s2".into()));
            snapshot.attacked.push("目标".into());
            assert_eq!(serde_json::to_value(&s).unwrap(), frozen);
            let snapshot_value = serde_json::to_value(&snapshot).unwrap();
            s.units[0].extra.get_mut("é").unwrap()["list"][1] = json!("局面独立写入");
            assert_eq!(serde_json::to_value(&snapshot).unwrap(), snapshot_value);
            s = serde_json::from_value(frozen).unwrap();
            let mut draft = s.fork();
            draft.units[0].extra.insert("new".into(), json!(true));
            draft.units[0].effects[0].amount = None;
            draft.extra.insert("hands".into(), json!({"1":[],"2":[]}));
            same(&draft);
            assert_ne!(hash(&draft), saved);
            assert_eq!(hash(&s), saved);
            let loaded: State = serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
            assert_eq!(hash(&loaded), saved);
            for p in [1, 2] {
                same(crate::runtime::public_view(&s, p).unwrap().position());
            }
            // flatten 的覆盖次序也属于旧规范，即使可信调用者传入同名扩展。
            s.extra.insert("active".into(), json!(77));
            s.units[0].extra.insert("hp".into(), json!(null));
            same(&s);
        }
    }
}
