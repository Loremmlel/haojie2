//! 普通局三档手工策略的 Rust 移植；逐模块与 TS 固定工作量结果对照。
pub mod candidates;
pub mod distribution;
pub mod evaluation;
pub mod keys;
pub mod ranking;
pub mod search;
pub mod settings;
pub mod spatial;
pub mod threats;

use crate::model::{Catalog, State, Unit};
pub fn passive(_s: &State, u: &Unit) -> bool {
    !u.silenced
}
pub fn hp(s: &State, side: usize) -> f64 {
    s.bases[side.to_string()].as_f64().unwrap_or(0.)
}
pub fn stored(c: &Catalog, k: &crate::model::Kind) -> bool {
    let d = c.by_kind(k);
    d.spell.is_some() || d.weapon.is_some() || d.aura
}
