//! 逐局状态机。仅保存公开搜索事实；规则状态与两个 RNG 不进入模型输入。
use crate::{boundary, catalog, encoding, model::State, runtime, tree::Tree};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
mod actor;
#[cfg(test)]
mod tests;

struct Random(u32);
impl Random {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f64 + 0.5) / 4294967296.0
    }
}
#[cfg(not(feature = "kernel-profile"))]
#[unsafe(no_mangle)]
pub extern "C" fn hj_memory_bytes() -> usize {
    crate::allocation::live_bytes()
}
#[unsafe(no_mangle)]
pub extern "C" fn hj_identity() -> *const std::ffi::c_char {
    static ID: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    ID.get_or_init(|| std::ffi::CString::new(crate::identity::describe().to_string()).unwrap())
        .as_ptr()
}
#[derive(Default, Serialize, Deserialize)]
struct Search {
    cursor: Vec<usize>,
    #[serde(with = "exclusions")]
    excluded: BTreeMap<Vec<usize>, Vec<usize>>,
    actor: usize,
    optional: bool,
}
mod exclusions {
    use super::*;
    pub fn serialize<S: serde::Serializer>(
        value: &BTreeMap<Vec<usize>, Vec<usize>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        value.iter().collect::<Vec<_>>().serialize(s)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<Vec<usize>, Vec<usize>>, D::Error> {
        Ok(Vec::<(Vec<usize>, Vec<usize>)>::deserialize(d)?
            .into_iter()
            .collect())
    }
}
#[derive(Serialize, Deserialize)]
struct SavedGame {
    id: u64,
    state: State,
    random: u32,
    learner: usize,
    opponent: u64,
    root_id: u64,
    commands: u64,
    microsteps: u64,
    failures: u64,
    backtracks: u64,
    offer_interrupt: bool,
    search: Option<Search>,
    reported_terminal: bool,
    waiting: bool,
}
struct Game<'a> {
    saved: SavedGame,
    tree: Option<Tree<'a>>,
    sent_root: u64,
}
impl<'a> Game<'a> {
    fn start_root(&mut self, cat: &'a crate::model::Catalog) -> Result<(), String> {
        let s = &self.saved.state;
        let mut actor = runtime::viewer(s);
        if s.phase == "shrine-draft"
            && s.extra["shrineDraft"]["committed"][actor.to_string()] == true
        {
            actor = 3 - actor;
        }
        let other = 3 - actor;
        let optional = self.saved.offer_interrupt
            && s.pending.is_empty()
            && s.phase != "shrine-draft"
            && s.pieces().any(|u| u.owner == other && u.has("u7"));
        if optional {
            actor = other;
        }
        self.saved.search = Some(Search {
            actor,
            optional,
            ..Search::default()
        });
        self.saved.root_id += 1;
        self.tree = Some(Tree::own_view(runtime::public_view(s, actor)?, cat)?);
        Ok(())
    }
    fn finish_root(&mut self, next: Option<State>) {
        let optional = self.saved.search.as_ref().unwrap().optional;
        if let Some(state) = next {
            self.saved.state = state;
            self.saved.commands += 1;
        }
        self.saved.offer_interrupt = !optional;
        self.saved.search = None;
        self.tree = None;
    }
    /// 一次只消费一个随机回复，至多执行 64 个确定性节点，随后让出 CPU lane。
    fn advance(
        &mut self,
        mut probabilities: Option<Vec<f64>>,
        cat: &'a crate::model::Catalog,
    ) -> Result<Vec<u8>, String> {
        let mut selected = -1i32;
        let mut logp = 0.0f64;
        for _ in 0..64 {
            if let Some(winner) = self.saved.state.extra.get("winner") {
                let winner = if winner == "draw" {
                    0
                } else {
                    winner.as_u64().ok_or("invalid terminal winner")? as i32
                };
                if probabilities.is_some() {
                    return Err("probabilities after terminal".into());
                }
                self.saved.reported_terminal = true;
                return Ok(self.frame(1, winner, selected, logp));
            }
            if self.saved.search.is_none() {
                self.start_root(cat)?;
            }
            if self.tree.is_none() {
                let actor = self.saved.search.as_ref().unwrap().actor;
                self.tree = Some(Tree::own_view(
                    runtime::public_view(&self.saved.state, actor)?,
                    cat,
                )?);
            }
            let tree = self.tree.as_mut().unwrap();
            let search = self.saved.search.as_mut().unwrap();
            tree.retain_path(&search.cursor);
            let node = tree.node(&search.cursor)?;
            let pass = search.optional && search.cursor.is_empty();
            if node.choices.is_empty() && pass {
                self.finish_root(None);
                continue;
            }
            let count = node.choices.len() + usize::from(pass);
            let excluded = search.excluded.entry(search.cursor.clone()).or_default();
            let remaining: Vec<_> = (0..count).filter(|i| !excluded.contains(i)).collect();
            if remaining.is_empty() {
                search.excluded.remove(&search.cursor);
                let Some(index) = search.cursor.pop() else {
                    return Err("no legal action at root".into());
                };
                search
                    .excluded
                    .entry(search.cursor.clone())
                    .or_default()
                    .push(index);
                self.saved.backtracks += 1;
                continue;
            }
            let index = if remaining.len() == 1 {
                if probabilities.is_some() {
                    return Err("probabilities for forced node".into());
                }
                remaining[0]
            } else if let Some(probs) = probabilities.take() {
                if probs.len() != count
                    || probs.iter().any(|v| !v.is_finite() || *v < 0.0)
                    || excluded.iter().any(|&i| probs[i] != 0.0)
                {
                    return Err("invalid probability support".into());
                }
                let sum: f64 = probs.iter().sum();
                if sum <= 0.0 {
                    return Err("empty probability mass".into());
                }
                self.saved.waiting = false;
                let mut rng = Random::new(self.saved.random);
                let draw = rng.next() * sum;
                self.saved.random = rng.0;
                let mut cumulative = 0.0;
                let index = probs
                    .iter()
                    .position(|p| {
                        cumulative += p;
                        draw < cumulative
                    })
                    .ok_or("probability rounding overflow")?;
                selected = index as i32;
                logp = (probs[index] / sum).ln();
                self.saved.microsteps += 1;
                index
            } else {
                return Ok(self.frame(0, -1, selected, logp));
            };
            let search = self.saved.search.as_mut().unwrap();
            if pass && index == node.choices.len() {
                self.finish_root(None);
                continue;
            }
            let choice = &node.choices[index];
            if choice.next.is_some() {
                search.cursor.push(index);
                continue;
            }
            match boundary::attempt(
                &self.saved.state,
                search.actor,
                &choice.command,
                choice.status,
                cat,
            )? {
                Ok(next) => self.finish_root(Some(next)),
                Err(_) => {
                    search
                        .excluded
                        .entry(search.cursor.clone())
                        .or_default()
                        .push(index);
                    self.saved.failures += 1;
                }
            }
        }
        Ok(self.frame(2, -1, selected, logp))
    }
    fn frame(&mut self, kind: u32, winner: i32, selected: i32, logp: f64) -> Vec<u8> {
        self.saved.waiting = kind == 0;
        let mut bytes = Vec::with_capacity(84);
        for n in [
            self.saved.id,
            self.saved.root_id,
            self.saved.opponent,
            self.saved.commands,
            self.saved.microsteps,
            self.saved.failures,
            self.saved.backtracks,
        ] {
            bytes.extend(n.to_le_bytes());
        }
        let actor = self.saved.search.as_ref().map_or(0, |s| s.actor);
        for n in [actor as u32, self.saved.learner as u32, kind] {
            bytes.extend(n.to_le_bytes());
        }
        bytes.extend(winner.to_le_bytes());
        bytes.extend(selected.to_le_bytes());
        bytes.extend(logp.to_le_bytes());
        bytes
    }
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let (head, tail) = self
            .0
            .split_at_checked(N)
            .ok_or("truncated native packet")?;
        self.0 = tail;
        Ok(head.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take()?))
    }
}
