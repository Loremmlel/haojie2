//! 批量 FFI、线程内拥有型对局和可序列化动作状态机。JSON 仅用于一致快照。
use crate::{boundary, catalog, encoding, model::State, runtime, tree::Tree};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};

#[cfg(test)]
#[path = "tests.rs"]
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
    #[serde(rename = "pending_events")]
    events: Vec<Vec<f64>>,
    sent_events: usize,
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
    announced: bool,
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
        self.announced = false;
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
        self.announced = false;
    }
    /// 每次最多推进 64 个确定性节点，确保保存和其他对局有服务机会。
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
                return Ok(self.frame(1, winner, selected, logp, None));
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
                let mut event = vec![0.0; 68];
                event[64] = -2.0;
                event[65] = search.cursor.len() as f64 / 256.0;
                event[66] = index as f64 / 4096.0;
                search.events.push(event);
                self.saved.backtracks += 1;
                continue;
            }
            let mut input = (*encoding::encode_sampling(tree, &node)?).clone();
            if pass {
                crate::records::add_pass(&mut input);
            }
            for &index in excluded.iter() {
                input.candidate_mask[index] = false;
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
                let result = self.frame(0, -1, selected, logp, Some(&input));
                self.announced = true;
                return Ok(result);
            };
            let search = self.saved.search.as_mut().unwrap();
            let mut event = input.candidates[index].to_vec();
            event.extend([
                1.0,
                search.cursor.len() as f64 / 256.0,
                index as f64 / 4096.0,
                (search.sent_events + search.events.len()) as f64 / 4096.0,
            ]);
            search.events.push(event);
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
                    let mut event = input.candidates[index].to_vec();
                    event.extend([
                        -1.0,
                        search.cursor.len() as f64 / 256.0,
                        index as f64 / 4096.0,
                        (search.sent_events + search.events.len()) as f64 / 4096.0,
                    ]);
                    search.events.push(event);
                    self.saved.failures += 1;
                }
            }
        }
        Ok(self.frame(2, -1, selected, logp, None))
    }
    fn frame(
        &mut self,
        kind: u32,
        winner: i32,
        selected: i32,
        logp: f64,
        input: Option<&encoding::Input>,
    ) -> Vec<u8> {
        self.saved.waiting = kind == 0;
        let mut bytes = vec![];
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
        let root = input.is_some() && !self.announced;
        let entities = input.map_or(0, |i| i.entities.fixed_len());
        let prefix = input.map_or(0, |i| i.entities.tail.len());
        let candidates = input.map_or(0, |i| i.candidates.len());
        let (event_start, events) = input.map_or((0, 0), |_| {
            let search = self.saved.search.as_ref().unwrap();
            (search.sent_events, search.events.len())
        });
        for n in [
            u32::from(root),
            entities as u32,
            prefix as u32,
            candidates as u32,
            events as u32,
            event_start as u32,
        ] {
            bytes.extend(n.to_le_bytes());
        }
        let floats = |bytes: &mut Vec<u8>, values: &[f64]| {
            for &n in values {
                bytes.extend(encoding::bf16_bits(n).to_le_bytes());
            }
        };
        if let Some(input) = input {
            if root {
                for row in input.entities.iter().take(entities) {
                    floats(&mut bytes, row);
                }
                for &n in input.kinds.iter().take(entities) {
                    bytes.extend((n as u32).to_le_bytes());
                }
                let mut globals = input.globals.clone();
                globals[31] = 0.0;
                floats(&mut bytes, &globals);
            }
            for row in &input.entities.tail {
                floats(&mut bytes, row);
            }
            for row in &input.candidates {
                floats(&mut bytes, row);
            }
            for &n in input.sources.iter().chain(&input.targets) {
                bytes.extend(n.to_le_bytes());
            }
            bytes.extend(input.candidate_mask.iter().map(|&v| u8::from(v)));
            let search = self.saved.search.as_mut().unwrap();
            for event in &search.events {
                floats(&mut bytes, event);
            }
            // 已交给所有者的历史由共享 BF16 缓冲保存，快照与原生状态一起提交。
            // 原生只保留尚未交接的事件，避免长根在两端重复积累完整历史。
            search.sent_events += search.events.len();
            search.events.clear();
        }
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
enum Job {
    Advance(Vec<u8>),
    Save,
    Restore(Vec<u8>),
}
struct Handle {
    tx: Option<SyncSender<Job>>,
    rx: Receiver<Result<Vec<u8>, String>>,
    thread: Option<JoinHandle<()>>,
    output: Vec<u8>,
    busy: bool,
    failed: bool,
}
fn worker(rx: Receiver<Job>, tx: SyncSender<Result<Vec<u8>, String>>) {
    let cat = match catalog::embedded() {
        Ok(cat) => cat,
        Err(error) => {
            let _ = tx.send(Err(error));
            return;
        }
    };
    let mut games: BTreeMap<u64, Game<'_>> = BTreeMap::new();
    while let Ok(job) = rx.recv() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<Vec<u8>, String> {
                match job {
                    Job::Save => {
                        serde_json::to_vec(&games.values().map(|g| &g.saved).collect::<Vec<_>>())
                            .map_err(|e| e.to_string())
                    }
                    Job::Restore(bytes) => {
                        let saved: Vec<SavedGame> =
                            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                        if !games.is_empty() {
                            return Err("restore requires empty runtime".into());
                        }
                        for saved in saved {
                            games.insert(
                                saved.id,
                                Game {
                                    saved,
                                    tree: None,
                                    announced: false,
                                },
                            );
                        }
                        Ok(vec![])
                    }
                    Job::Advance(bytes) => {
                        games.retain(|_, g| !g.saved.reported_terminal);
                        let mut reader = Reader(&bytes);
                        let admissions = reader.u32()?;
                        for _ in 0..admissions {
                            let id = reader.u64()?;
                            let seed = reader.u32()?;
                            let learner = reader.u32()? as usize;
                            let opponent = reader.u64()?;
                            let shrine = reader.u32()?;
                            if games.contains_key(&id) || ![1, 2].contains(&learner) || shrine > 1 {
                                return Err("invalid admission".into());
                            }
                            let saved = SavedGame {
                                id,
                                state: runtime::create(seed, shrine == 1, &cat)?,
                                random: seed ^ 0xa0761d65,
                                learner,
                                opponent,
                                root_id: 0,
                                commands: 0,
                                microsteps: 0,
                                failures: 0,
                                backtracks: 0,
                                offer_interrupt: true,
                                search: None,
                                reported_terminal: false,
                                waiting: false,
                            };
                            games.insert(
                                id,
                                Game {
                                    saved,
                                    tree: None,
                                    announced: false,
                                },
                            );
                        }
                        let replies = reader.u32()?;
                        let mut probabilities = BTreeMap::new();
                        for _ in 0..replies {
                            let id = reader.u64()?;
                            let count = reader.u32()?;
                            let mut values = Vec::with_capacity(count as usize);
                            for _ in 0..count {
                                values
                                    .push(encoding::bf16_value(u16::from_le_bytes(reader.take()?)));
                            }
                            if !games.contains_key(&id)
                                || probabilities.insert(id, values).is_some()
                            {
                                return Err("unknown or duplicate reply".into());
                            }
                        }
                        if !reader.0.is_empty() {
                            return Err("trailing native packet".into());
                        }
                        let mut frames = vec![];
                        let mut count = 0u32;
                        for game in games.values_mut() {
                            // 未获得推理额度的局保持同一节点；不重复发帧，也不重新抽样。
                            if game.saved.waiting && !probabilities.contains_key(&game.saved.id) {
                                continue;
                            }
                            let frame = game.advance(probabilities.remove(&game.saved.id), &cat)?;
                            count += 1;
                            frames.extend((frame.len() as u32).to_le_bytes());
                            frames.extend(frame);
                        }
                        let mut output = count.to_le_bytes().to_vec();
                        output.extend(frames);
                        Ok(output)
                    }
                }
            },
        ))
        .unwrap_or_else(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("non-string Rust panic");
            Err(format!("native worker panic: {message}"))
        });
        // 故障不伪装成可继续训练的快照；保留所有局面与根内状态供诊断。
        let result = result.map_err(|error| {
            serde_json::to_string(&serde_json::json!({
                "error": error,
                "games": games.values().map(|g| &g.saved).collect::<Vec<_>>()
            }))
            .unwrap_or_else(|serialization| {
                format!("{error}; diagnostic serialization: {serialization}")
            })
        });
        let failed = result.is_err();
        if tx.send(result).is_err() || failed {
            break;
        }
    }
}

/// 句柄只由 GPU 所有者线程调用；跨线程消息不包含 State、Tree 或 Rc。
#[unsafe(no_mangle)]
pub extern "C" fn hj_create() -> *mut std::ffi::c_void {
    let (tx, rx) = mpsc::sync_channel(1);
    let (out_tx, out_rx) = mpsc::sync_channel(1);
    let thread = thread::spawn(move || worker(rx, out_tx));
    Box::into_raw(Box::new(Handle {
        tx: Some(tx),
        rx: out_rx,
        thread: Some(thread),
        output: vec![],
        busy: false,
        failed: false,
    }))
    .cast()
}
/// 返回 0 表示已接受；调用方必须等待 poll 完成才能再提交或读取快照。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hj_submit(
    handle: *mut std::ffi::c_void,
    op: u32,
    data: *const u8,
    len: usize,
) -> i32 {
    let Some(h) = (unsafe { handle.cast::<Handle>().as_mut() }) else {
        return -1;
    };
    if h.busy || h.failed || (len != 0 && data.is_null()) {
        return -1;
    }
    let bytes = if len == 0 {
        vec![]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }.to_vec()
    };
    let job = match op {
        0 => Job::Advance(bytes),
        1 => Job::Save,
        2 => Job::Restore(bytes),
        _ => return -1,
    };
    if h.tx.as_ref().unwrap().send(job).is_err() {
        return -1;
    }
    h.busy = true;
    0
}
/// 0 尚未完成，1 完成，-1 错误；返回缓冲保持到本句柄下一次 poll 完成为止。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hj_poll(
    handle: *mut std::ffi::c_void,
    data: *mut *const u8,
    len: *mut usize,
) -> i32 {
    let Some(h) = (unsafe { handle.cast::<Handle>().as_mut() }) else {
        return -1;
    };
    if !h.busy {
        return -1;
    }
    match h.rx.try_recv() {
        Err(TryRecvError::Empty) => return 0,
        Err(TryRecvError::Disconnected) => {
            h.output = b"native worker disconnected".to_vec();
            h.failed = true;
        }
        Ok(Err(error)) => {
            h.output = error.into_bytes();
            h.failed = true;
        }
        Ok(Ok(bytes)) => h.output = bytes,
    }
    h.busy = false;
    unsafe {
        *data = h.output.as_ptr();
        *len = h.output.len();
    }
    if h.failed { -1 } else { 1 }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hj_close(handle: *mut std::ffi::c_void) {
    if handle.is_null() {
        return;
    }
    let mut h = unsafe { Box::from_raw(handle.cast::<Handle>()) };
    h.tx.take();
    // 仍有输出时先释放接收端，避免工作线程阻塞在有界发送上。
    drop(h.rx);
    if let Some(thread) = h.thread.take() {
        let _ = thread.join();
    }
}
