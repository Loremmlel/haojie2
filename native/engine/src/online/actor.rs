//! C++ lane 拥有的持久 actor；一次调用只推进指定对局，不存在 worker 整批屏障。
use super::*;
use std::ffi::c_void;

#[derive(Default)]
struct SearchStates {
    values: Vec<Option<SavedGame>>,
    free: Vec<usize>,
}

struct Actor {
    // 必须先销毁树，再释放它们借用的、地址稳定的目录。
    games: BTreeMap<u64, Game<'static>>,
    search_states: BTreeMap<u64, SearchStates>,
    cat: Box<crate::model::Catalog>,
    owner: std::thread::ThreadId,
    output: Vec<u8>,
}

impl Actor {
    fn call(&mut self, op: u32, bytes: &[u8]) -> Result<Vec<u8>, String> {
        if self.owner != std::thread::current().id() {
            return Err("actor used outside owning lane".into());
        }
        let mut input = Reader(bytes);
        // 目录由 Box 固定地址持有；games 在 cat 前销毁，且句柄只由 owner 使用。
        let cat: &'static crate::model::Catalog = unsafe { &*(&*self.cat as *const _) };
        match op {
            0 => {
                let id = input.u64()?;
                let seed = input.u32()?;
                let learner = input.u32()? as usize;
                let shrine = input.u32()?;
                if !input.0.is_empty()
                    || self.games.contains_key(&id)
                    || ![1, 2].contains(&learner)
                    || shrine > 1
                {
                    return Err("invalid actor admission".into());
                }
                self.games.insert(
                    id,
                    Game {
                        saved: SavedGame {
                            id,
                            state: runtime::create(seed, shrine == 1, cat)?,
                            random: seed ^ 0xa0761d65,
                            learner,
                            opponent: 0,
                            root_id: 0,
                            commands: 0,
                            microsteps: 0,
                            failures: 0,
                            backtracks: 0,
                            offer_interrupt: true,
                            search: None,
                            reported_terminal: false,
                            waiting: false,
                        },
                        tree: None,
                        sent_root: 0,
                        last_path: vec![],
                    },
                );
                Ok(vec![])
            }
            1 | 6 | 10 => {
                let id = input.u64()?;
                let count = input.u32()?;
                let mut probabilities = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    probabilities.push(encoding::bf16_value(u16::from_le_bytes(input.take()?)));
                }
                if !input.0.is_empty() {
                    return Err("trailing actor reply".into());
                }
                let game = self.games.get_mut(&id).ok_or("unknown actor")?;
                let masses = (count != 0).then_some(probabilities);
                let frame = if op == 10 {
                    if game.saved.state.extra.contains_key("winner") {
                        game.advance_until(None, cat, true)?
                    } else {
                        if game.saved.search.is_none() {
                            game.start_root(cat)?;
                        }
                        if game.tree.is_none() {
                            let actor = game.saved.search.as_ref().unwrap().actor;
                            game.tree = Some(crate::tree::Tree::own_view(
                                runtime::public_view(&game.saved.state, actor)?,
                                cat,
                            )?);
                        }
                        game.frame(0, -1, -1, 0.0)
                    }
                } else if op == 6 {
                    game.advance_until(masses, cat, true)?
                } else {
                    game.advance(masses, cat)?
                };
                let mut output = frame;
                if game.saved.waiting {
                    let search = game.saved.search.as_ref().unwrap();
                    let tree = game.tree.as_mut().unwrap();
                    let node = tree.node(&search.cursor)?;
                    let mut encoded = encoding::encode_sampling(tree, &node)?;
                    if search.optional && search.cursor.is_empty() {
                        crate::records::add_pass(&mut encoded);
                    }
                    if let Some(excluded) = search.excluded.get(&search.cursor) {
                        for &i in excluded {
                            encoded.candidate_mask[i] = false;
                        }
                    }
                    let has_root = game.sent_root != game.saved.root_id;
                    output.extend(u32::from(has_root).to_le_bytes());
                    for n in [
                        encoded.entities.fixed_len(),
                        encoded.entities.tail.len(),
                        encoded.candidates.len(),
                        search.cursor.len(),
                        search.excluded.len(),
                    ] {
                        output.extend((n as u32).to_le_bytes());
                    }
                    for row in encoded.entities.iter().skip(if has_root {
                        0
                    } else {
                        encoded.entities.fixed_len()
                    }) {
                        row.write(&mut output);
                    }
                    if has_root {
                        for &kind in encoded.kinds.iter().take(encoded.entities.fixed_len()) {
                            output.extend((kind as u32).to_le_bytes());
                        }
                        encoded.globals.write(&mut output);
                    }
                    game.sent_root = game.saved.root_id;
                    for row in &encoded.candidates {
                        for value in row {
                            output.extend(value.to_le_bytes());
                        }
                    }
                    for &index in encoded.sources.iter().chain(&encoded.targets) {
                        output.extend(index.to_le_bytes());
                    }
                    output.extend(encoded.candidate_mask.iter().map(|&v| u8::from(v)));
                    for &index in &search.cursor {
                        output.extend((index as u32).to_le_bytes());
                    }
                    // 整个排除 frontier 按层保存精确路径；不含未来随机数或暗选内容。
                    for (path, excluded) in &search.excluded {
                        output.extend((path.len() as u32).to_le_bytes());
                        for &index in path {
                            output.extend((index as u32).to_le_bytes());
                        }
                        output.extend((excluded.len() as u32).to_le_bytes());
                        for &index in excluded {
                            output.extend((index as u32).to_le_bytes());
                        }
                    }
                }
                Ok(output)
            }
            // 停放会移走整局及其树缓存；返回的自包含快照交由 C++ 保存。
            2 | 3 => {
                let id = input.u64()?;
                if !input.0.is_empty() {
                    return Err("trailing snapshot request".into());
                }
                let game = self.games.get(&id).ok_or("unknown actor")?;
                let saved = serde_json::to_vec(&game.saved).map_err(|e| e.to_string())?;
                if op == 3 {
                    self.games.remove(&id);
                }
                Ok(saved)
            }
            4 => {
                let saved: SavedGame = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
                if self.games.contains_key(&saved.id) {
                    return Err("duplicate restored actor".into());
                }
                self.games.insert(
                    saved.id,
                    Game {
                        saved,
                        tree: None,
                        sent_root: 0,
                        last_path: vec![],
                    },
                );
                Ok(vec![])
            }
            5 => {
                let id = input.u64()?;
                if !input.0.is_empty() || self.games.remove(&id).is_none() {
                    return Err("unknown actor removal".into());
                }
                self.search_states.remove(&id);
                Ok(vec![])
            }
            // 测评在根边界交替模型与真实游戏 AI；只向对手提供公开观察。
            7 => {
                let id = input.u64()?;
                if !input.0.is_empty() {
                    return Err("trailing evaluation query".into());
                }
                let game = self.games.get_mut(&id).ok_or("unknown actor")?;
                let terminal = game.saved.state.extra.get("winner").cloned();
                if terminal.is_none() && game.saved.search.is_none() {
                    game.start_root(cat)?;
                }
                let search = game.saved.search.as_ref();
                let actor = search.map_or(0, |s| s.actor);
                let observation = if terminal.is_none() {
                    Some(runtime::observe(&game.saved.state, actor)?)
                } else {
                    None
                };
                serde_json::to_vec(&serde_json::json!({
                    "winner": terminal, "actor": actor,
                    "optional": search.is_some_and(|s| s.optional),
                    "observation": observation, "commands": game.saved.commands,
                    "microsteps": game.saved.microsteps,
                }))
                .map_err(|e| e.to_string())
            }
            8 => {
                let id = input.u64()?;
                let command: serde_json::Value =
                    serde_json::from_slice(input.0).map_err(|e| e.to_string())?;
                let game = self.games.get_mut(&id).ok_or("unknown actor")?;
                let search = game.saved.search.as_ref().ok_or("no evaluation root")?;
                if !search.cursor.is_empty() || game.saved.waiting {
                    return Err("teacher command outside root boundary".into());
                }
                let next = if command.is_null() {
                    if !search.optional {
                        return Err("pass outside optional interrupt".into());
                    }
                    None
                } else {
                    Some(boundary::step(
                        &game.saved.state,
                        search.actor,
                        &command,
                        cat,
                    )?)
                };
                game.finish_root(next);
                Ok(vec![])
            }
            // 搜索分支只能由公开视图建立；真实 RNG 和敌方暗选不进入分支。
            9 => {
                let source = input.u64()?;
                let id = input.u64()?;
                let game = self.games.get(&source).ok_or("unknown search source")?;
                if id >> 63 == 0 || self.games.contains_key(&id) || !input.0.is_empty() {
                    return Err("invalid search fork".into());
                }
                let actor = game
                    .saved
                    .search
                    .as_ref()
                    .ok_or("missing search root")?
                    .actor;
                // 暗选仍按当前信息集直接采样，不补造对手尚未公开的选择。
                if game.saved.state.phase == "shrine-draft" {
                    return Ok(vec![0]);
                }
                let saved = SavedGame {
                    id,
                    state: runtime::public_view(&game.saved.state, actor)?.into_position(),
                    random: 1,
                    learner: game.saved.learner,
                    opponent: game.saved.opponent,
                    root_id: game.saved.root_id,
                    commands: 0,
                    microsteps: 0,
                    failures: 0,
                    backtracks: 0,
                    offer_interrupt: game.saved.offer_interrupt,
                    search: game.saved.search.clone(),
                    reported_terminal: false,
                    waiting: false,
                };
                self.games.insert(
                    id,
                    Game {
                        saved,
                        tree: None,
                        sent_root: 0,
                        last_path: vec![],
                    },
                );
                self.search_states.insert(id, SearchStates::default());
                Ok(vec![1])
            }
            // 独立模拟随机序列与公开转置键；不允许重置正式对局。
            11 => {
                let id = input.u64()?;
                let seed = input.u32()?;
                if id >> 63 == 0 || !input.0.is_empty() {
                    return Err("invalid search snapshot".into());
                }
                let game = self.games.get_mut(&id).ok_or("unknown search actor")?;
                if game.saved.search.is_none() && !game.saved.state.extra.contains_key("winner") {
                    game.start_root(cat)?;
                }
                let actor = game
                    .saved
                    .search
                    .as_ref()
                    .map_or_else(|| runtime::viewer(&game.saved.state), |s| s.actor);
                let public = runtime::public_view(&game.saved.state, actor)?.into_position();
                let key =
                    serde_json::to_vec(&(&public, game.saved.offer_interrupt, &game.saved.search))
                        .map_err(|e| e.to_string())?;
                game.saved.state = public;
                game.saved
                    .state
                    .extra
                    .insert("rng".into(), serde_json::json!(seed.max(1)));
                game.saved.random = seed ^ 0xa0761d65;
                let saved = game.saved.branch();
                let states = self
                    .search_states
                    .get_mut(&id)
                    .ok_or("unknown search scope")?;
                let index = if let Some(index) = states.free.pop() {
                    states.values[index] = Some(saved);
                    index
                } else {
                    states.values.push(Some(saved));
                    states.values.len() - 1
                };
                let mut out = (index as u64).to_le_bytes().to_vec();
                out.extend((key.len() as u64).to_le_bytes());
                out.extend(key);
                Ok(out)
            }
            12 => {
                let id = input.u64()?;
                if id >> 63 == 0 || !input.0.is_empty() {
                    return Err("invalid search path request".into());
                }
                let game = self.games.get(&id).ok_or("unknown search actor")?;
                let mut out = (game.last_path.len() as u32).to_le_bytes().to_vec();
                for &index in &game.last_path {
                    out.extend((index as u32).to_le_bytes());
                }
                Ok(out)
            }
            // 搜索后继只通过内核内的写时分离句柄恢复，不往返 JSON 状态。
            14 | 15 => {
                let id = input.u64()?;
                let index = input.u64()? as usize;
                if id >> 63 == 0 {
                    return Err("invalid search state handle".into());
                }
                let states = self
                    .search_states
                    .get_mut(&id)
                    .ok_or("unknown search scope")?;
                let slot = states.values.get_mut(index).ok_or("unknown search state")?;
                if op == 15 {
                    if !input.0.is_empty() || slot.take().is_none() {
                        return Err("released search state".into());
                    }
                    states.free.push(index);
                } else {
                    let seed = input.u32()?;
                    if !input.0.is_empty() {
                        return Err("trailing search state request".into());
                    }
                    let mut saved = slot.as_ref().ok_or("released search state")?.branch();
                    saved
                        .state
                        .extra
                        .insert("rng".into(), serde_json::json!(seed.max(1)));
                    saved.random = seed ^ 0xa0761d65;
                    self.games.insert(
                        id,
                        Game {
                            saved,
                            tree: None,
                            sent_root: 0,
                            last_path: vec![],
                        },
                    );
                }
                Ok(vec![])
            }
            _ => Err("unknown actor operation".into()),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hj_actor_create() -> *mut c_void {
    match catalog::embedded() {
        Ok(cat) => Box::into_raw(Box::new(Actor {
            games: BTreeMap::new(),
            search_states: BTreeMap::new(),
            cat: Box::new(cat),
            owner: std::thread::current().id(),
            output: vec![],
        }))
        .cast(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// op 的输入在调用期间借用；输出由句柄拥有，错误以 UTF-8 返回，panic 不越过 FFI。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hj_actor_call(
    handle: *mut c_void,
    op: u32,
    bytes: *const u8,
    size: usize,
    data: *mut *const u8,
    len: *mut usize,
) -> i32 {
    let Some(actor) = (unsafe { handle.cast::<Actor>().as_mut() }) else {
        return -1;
    };
    if data.is_null() || len.is_null() || (size > 0 && bytes.is_null()) {
        return -1;
    }
    let bytes = if size == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(bytes, size) }
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| actor.call(op, bytes)))
        .unwrap_or_else(|_| Err("actor panic; lane must stop".into()));
    let status = if result.is_ok() { 0 } else { -1 };
    actor.output = result.unwrap_or_else(String::into_bytes);
    unsafe {
        *data = actor.output.as_ptr();
        *len = actor.output.len();
    }
    status
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hj_actor_close(handle: *mut c_void) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle.cast::<Actor>()) });
    }
}

#[cfg(test)]
mod checks {
    use super::*;
    #[test]
    fn search_forks_only_public_state_and_never_resets_the_real_random_sequence() {
        for shrine in [0u32, 1] {
            let mut actor = Actor {
                games: BTreeMap::new(),
                search_states: BTreeMap::new(),
                cat: Box::new(catalog::embedded().unwrap()),
                owner: std::thread::current().id(),
                output: vec![],
            };
            let id = 1u64.to_le_bytes();
            let mut create = id.to_vec();
            for value in [73127u32, 1, shrine] {
                create.extend(value.to_le_bytes());
            }
            actor.call(0, &create).unwrap();
            actor.call(7, &id).unwrap();
            let before = actor.call(2, &id).unwrap();
            let scratch = (1u64 << 63) | 1;
            let mut fork = id.to_vec();
            fork.extend(scratch.to_le_bytes());
            assert_eq!(actor.call(9, &fork).unwrap(), vec![u8::from(shrine == 0)]);
            if shrine == 0 {
                let public = &actor.games[&scratch].saved.state;
                assert!(public.extra.get("seed").is_none() && public.extra.get("rng").is_none());
                let mut reset = scratch.to_le_bytes().to_vec();
                reset.extend(3177u32.to_le_bytes());
                let position = actor.call(11, &reset).unwrap();
                let handle = &position[..8];
                let mut query = scratch.to_le_bytes().to_vec();
                query.extend(0u32.to_le_bytes());
                let first = actor.call(10, &query).unwrap();
                let mut restore = scratch.to_le_bytes().to_vec();
                restore.extend(handle);
                restore.extend(1729u32.to_le_bytes());
                actor.call(14, &restore).unwrap();
                assert_eq!(first, actor.call(10, &query).unwrap());
                let mut release = scratch.to_le_bytes().to_vec();
                release.extend(handle);
                actor.call(15, &release).unwrap();
                assert!(actor.call(14, &restore).is_err());
            }
            let mut invalid_reset = id.to_vec();
            invalid_reset.extend(3177u32.to_le_bytes());
            assert!(actor.call(11, &invalid_reset).is_err());
            assert_eq!(before, actor.call(2, &id).unwrap());
        }
    }
    #[test]
    fn evaluation_query_is_public_and_rejected_teacher_command_is_atomic() {
        let mut actor = Actor {
            games: BTreeMap::new(),
            search_states: BTreeMap::new(),
            cat: Box::new(catalog::embedded().unwrap()),
            owner: std::thread::current().id(),
            output: vec![],
        };
        let id = 1u64.to_le_bytes();
        let mut create = id.to_vec();
        for value in [73127u32, 1, 1] {
            create.extend(value.to_le_bytes());
        }
        actor.call(0, &create).unwrap();
        let query: serde_json::Value =
            serde_json::from_slice(&actor.call(7, &id).unwrap()).unwrap();
        let observation = &query["observation"];
        for private in ["seed", "rng", "events", "log"] {
            assert!(observation.get(private).is_none());
        }
        let before = actor.call(2, &id).unwrap();
        let mut pass = id.to_vec();
        pass.extend(b"null");
        assert!(actor.call(8, &pass).is_err());
        assert_eq!(before, actor.call(2, &id).unwrap());
        let mut invalid = id.to_vec();
        invalid.extend(br#"{"type":"end"}"#);
        assert!(actor.call(8, &invalid).is_err());
        assert_eq!(before, actor.call(2, &id).unwrap());
        let mut advance = id.to_vec();
        advance.extend(0u32.to_le_bytes());
        let frame = actor.call(6, &advance).unwrap();
        assert_eq!(u32::from_le_bytes(frame[64..68].try_into().unwrap()), 0);
        assert_eq!(actor.games[&1].saved.commands, 0);
    }
    #[test]
    fn typed_identity_and_presence_do_not_round() {
        let mut row = encoding::TypedRow::default();
        row.ids[2] = 4097;
        row.present = 4097;
        let mut bytes = vec![];
        row.write(&mut bytes);
        assert_eq!(i64::from_le_bytes(bytes[16..24].try_into().unwrap()), 4097);
        assert_eq!(
            u64::from_le_bytes(bytes[480..488].try_into().unwrap()),
            4097
        );
    }
}
