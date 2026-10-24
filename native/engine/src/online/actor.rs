//! C++ lane 拥有的持久 actor；一次调用只推进指定对局，不存在 worker 整批屏障。
use super::*;
use std::ffi::c_void;

struct Actor {
    // 必须先销毁树，再释放它们借用的、地址稳定的目录。
    games: BTreeMap<u64, Game<'static>>,
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
                    },
                );
                Ok(vec![])
            }
            1 => {
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
                let frame = game.advance((count != 0).then_some(probabilities), cat)?;
                let mut output = frame[..84].to_vec();
                if game.saved.waiting {
                    let search = game.saved.search.as_ref().unwrap();
                    let tree = game.tree.as_mut().unwrap();
                    let node = tree.node(&search.cursor)?;
                    let mut encoded = (*encoding::encode_sampling(tree, &node)?).clone();
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
                    for row in encoded.entities.typed_iter().skip(if has_root {
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
                        for (i, &value) in encoded.globals.iter().enumerate() {
                            output.extend(
                                encoding::bf16_bits(if i == 31 { 0.0 } else { value })
                                    .to_le_bytes(),
                            );
                        }
                    }
                    game.sent_root = game.saved.root_id;
                    for row in &encoded.candidates {
                        // 参数域全为离散选择；从尚未舍入的规范行恢复精确整数。
                        let mut scales = [1.0; 64];
                        for (i, scale) in [
                            (36, 9.),
                            (37, 13.),
                            (38, 13.),
                            (39, 9.),
                            (42, 4.),
                            (43, 128.),
                            (44, 128.),
                            (45, 128.),
                            (46, 8.),
                            (48, 16.),
                            (51, 16.),
                            (52, 16.),
                            (53, 3.),
                            (54, 2.),
                            (55, 117.),
                            (56, 256.),
                            (57, 256.),
                            (60, 9.),
                            (61, 13.),
                        ] {
                            scales[i] = scale;
                        }
                        for (i, value) in row.iter().enumerate() {
                            output.extend(((value * scales[i]).round() as i64).to_le_bytes());
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
                    },
                );
                Ok(vec![])
            }
            5 => {
                let id = input.u64()?;
                if !input.0.is_empty() || self.games.remove(&id).is_none() {
                    return Err("unknown actor removal".into());
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
