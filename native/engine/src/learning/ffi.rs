//! 句柄仅由创建它的 worker 使用；JSON 只传控制/快照，模型张量通过调用者缓冲复制。
use super::encoding;
use crate::{
    model::{Catalog, Command, State},
    runtime,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
};

struct Position {
    state: State,
    encoding: Option<encoding::Compact>,
    decoder: Option<super::decoder::Decoder>,
}
pub struct Host {
    catalog: Catalog,
    positions: BTreeMap<u64, Position>,
    next: u64,
    tensor: Vec<f32>,
    response: Vec<u8>,
    teacher: Value,
}
impl Host {
    fn info(&self, id: u64) -> Result<Value, String> {
        let s = &self.positions.get(&id).ok_or("unknown position")?.state;
        let public = runtime::public_view(s, runtime::viewer(s))?;
        let winner = s.extra.get("winner").cloned().unwrap_or(Value::Null);
        let mut hp = [0., 0.];
        let mut count = [0., 0.];
        for u in &s.units {
            if u.owner == 1 || u.owner == 2 {
                hp[u.owner - 1] += u.hp;
                count[u.owner - 1] += 1.;
            }
        }
        let h = s.extra.get("heads").unwrap_or(&Value::Null);
        let metrics = [
            (s.bases["1"].as_f64().unwrap_or(0.) - s.bases["2"].as_f64().unwrap_or(0.)) / 300.,
            (count[0] - count[1]) / 32.,
            (hp[0] - hp[1]) / 512.,
            (h["1"].as_f64().unwrap_or(0.) - h["2"].as_f64().unwrap_or(0.)) / 20.,
        ];
        Ok(
            json!({"id":id,"actor":runtime::viewer(s),"ply":s.ply,"winner":winner,
                  "key":crate::canonical::hash(public.position()),"metrics":metrics}),
        )
    }
    fn handle(&mut self, request: Value) -> Result<Value, String> {
        let id = request["id"].as_u64().unwrap_or(0);
        match request["op"].as_str().unwrap_or("") {
            "encode" => {
                let p = self.positions.get_mut(&id).ok_or("unknown position")?;
                if p.encoding.is_none() {
                    p.encoding = Some(encoding::Compact::new(&p.state, &self.catalog)?);
                }
                let e = p.encoding.as_ref().unwrap();
                self.tensor.clear();
                self.tensor.extend_from_slice(&e.states);
                self.tensor.extend_from_slice(&e.records);
                self.tensor.extend_from_slice(&e.owners);
                self.tensor.extend_from_slice(&e.relations);
                let n = e.states.len() / encoding::STATE_WIDTH;
                let m = e.owners.len();
                let mut info = self.info(id)?;
                info["entities"] = json!(n);
                info["records"] = json!(m);
                info["floats"] = json!(self.tensor.len());
                Ok(info)
            }
            "decode" | "trace" => {
                let p = self.positions.get_mut(&id).ok_or("unknown position")?;
                let view = runtime::public_view(&p.state, runtime::viewer(&p.state))?;
                if p.decoder.is_none() {
                    p.decoder = Some(super::decoder::Decoder::new(
                        view.position(),
                        &self.catalog,
                    )?);
                }
                let decoder = p.decoder.as_mut().unwrap();
                if request["op"] == "trace" {
                    let command: Command = serde_json::from_value(request["command"].clone())
                        .map_err(|e| e.to_string())?;
                    return Ok(json!(decoder.trace(
                        view.position(),
                        &self.catalog,
                        &command
                    )?));
                }
                let mut front = decoder.choices(
                    view.position(),
                    &self.catalog,
                    request["prefix"].as_u64().unwrap_or(0) as usize,
                )?;
                if front["leaf"] != true {
                    if p.encoding.is_none() {
                        p.encoding = Some(encoding::Compact::new(&p.state, &self.catalog)?);
                    }
                    let commands: Vec<Command> = serde_json::from_value(front["choices"].take())
                        .map_err(|e| e.to_string())?;
                    self.tensor = p.encoding.as_ref().unwrap().options(
                        &self.catalog,
                        &commands,
                        front["stage"].as_u64().unwrap() as usize,
                        front["focus"].as_str().unwrap(),
                    )?;
                    front.as_object_mut().unwrap().remove("choices");
                    front["count"] = json!(commands.len());
                    front["floats"] = json!(self.tensor.len());
                }
                Ok(front)
            }
            "teacher-next" => {
                let observation = if let Some(o) = request.get("observation") {
                    o.clone()
                } else {
                    crate::ai::keys::Keys::new(&Value::Null).observation(&self.positions[&0].state)
                };
                let difficulty = crate::ai::settings::Difficulty::parse(
                    request["difficulty"].as_str().unwrap_or("easy"),
                )?;
                crate::ai::search::teacher_next(
                    &observation,
                    difficulty,
                    &mut self.teacher,
                    &self.catalog,
                )
            }
            "teacher-snapshot" => Ok(self.teacher.clone()),
            "teacher-restore" => {
                self.teacher = request["teacher"].clone();
                Ok(json!(true))
            }
            "info" => self.info(id),
            "observe" => {
                let s = &self.positions.get(&id).ok_or("unknown position")?.state;
                runtime::observe(s, runtime::viewer(s))
            }
            "snapshot" => Ok(
                json!({"format":"haojie-classic-environment-v1","ruleset":crate::model::RULESET,
                                 "state":self.positions[&0].state}),
            ),
            "restore" => {
                let snapshot = &request["snapshot"];
                if snapshot["format"] != "haojie-classic-environment-v1"
                    || snapshot["ruleset"] != crate::model::RULESET
                {
                    return Err("snapshot version mismatch".into());
                }
                let state: State =
                    serde_json::from_value(snapshot["state"].clone()).map_err(|e| e.to_string())?;
                if state.extra.contains_key("mode") {
                    return Err("only classic is supported".into());
                }
                self.positions.clear();
                self.positions.insert(
                    0,
                    Position {
                        state,
                        encoding: None,
                        decoder: None,
                    },
                );
                self.next = 1;
                self.info(0)
            }
            "apply" | "sample-command" => {
                let position = self.positions.get(&id).ok_or("unknown position")?;
                let command: Command = serde_json::from_value(request["command"].clone())
                    .map_err(|e| e.to_string())?;
                let simulation = request["op"] == "sample-command";
                if !simulation && id != 0 {
                    return Err("only root can commit".into());
                }
                let source = if simulation {
                    let mut public =
                        runtime::public_view(&position.state, runtime::viewer(&position.state))?
                            .into_position();
                    let seed = request["seed"].as_u64().ok_or("simulation seed required")? as u32;
                    public.extra.insert("rng".into(), json!(seed));
                    public.extra.insert("seed".into(), json!(seed));
                    public
                } else {
                    position.state.fork()
                };
                let result = crate::resolution::without_presentation(|| {
                    crate::command::apply_runtime(&source, &command, &self.catalog)
                });
                let mut next = match result {
                    Ok(next) => next,
                    Err(crate::model::Failure::Invalid(message)) => {
                        return Ok(json!({"rejected":true,"message":message}));
                    }
                    Err(crate::model::Failure::InvalidOwned(message)) => {
                        return Ok(json!({"rejected":true,"message":message}));
                    }
                    Err(error) => return Err(format!("rule failure: {error:?}")),
                };
                next.events.clear();
                next.extra.insert("log".into(), json!([]));
                let next_id = if simulation {
                    let value = self.next;
                    self.next += 1;
                    value
                } else {
                    self.positions.clear();
                    self.next = 1;
                    0
                };
                self.positions.insert(
                    next_id,
                    Position {
                        state: next,
                        encoding: None,
                        decoder: None,
                    },
                );
                let mut info = self.info(next_id)?;
                info["command"] = json!(command);
                Ok(info)
            }
            _ => Err("unknown classic operation".into()),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hc_new(seed: u32) -> *mut Host {
    match catch_unwind(|| {
        let catalog = crate::catalog::embedded()?;
        let state = runtime::create(seed, false, &catalog)?;
        Ok::<_, String>(Box::new(Host {
            catalog,
            positions: BTreeMap::from([(
                0,
                Position {
                    state,
                    encoding: None,
                    decoder: None,
                },
            )]),
            next: 1,
            tensor: vec![],
            response: vec![],
            teacher: Value::Null,
        }))
    }) {
        Ok(Ok(host)) => Box::into_raw(host),
        _ => std::ptr::null_mut(),
    }
}
/// # Safety
/// host 必须来自 hc_new，且本调用期间独占；输入字节必须有效。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_call(host: *mut Host, data: *const u8, length: usize) -> *const u8 {
    let host = unsafe { &mut *host };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let request = serde_json::from_slice(unsafe { std::slice::from_raw_parts(data, length) })
            .map_err(|e| e.to_string())?;
        host.handle(request)
    }));
    let response = match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => json!({"error":error}),
        Err(_) => json!({"error":"native panic"}),
    };
    host.response = serde_json::to_vec(&response).unwrap();
    host.response.as_ptr()
}
/// # Safety
/// host 为仍存活的独占句柄。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_length(host: *const Host) -> usize {
    unsafe { (*host).response.len() }
}
/// # Safety
/// out 至少可写 capacity 个 f32，调用期间不能访问同一句柄。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_tensor(host: *const Host, out: *mut f32, capacity: usize) -> usize {
    let tensor = unsafe { &(*host).tensor };
    if capacity >= tensor.len() {
        unsafe { std::ptr::copy_nonoverlapping(tensor.as_ptr(), out, tensor.len()) };
    }
    tensor.len()
}
/// # Safety
/// 句柄只能在所属进程释放一次，不能在其他调用期间释放。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_close(host: *mut Host) {
    if !host.is_null() {
        drop(unsafe { Box::from_raw(host) });
    }
}
