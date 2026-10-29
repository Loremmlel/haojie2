//! 环境仅由所属线程使用；开发 JSON 入口与原生搜索共用类型化方法。
use super::encoding;
use crate::{
    model::{Catalog, Command, State},
    runtime,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

#[derive(Clone)]
pub struct Info {
    pub id: u64,
    pub actor: usize,
    pub ply: f64,
    pub winner: Option<Value>,
    pub key: String,
    pub metrics: [f64; 4],
}
impl Info {
    pub fn value(&self) -> Value {
        json!({"id":self.id,"actor":self.actor,"ply":self.ply,"winner":self.winner,
               "key":self.key,"metrics":self.metrics})
    }
}

#[derive(Clone)]
pub struct Bundle {
    pub states: Arc<[f32]>,
    pub records: Arc<[f32]>,
    pub owners: Arc<[f32]>,
    pub relations: Arc<[f32]>,
}
pub struct Front {
    pub command: Option<Command>,
    pub next: Vec<usize>,
    pub options: Arc<[f32]>,
    pub stage: usize,
    pub focus: &'static str,
}

struct Position {
    state: State,
    encoding: Option<encoding::Compact>,
    bundle: Option<Arc<Bundle>>,
    decoder: Option<super::decoder::Decoder>,
}
impl Position {
    fn new(state: State) -> Self {
        Self {
            state,
            encoding: None,
            bundle: None,
            decoder: None,
        }
    }
}

pub struct Host {
    catalog: Catalog,
    positions: BTreeMap<u64, Position>,
    next: u64,
    tensor: Vec<f32>,
    response: Vec<u8>,
    teacher: Value,
}
enum TransitionError {
    Rejected(String),
    Rule(String),
}
impl TransitionError {
    fn text(self) -> String {
        match self {
            Self::Rejected(message) => format!("合法命令被拒绝: {message}"),
            Self::Rule(message) => message,
        }
    }
}

impl Host {
    pub fn new(seed: u32) -> Result<Self, String> {
        let catalog = crate::catalog::embedded()?;
        let state = runtime::create(seed, false, &catalog)?;
        Ok(Self {
            catalog,
            positions: BTreeMap::from([(0, Position::new(state))]),
            next: 1,
            tensor: vec![],
            response: vec![],
            teacher: Value::Null,
        })
    }

    pub fn info_typed(&self, id: u64) -> Result<Info, String> {
        let s = &self.positions.get(&id).ok_or("unknown position")?.state;
        let public = runtime::public_view(s, runtime::viewer(s))?;
        let winner = s.extra.get("winner").filter(|v| !v.is_null()).cloned();
        let mut hp = [0., 0.];
        let mut count = [0., 0.];
        for u in &s.units {
            if u.owner == 1 || u.owner == 2 {
                hp[u.owner - 1] += u.hp;
                count[u.owner - 1] += 1.;
            }
        }
        let h = s.extra.get("heads").unwrap_or(&Value::Null);
        Ok(Info {
            id,
            actor: runtime::viewer(s),
            ply: s.ply,
            winner,
            key: crate::canonical::hash(public.position()),
            metrics: [
                (s.bases["1"].as_f64().unwrap_or(0.) - s.bases["2"].as_f64().unwrap_or(0.)) / 300.,
                (count[0] - count[1]) / 32.,
                (hp[0] - hp[1]) / 512.,
                (h["1"].as_f64().unwrap_or(0.) - h["2"].as_f64().unwrap_or(0.)) / 20.,
            ],
        })
    }

    pub fn encode_typed(&mut self, id: u64) -> Result<(Info, Arc<Bundle>), String> {
        let p = self.positions.get_mut(&id).ok_or("unknown position")?;
        if p.encoding.is_none() {
            p.encoding = Some(encoding::Compact::new(&p.state, &self.catalog)?);
        }
        if p.bundle.is_none() {
            let e = p.encoding.as_ref().unwrap();
            p.bundle = Some(Arc::new(Bundle {
                states: e.states.clone().into(),
                records: e.records.clone().into(),
                owners: e.owners.clone().into(),
                relations: e.relations.clone().into(),
            }));
        }
        let bundle = p.bundle.as_ref().unwrap().clone();
        Ok((self.info_typed(id)?, bundle))
    }

    pub fn decode_typed(&mut self, id: u64, prefix: usize) -> Result<Front, String> {
        let p = self.positions.get_mut(&id).ok_or("unknown position")?;
        let view = runtime::public_view(&p.state, runtime::viewer(&p.state))?;
        if p.decoder.is_none() {
            p.decoder = Some(super::decoder::Decoder::new(
                view.position(),
                &self.catalog,
            )?);
        }
        let front = p
            .decoder
            .as_mut()
            .unwrap()
            .front(view.position(), &self.catalog, prefix)?;
        if let Some(command) = front.command {
            return Ok(Front {
                command: Some(command),
                next: vec![],
                options: Arc::from([]),
                stage: front.stage,
                focus: front.focus,
            });
        }
        if p.encoding.is_none() {
            p.encoding = Some(encoding::Compact::new(&p.state, &self.catalog)?);
        }
        let options = p.encoding.as_ref().unwrap().options(
            &self.catalog,
            &front.commands,
            front.stage,
            front.focus,
        )?;
        Ok(Front {
            command: None,
            next: front.children,
            options: options.into(),
            stage: front.stage,
            focus: front.focus,
        })
    }

    pub fn trace_typed(
        &mut self,
        id: u64,
        command: &Command,
    ) -> Result<Vec<(usize, usize)>, String> {
        let p = self.positions.get_mut(&id).ok_or("unknown position")?;
        let view = runtime::public_view(&p.state, runtime::viewer(&p.state))?;
        if p.decoder.is_none() {
            p.decoder = Some(super::decoder::Decoder::new(
                view.position(),
                &self.catalog,
            )?);
        }
        p.decoder
            .as_mut()
            .unwrap()
            .trace(view.position(), &self.catalog, command)
    }

    pub fn snapshot(&self) -> Value {
        json!({"format":"haojie-classic-environment-v1","ruleset":crate::model::RULESET,
               "state":self.positions[&0].state})
    }

    pub fn restore(&mut self, snapshot: &Value) -> Result<Info, String> {
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
        self.positions.insert(0, Position::new(state));
        self.next = 1;
        self.info_typed(0)
    }

    pub fn teacher_next(&mut self, difficulty: &str) -> Result<Command, String> {
        let observation =
            crate::ai::keys::Keys::new(&Value::Null).observation(&self.positions[&0].state);
        let difficulty = crate::ai::settings::Difficulty::parse(difficulty)?;
        let result = crate::ai::search::teacher_next(
            &observation,
            difficulty,
            &mut self.teacher,
            &self.catalog,
        )?;
        serde_json::from_value(result["command"].clone()).map_err(|e| e.to_string())
    }
    pub fn teacher_snapshot(&self) -> Value {
        self.teacher.clone()
    }
    pub fn teacher_restore(&mut self, value: &Value) {
        self.teacher = value.clone();
    }

    fn transition(
        &mut self,
        id: u64,
        command: &Command,
        seed: Option<u32>,
    ) -> Result<Info, TransitionError> {
        let position = self
            .positions
            .get(&id)
            .ok_or_else(|| TransitionError::Rule("unknown position".into()))?;
        if seed.is_none() && id != 0 {
            return Err(TransitionError::Rule("only root can commit".into()));
        }
        let source = if let Some(seed) = seed {
            let mut public =
                runtime::public_view(&position.state, runtime::viewer(&position.state))
                    .map_err(TransitionError::Rule)?
                    .into_position();
            public.extra.insert("rng".into(), json!(seed));
            public.extra.insert("seed".into(), json!(seed));
            public
        } else {
            position.state.fork()
        };
        let result = crate::resolution::without_presentation(|| {
            crate::command::apply_runtime(&source, command, &self.catalog)
        });
        let mut next = match result {
            Ok(next) => next,
            Err(crate::model::Failure::Invalid(message)) => {
                return Err(TransitionError::Rejected(message.into()));
            }
            Err(crate::model::Failure::InvalidOwned(message)) => {
                return Err(TransitionError::Rejected(message));
            }
            Err(error) => return Err(TransitionError::Rule(format!("rule failure: {error:?}"))),
        };
        next.events.clear();
        next.extra.insert("log".into(), json!([]));
        let next_id = if seed.is_some() {
            let value = self.next;
            self.next += 1;
            value
        } else {
            self.positions.clear();
            self.next = 1;
            0
        };
        self.positions.insert(next_id, Position::new(next));
        self.info_typed(next_id).map_err(TransitionError::Rule)
    }
    pub fn sample(&mut self, id: u64, command: &Command, seed: u32) -> Result<Info, String> {
        self.transition(id, command, Some(seed))
            .map_err(TransitionError::text)
    }
    pub fn apply(&mut self, command: &Command) -> Result<Info, String> {
        self.transition(0, command, None)
            .map_err(TransitionError::text)
    }

    fn handle(&mut self, request: Value) -> Result<Value, String> {
        let id = request["id"].as_u64().unwrap_or(0);
        match request["op"].as_str().unwrap_or("") {
            "encode" => {
                let (info, bundle) = self.encode_typed(id)?;
                self.tensor.clear();
                for array in [
                    &bundle.states,
                    &bundle.records,
                    &bundle.owners,
                    &bundle.relations,
                ] {
                    self.tensor.extend_from_slice(array);
                }
                let mut info = info.value();
                info["entities"] = json!(bundle.states.len() / encoding::STATE_WIDTH);
                info["records"] = json!(bundle.owners.len());
                info["floats"] = json!(self.tensor.len());
                Ok(info)
            }
            "decode" => {
                let front =
                    self.decode_typed(id, request["prefix"].as_u64().unwrap_or(0) as usize)?;
                if let Some(command) = front.command {
                    Ok(json!({"leaf":true,"command":command}))
                } else {
                    self.tensor = front.options.to_vec();
                    Ok(
                        json!({"leaf":false,"next":front.next,"count":front.next.len(),"floats":self.tensor.len(),"stage":front.stage,"focus":front.focus}),
                    )
                }
            }
            "trace" => {
                let command = serde_json::from_value(request["command"].clone())
                    .map_err(|e| e.to_string())?;
                Ok(json!(self.trace_typed(id, &command)?))
            }
            "teacher-next" => {
                let observation = request.get("observation").cloned().unwrap_or_else(|| {
                    crate::ai::keys::Keys::new(&Value::Null).observation(&self.positions[&0].state)
                });
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
            "teacher-snapshot" => Ok(self.teacher_snapshot()),
            "teacher-restore" => {
                self.teacher_restore(&request["teacher"]);
                Ok(json!(true))
            }
            "info" => Ok(self.info_typed(id)?.value()),
            "observe" => {
                let s = &self.positions.get(&id).ok_or("unknown position")?.state;
                runtime::observe(s, runtime::viewer(s))
            }
            "snapshot" => Ok(self.snapshot()),
            "restore" => Ok(self.restore(&request["snapshot"])?.value()),
            "apply" | "sample-command" => {
                let command: Command = serde_json::from_value(request["command"].clone())
                    .map_err(|e| e.to_string())?;
                let seed = if request["op"] == "sample-command" {
                    Some(request["seed"].as_u64().ok_or("simulation seed required")? as u32)
                } else {
                    None
                };
                match self.transition(id, &command, seed) {
                    Ok(info) => {
                        let mut value = info.value();
                        value["command"] = json!(command);
                        Ok(value)
                    }
                    Err(TransitionError::Rejected(message)) => {
                        Ok(json!({"rejected":true,"message":message}))
                    }
                    Err(TransitionError::Rule(message)) => Err(message),
                }
            }
            _ => Err("unknown classic operation".into()),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hc_new(seed: u32) -> *mut Host {
    match catch_unwind(|| Host::new(seed).map(Box::new)) {
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
