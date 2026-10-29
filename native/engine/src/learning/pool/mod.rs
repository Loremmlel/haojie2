//! 单进程内固定归属线程池；工作线程之间只移动请求、结果与控制值。
mod ffi;
mod lane;

use crate::learning::{
    ffi::Bundle,
    random::Functions,
    search::{Reply, Request},
    timing::{self, Phase},
};
use lane::{Advanced, Lane};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Clone)]
pub struct Array {
    values: Arc<[f32]>,
    shape: Vec<usize>,
}
impl Array {
    fn bundle(bundle: &Bundle) -> Vec<Self> {
        let n = bundle.states.len() / 128;
        let m = bundle.owners.len();
        vec![
            Self {
                values: bundle.states.clone(),
                shape: vec![n, 128],
            },
            Self {
                values: bundle.records.clone(),
                shape: vec![m, 128],
            },
            Self {
                values: bundle.owners.clone(),
                shape: vec![m],
            },
            Self {
                values: bundle.relations.clone(),
                shape: vec![n, n, 8],
            },
        ]
    }
}

struct Config {
    workers: usize,
    games_per_worker: usize,
    simulations: usize,
    fast_simulations: usize,
    depth: usize,
    full_fraction: f64,
    temperature: f64,
    max_commands: u64,
    audit: bool,
    profile: bool,
}
impl Config {
    fn read(v: &Value) -> Result<Self, String> {
        let result = Self {
            workers: v["workers"].as_u64().unwrap_or(4) as usize,
            games_per_worker: v["games_per_worker"].as_u64().unwrap_or(16) as usize,
            simulations: v["simulations"].as_u64().unwrap_or(64) as usize,
            fast_simulations: v["fast_simulations"].as_u64().unwrap_or(16) as usize,
            depth: v["depth"].as_u64().unwrap_or(16) as usize,
            full_fraction: v["full_fraction"].as_f64().unwrap_or(0.25),
            temperature: v["temperature"].as_f64().unwrap_or(1.),
            max_commands: v["max_commands"].as_u64().unwrap_or(20000),
            audit: v["audit"].as_bool().unwrap_or(false),
            profile: std::env::var("HAOJIE_PROFILE").is_ok_and(|v| v == "1"),
        };
        if result.workers == 0
            || result.games_per_worker == 0
            || result.depth == 0
            || result.max_commands == 0
            || !(0. ..=1.).contains(&result.full_fraction)
            || result.temperature < 0.
        {
            return Err("原生池配置无效".into());
        }
        Ok(result)
    }
}

struct Record {
    metadata: Value,
    arrays: Vec<Array>,
}
struct RequestEnvelope {
    id: u64,
    lane: usize,
    game_id: u64,
    step: u64,
    model: i64,
    request: Request,
    at: Instant,
    replied: bool,
}
impl RequestEnvelope {
    fn arrays(&self) -> Vec<Array> {
        if let Some(bundle) = &self.request.bundle {
            Array::bundle(bundle)
        } else {
            vec![Array {
                values: self.request.options.clone(),
                shape: vec![self.request.options.len() / 704, 704],
            }]
        }
    }
}

#[derive(Default)]
struct Queues {
    requests: VecDeque<RequestEnvelope>,
    records: VecDeque<Record>,
    error: Option<String>,
    wake_requests: bool,
    wake_records: bool,
}
struct Shared {
    queues: Mutex<Queues>,
    changed: Condvar,
    sequence: AtomicU64,
    requests_taken: AtomicU64,
    queue_nanoseconds: AtomicU64,
    tids: Vec<AtomicU64>,
}
impl Shared {
    fn fail(&self, error: String) {
        self.queues.lock().unwrap_or_else(|e| e.into_inner()).error = Some(error);
        self.changed.notify_all();
    }
    fn failed(&self) -> bool {
        self.queues
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .error
            .is_some()
    }
}

type Answer = mpsc::Sender<Result<Value, String>>;
enum Control {
    Assign(usize, Value, Answer),
    Reply(usize, u64, Reply),
    Ack(usize),
    Pause,
    Resume,
    Inspect(bool, Answer),
    Stop,
}
struct Slot {
    lane: Lane,
    waiting: Option<u64>,
    record_pending: bool,
}

fn thread_id() -> u64 {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentThreadId() -> u32;
        }
        unsafe { GetCurrentThreadId() as u64 }
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn gettid() -> i32;
        }
        unsafe { gettid() as u64 }
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        0
    }
}

fn worker(
    index: usize,
    receiver: mpsc::Receiver<Control>,
    shared: Arc<Shared>,
    config: Arc<Config>,
    functions: Functions,
) {
    timing::initialize(config.profile);
    shared.tids[index].store(thread_id(), Ordering::Release);
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let mut slots: Vec<Option<Slot>> = (0..config.games_per_worker).map(|_| None).collect();
        let mut ready = VecDeque::new();
        let mut paused = false;
        loop {
            let mut controls: Vec<Control> = receiver.try_iter().collect();
            if controls.is_empty() && (paused || ready.is_empty()) {
                let _waiting = timing::scope(if paused {
                    Phase::WorkerPaused
                } else {
                    Phase::WorkerWait
                });
                match receiver.recv() {
                    Ok(value) => controls.push(value),
                    Err(_) => break,
                }
            }
            for control in controls {
                let _control = timing::scope(Phase::WorkerControl);
                match control {
                    Control::Stop => return Ok(()),
                    Control::Pause => paused = true,
                    Control::Resume => paused = false,
                    Control::Assign(local, value, answer) => {
                        let assigned = if slots[local].is_some() {
                            Err("lane 尚未交还".into())
                        } else {
                            Lane::new(&value, config.clone(), functions).map(|lane| {
                                slots[local] = Some(Slot {
                                    lane,
                                    waiting: None,
                                    record_pending: false,
                                });
                                ready.push_back(local);
                                json!({"ok":true})
                            })
                        };
                        let _ = answer.send(assigned);
                    }
                    Control::Reply(local, id, value) => {
                        let slot = slots[local].as_mut().ok_or("回应对应的lane已结束")?;
                        if slot.waiting != Some(id) {
                            return Err("神经请求回应编号不匹配".into());
                        }
                        slot.lane.reply(value);
                        slot.waiting = None;
                        ready.push_back(local);
                    }
                    Control::Ack(local) => {
                        let slot = slots[local].as_mut().ok_or("确认的lane不存在")?;
                        if !slot.record_pending {
                            return Err("lane没有待确认的正式记录".into());
                        }
                        slot.record_pending = false;
                        if slot.lane.done {
                            slots[local] = None;
                        } else {
                            ready.push_back(local);
                        }
                    }
                    Control::Inspect(snapshot, answer) => {
                        let waiting = slots
                            .iter()
                            .flatten()
                            .filter(|s| s.waiting.is_some())
                            .count();
                        let records = slots.iter().flatten().filter(|s| s.record_pending).count();
                        let versions: Vec<i64> = slots
                            .iter()
                            .flatten()
                            .flat_map(|s| {
                                [s.lane.root_version, s.lane.game["version"].as_i64()]
                                    .into_iter()
                                    .flatten()
                            })
                            .filter(|v| *v >= 0)
                            .collect();
                        let value = if snapshot {
                            if !paused || waiting != 0 || records != 0 {
                                Err("池尚未达到保存边界".into())
                            } else {
                                Ok(Value::Array(
                                    slots
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(local, slot)| {
                                            slot.as_ref().map(|slot| {
                                                let mut value = slot.lane.snapshot();
                                                value["lane"] =
                                                    json!(index * config.games_per_worker + local);
                                                value
                                            })
                                        })
                                        .collect(),
                                ))
                            }
                        } else {
                            Ok(json!({"ready":paused && waiting==0 && records==0,
                                "active":slots.iter().flatten().count(),"pending_requests":waiting,
                                "pending_records":records,"versions":versions,
                                "profile":{"index":index,
                                    "thread_id":shared.tids[index].load(Ordering::Acquire),
                                    "phases":timing::snapshot()}}))
                        };
                        let _ = answer.send(value);
                    }
                }
            }
            if shared.failed() {
                paused = true;
            }
            if paused {
                continue;
            }
            let Some(local) = ready.pop_front() else {
                continue;
            };
            let _active = timing::scope(Phase::WorkerActive);
            let slot = slots[local].as_mut().ok_or("就绪lane不存在")?;
            let lane = index * config.games_per_worker + local;
            match slot.lane.advance()? {
                Advanced::Request(request) => {
                    let id = shared.sequence.fetch_add(1, Ordering::Relaxed);
                    slot.waiting = Some(id);
                    let message = RequestEnvelope {
                        id,
                        lane,
                        game_id: slot.lane.game["id"].as_u64().unwrap_or(0),
                        step: slot.lane.steps,
                        model: slot.lane.model(),
                        request,
                        at: Instant::now(),
                        replied: false,
                    };
                    shared.queues.lock().unwrap().requests.push_back(message);
                }
                Advanced::Record(mut record) => {
                    record.metadata["lane"] = json!(lane);
                    slot.record_pending = true;
                    shared.queues.lock().unwrap().records.push_back(record);
                }
            }
            shared.changed.notify_all();
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => (),
        Ok(Err(message)) => shared.fail(format!("原生工作线程 {index}: {message}")),
        Err(_) => shared.fail(format!("原生工作线程 {index} panic")),
    }
}

#[derive(Default)]
struct RecordBatch {
    metadata: Vec<u8>,
    arrays: Vec<Array>,
    count: usize,
}

pub struct Pool {
    config: Arc<Config>,
    shared: Arc<Shared>,
    senders: Vec<mpsc::Sender<Control>>,
    threads: Vec<JoinHandle<()>>,
    requests: Mutex<Vec<RequestEnvelope>>,
    records: Mutex<RecordBatch>,
    response: Mutex<Vec<u8>>,
}
impl Pool {
    fn new(value: Value) -> Result<Self, String> {
        let config = Arc::new(Config::read(&value)?);
        let functions = unsafe {
            Functions::from_addresses(
                value["gamma_address"].as_u64().unwrap_or(0) as usize,
                value["bounded_address"].as_u64().unwrap_or(0) as usize,
                value["exp_address"].as_u64().unwrap_or(0) as usize,
                value["exp_context"].as_u64().unwrap_or(0) as usize,
                value["exp_auxdata"].as_u64().unwrap_or(0) as usize,
            )?
        };
        let shared = Arc::new(Shared {
            queues: Mutex::new(Queues::default()),
            changed: Condvar::new(),
            sequence: AtomicU64::new(1),
            requests_taken: AtomicU64::new(0),
            queue_nanoseconds: AtomicU64::new(0),
            tids: (0..config.workers).map(|_| AtomicU64::new(0)).collect(),
        });
        let mut threads = vec![];
        let mut senders = vec![];
        for index in 0..config.workers {
            let (sender, receiver) = mpsc::channel();
            senders.push(sender);
            let state = shared.clone();
            let settings = config.clone();
            threads.push(
                thread::Builder::new()
                    .name(format!("haojie-native-{index}"))
                    .spawn(move || worker(index, receiver, state, settings, functions))
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(Self {
            config,
            shared,
            senders,
            threads,
            requests: Mutex::new(vec![]),
            records: Mutex::new(RecordBatch::default()),
            response: Mutex::new(vec![]),
        })
    }

    fn destination(&self, lane: usize) -> Result<(usize, usize), String> {
        let worker = lane / self.config.games_per_worker;
        if worker >= self.config.workers {
            return Err("lane超出池容量".into());
        }
        Ok((worker, lane % self.config.games_per_worker))
    }

    fn inspect(&self, snapshot: bool) -> Result<Vec<Value>, String> {
        let mut receivers = vec![];
        for sender in &self.senders {
            let (response, receiver) = mpsc::channel();
            sender
                .send(Control::Inspect(snapshot, response))
                .map_err(|_| "原生线程已经退出")?;
            receivers.push(receiver);
        }
        receivers
            .into_iter()
            .map(|r| {
                r.recv_timeout(Duration::from_secs(30))
                    .map_err(|_| "原生线程响应超时".to_owned())?
            })
            .collect()
    }

    fn control(&self, value: Value) -> Result<Value, String> {
        match value["op"].as_str().ok_or("缺少池操作")? {
            "assign" => {
                let (worker, local) =
                    self.destination(value["lane"].as_u64().ok_or("缺少lane")? as usize)?;
                let (answer, receiver) = mpsc::channel();
                self.senders[worker]
                    .send(Control::Assign(local, value, answer))
                    .map_err(|_| "原生线程已经退出")?;
                receiver
                    .recv_timeout(Duration::from_secs(30))
                    .map_err(|_| "分配对局超时".to_owned())?
            }
            "pause" | "resume" => {
                let pause = value["op"] == "pause";
                for sender in &self.senders {
                    sender
                        .send(if pause {
                            Control::Pause
                        } else {
                            Control::Resume
                        })
                        .map_err(|_| "原生线程已经退出")?;
                }
                self.wake();
                Ok(json!({"ok":true}))
            }
            "snapshot" => {
                let groups = self.inspect(true)?;
                Ok(
                    json!({"lanes":groups.into_iter().flat_map(|v| v.as_array().unwrap().clone()).collect::<Vec<_>>()}),
                )
            }
            "status" | "barrier" | "active_versions" => {
                let error = self
                    .shared
                    .queues
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .error
                    .clone();
                if let Some(error) = error {
                    return Ok(json!({"error":error,
                    "threads":self.shared.tids.iter().map(|id|id.load(Ordering::Acquire)).collect::<Vec<_>>()}));
                }
                let groups = self.inspect(false)?;
                Ok(
                    json!({"error":error,"ready":groups.iter().all(|g|g["ready"]==true),
                    "active":groups.iter().map(|g|g["active"].as_u64().unwrap()).sum::<u64>(),
                    "pending_requests":groups.iter().map(|g|g["pending_requests"].as_u64().unwrap()).sum::<u64>(),
                    "pending_records":groups.iter().map(|g|g["pending_records"].as_u64().unwrap()).sum::<u64>(),
                    "versions":groups.iter().flat_map(|g|g["versions"].as_array().unwrap().clone()).collect::<Vec<_>>(),
                    "requests_taken":self.shared.requests_taken.load(Ordering::Relaxed),
                    "queue_seconds":self.shared.queue_nanoseconds.load(Ordering::Relaxed) as f64 / 1e9,
                    "profile":{"enabled":self.config.profile,
                        "workers":if self.config.profile {
                            groups.iter().map(|g|g["profile"].clone()).collect::<Vec<_>>()
                        } else { vec![] }},
                    "threads":self.shared.tids.iter().map(|id|id.load(Ordering::Acquire)).collect::<Vec<_>>()}),
                )
            }
            _ => Err("未知池操作".into()),
        }
    }

    fn wake(&self) {
        let mut queues = self.shared.queues.lock().unwrap_or_else(|e| e.into_inner());
        queues.wake_requests = true;
        queues.wake_records = true;
        self.shared.changed.notify_all();
    }

    fn take_requests(&self, maximum: usize, wait_ms: u64) -> Result<usize, String> {
        let mut held = self.requests.lock().unwrap();
        if !held.is_empty() {
            return Err("前一请求批次尚未释放".into());
        }
        let deadline = Instant::now() + Duration::from_millis(wait_ms);
        let mut queues = self.shared.queues.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(error) = &queues.error {
                return Err(error.clone());
            }
            let now = Instant::now();
            let batch_deadline = queues.requests.front().map_or(deadline, |r| {
                (r.at + Duration::from_millis(2)).min(deadline)
            });
            if queues.wake_requests || queues.requests.len() >= maximum || now >= batch_deadline {
                queues.wake_requests = false;
                let count = maximum.min(queues.requests.len());
                held.extend(queues.requests.drain(..count));
                self.shared
                    .requests_taken
                    .fetch_add(count as u64, Ordering::Relaxed);
                self.shared.queue_nanoseconds.fetch_add(
                    held.iter()
                        .map(|r| now.duration_since(r.at).as_nanos() as u64)
                        .sum(),
                    Ordering::Relaxed,
                );
                return Ok(count);
            }
            queues = self
                .shared
                .changed
                .wait_timeout(queues, batch_deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn take_records(&self, maximum: usize, wait_ms: u64) -> Result<usize, String> {
        let mut held = self.records.lock().unwrap();
        if held.count != 0 {
            return Err("前一正式记录批次尚未释放".into());
        }
        let deadline = Instant::now() + Duration::from_millis(wait_ms);
        let mut queues = self.shared.queues.lock().unwrap_or_else(|e| e.into_inner());
        while queues.records.is_empty() && !queues.wake_records && Instant::now() < deadline {
            if let Some(error) = &queues.error {
                return Err(error.clone());
            }
            queues = self
                .shared
                .changed
                .wait_timeout(queues, deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        if let Some(error) = &queues.error {
            return Err(error.clone());
        }
        queues.wake_records = false;
        let count = maximum.min(queues.records.len());
        let records: Vec<_> = queues.records.drain(..count).collect();
        drop(queues);
        let mut metadata = vec![];
        for mut record in records {
            let offset = held.arrays.len();
            if let Some(sample) = record.metadata.get_mut("sample").filter(|s| !s.is_null()) {
                for index in sample["bundle"].as_array_mut().unwrap() {
                    *index = json!(index.as_u64().unwrap() as usize + offset);
                }
                for decision in sample["decisions"].as_array_mut().unwrap() {
                    for key in ["options", "target"] {
                        decision[key] = json!(decision[key].as_u64().unwrap() as usize + offset);
                    }
                }
            }
            held.arrays.extend(record.arrays);
            metadata.push(record.metadata);
        }
        held.metadata = serde_json::to_vec(&metadata).map_err(|e| e.to_string())?;
        held.count = count;
        Ok(count)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for sender in &self.senders {
            let _ = sender.send(Control::Stop);
        }
        self.wake();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
