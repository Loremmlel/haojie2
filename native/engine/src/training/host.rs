//! 常驻训练宿主：JSON 控制与拥有型二进制公开张量，递归决策上下文不离开 Rust。
use crate::{
    boundary, encoding,
    model::Catalog,
    policy::Random,
    records::{self, Start, Writer},
    runtime,
    sampler::{self, Inference, Selected},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Read, Write},
    path::Path,
    time::Instant,
};

pub const PROTOCOL: &str = "haojie-training-binary-v1";
pub struct Wire<R, W> {
    input: R,
    output: W,
    model: String,
    actor: usize,
    id: u64,
    command_index: usize,
    ply: f64,
    decision_id: u64,
    emit_decisions: bool,
    tensor_bytes: Vec<u8>,
    control_bytes: Vec<u8>,
}
impl<R: BufRead, W: Write> Wire<R, W> {
    fn read(&mut self) -> Result<Value, String> {
        let bytes = &mut self.control_bytes;
        bytes.clear();
        (&mut self.input)
            .take(16 * 1024 * 1024 + 1)
            .read_until(b'\n', bytes)
            .map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            return Err("host disconnected".into());
        }
        if bytes.len() > 16 * 1024 * 1024 || !bytes.ends_with(b"\n") {
            return Err("invalid control frame".into());
        }
        serde_json::from_slice(bytes).map_err(|e| e.to_string())
    }
    fn send(&mut self, value: &Value) -> Result<(), String> {
        serde_json::to_writer(&mut self.output, value).map_err(|e| e.to_string())?;
        self.output.write_all(b"\n").map_err(|e| e.to_string())?;
        self.output.flush().map_err(|e| e.to_string())
    }
    fn tensor(&mut self, mut meta: Value, input: &encoding::Input) -> Result<(), String> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Protocol);
        let bytes = &mut self.tensor_bytes;
        let payload_size = encoding::tensor_size(input);
        meta["entities"] = json!(input.entities.len());
        meta["candidates"] = json!(input.candidates.len());
        meta["bytes"] = json!(payload_size);
        // 控制头与张量直接生成到同一最终帧，不再先生成完整载荷后搬到外围缓冲。
        bytes.clear();
        bytes.reserve(payload_size + 256);
        serde_json::to_writer(&mut *bytes, &meta).map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        encoding::append_tensor_bytes(input, bytes);
        self.output
            .write_all(&self.tensor_bytes)
            .map_err(|e| e.to_string())?;
        self.output.flush().map_err(|e| e.to_string())
    }
}
impl<R: BufRead, W: Write> Inference for Wire<R, W> {
    fn decision(
        &mut self,
        input: &encoding::Input,
        selected: usize,
        depth: usize,
        stage: &str,
        pass: bool,
    ) -> Result<(), String> {
        if !self.emit_decisions {
            return Ok(());
        }
        self.decision_id += 1;
        self.tensor(
            json!({"type":"example","index":self.command_index,"decision":self.decision_id,
            "step":depth,"stage":stage,"actor":self.actor,"model":self.model,"selected":selected,
            "pass":pass,"semantics":"mc-context-v2"}),
            input,
        )
    }
    fn logits(&mut self, input: &encoding::Input) -> Result<Vec<f64>, String> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Protocol);
        self.id += 1;
        self.tensor(
            json!({"type":"infer","id":self.id,"model":self.model,
                "commandIndex":self.command_index,"ply":self.ply,"actor":self.actor}),
            input,
        )?;
        let mut response = self.read()?;
        encoding::known(&response, &["id", "model", "logits", "cancel"])?;
        if response["id"] != self.id || response["model"] != self.model {
            return Err("stale inference/model response".into());
        }
        if response["cancel"] == true {
            return Err("cancelled".into());
        }
        let logits: Vec<f64> =
            serde_json::from_value(response["logits"].take()).map_err(|e| e.to_string())?;
        if logits.len() != input.candidates.len() || logits.iter().any(|n| !n.is_finite()) {
            return Err("invalid model logits".into());
        }
        Ok(logits)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Request {
    op: String,
    #[serde(default)]
    record: String,
    #[serde(default)]
    memory: bool,
    #[serde(default)]
    models: Option<[String; 2]>,
    #[serde(default)]
    teacher: Option<usize>,
    #[serde(default)]
    mc_context: bool,
    start: Start,
    model: String,
    sampler_seed: u32,
    max_commands: usize,
    max_plies: u32,
}
fn sample<R: BufRead, W: Write>(
    value: Value,
    catalog: &Catalog,
    wire: &mut Wire<R, W>,
) -> Result<Value, String> {
    #[cfg(feature = "kernel-profile")]
    let kernel = {
        crate::profile::start();
        crate::profile::scope(crate::profile::Phase::Kernel)
    };
    let request: Request = serde_json::from_value(value).map_err(|e| e.to_string())?;
    let valid_model =
        |model: &str| model.len() == 64 && model.bytes().all(|c| c.is_ascii_hexdigit());
    if request.op != "sample"
        || request.model.len() != 64
        || !request.model.bytes().all(|c| c.is_ascii_hexdigit())
        || request.max_commands == 0
        || request.max_commands > 20000
        || request.max_plies == 0
        || request.max_plies > 1000
        || (request.memory && !request.record.is_empty())
        || (!request.memory && request.record.is_empty())
        || (!request.memory && (request.models.is_some() || request.teacher.is_some()))
        || (!request.memory && request.mc_context)
        || request
            .models
            .as_ref()
            .is_some_and(|models| models.iter().any(|m| !valid_model(m)))
        || request.teacher.is_some_and(|side| side != 1 && side != 2)
    {
        return Err("invalid sampling configuration".into());
    }
    let start = Instant::now();
    #[cfg(feature = "kernel-profile")]
    let importing = crate::profile::scope(crate::profile::Phase::StateImport);
    let mut state = request.start.state(catalog)?;
    #[cfg(feature = "kernel-profile")]
    drop(importing);
    let initial_ply = state.ply;
    if state.extra.contains_key("winner") {
        return Err("sampling start already terminal".into());
    }
    let mut writer = if request.memory {
        None
    } else {
        Some(Writer::new(Path::new(&request.record))?)
    };
    let mut current_hash = records::state_hash(&state)?;
    if let Some(writer) = writer.as_mut() {
        writer.push(json!({"type":"game","format":records::FORMAT,"rulesHash":records::rules_hash(),"ruleset":crate::model::RULESET,"encoding":catalog.encoding["encoding"],"model":request.model,"start":request.start,"initialHash":current_hash,"samplerSeed":request.sampler_seed,"policyKind":"model-gumbel-backtracking-v2","maxCommands":request.max_commands,"maxPlies":request.max_plies}))?;
    }
    wire.model = request.model.clone();
    wire.emit_decisions = request.mc_context && request.memory && request.teacher.is_none();
    wire.decision_id = 0;
    let models = request
        .models
        .unwrap_or([request.model.clone(), request.model.clone()]);
    let mut random = Random::new(request.sampler_seed);
    let mut count = 0;
    let mut offer = true;
    let mut reason = "commands";
    let mut error = None;
    let mut metrics = sampler::Metrics::default();
    while !state.extra.contains_key("winner") && count < request.max_commands {
        wire.command_index = count;
        wire.ply = state.ply;
        if state.ply - initial_ply >= request.max_plies as f64 {
            reason = "plies";
            break;
        }
        let result = (|| -> Result<(), String> {
            let observed = Instant::now();
            let mut actor = runtime::viewer(&state);
            let mut observation = runtime::public_view(&state, actor)?;
            if observation.position().phase == "shrine-draft"
                && observation.position().extra["shrineDraft"]["committed"][actor.to_string()]
                    == true
            {
                actor = 3 - actor;
                observation = runtime::public_view(&state, actor)?;
            }
            let other = 3 - actor;
            wire.actor = actor;
            wire.model = models[actor - 1].clone();
            let public = observation.position();
            let optional = offer
                && public.pending.is_empty()
                && public.phase != "shrine-draft"
                && public.pieces().any(|u| u.owner == other && u.has("u7"));
            // 教师与模型具有相同回合外窗口；null 仅在可选窗口表示真实 Pass。
            if optional && request.teacher == Some(other) {
                wire.id += 1;
                wire.send(
                    &json!({"type":"teacher","id":wire.id,"actor":other,"optional":true,"commandIndex":count,"ply":state.ply,"observation":runtime::observe(&state, other)?}),
                )?;
                let response = wire.read()?;
                encoding::known(&response, &["id", "command", "cancel"])?;
                if response["id"] != wire.id {
                    return Err("stale teacher response".into());
                }
                if response["cancel"] == true {
                    return Err("cancelled".into());
                }
                if !response["command"].is_null() {
                    state = boundary::step(&state, other, &response["command"], catalog)?;
                    count += 1;
                    offer = false;
                    metrics.off_turn_commands += 1;
                    return Ok(());
                }
                metrics.off_turn_passes += 1;
            }
            let mut selected = Selected::Empty;
            let mut path = vec![];
            let mut off_turn = false;
            let mut committed = None;
            let mut step_ms = 0.0;
            let mut rejection_record_ms = 0.0;
            let model = wire.model.clone();
            let mut accept = |actor,
                              optional,
                              command: &crate::model::Command,
                              path: &[usize],
                              status: &str,
                              sampler_state| {
                let step = Instant::now();
                let result = boundary::attempt(&state, actor, command, status, catalog)?;
                step_ms += step.elapsed().as_secs_f64() * 1000.0;
                match result {
                    Ok(next) => {
                        committed = Some(next);
                        Ok(true)
                    }
                    Err(error) => {
                        let record = Instant::now();
                        if let Some(writer) = writer.as_mut() {
                            writer.push(json!({"type":"rejected","index":count,"actor":actor,"command":command,"path":path,"optional":optional,"before":current_hash,"after":current_hash,"model":model,"samplerState":sampler_state,"error":error}))?;
                        }
                        rejection_record_ms += record.elapsed().as_secs_f64() * 1000.0;
                        Ok(false)
                    }
                }
            };
            metrics.observation_ms += observed.elapsed().as_secs_f64() * 1000.0;
            if optional && request.teacher != Some(other) {
                wire.actor = other;
                wire.model = models[other - 1].clone();
                let observed = Instant::now();
                let other_observation = runtime::public_view(&state, other)?;
                metrics.observation_ms += observed.elapsed().as_secs_f64() * 1000.0;
                let (choice, trace, costs) = (if request.mc_context {
                    sampler::external_mc
                } else {
                    sampler::external
                })(
                    &other_observation,
                    other,
                    wire,
                    &mut random,
                    catalog,
                    true,
                    &mut |c, p, s, r| accept(other, true, c, p, s, r),
                )?;
                selected = choice;
                path = trace;
                metrics.merge(costs);
                if matches!(selected, Selected::Command(_)) {
                    actor = other;
                    offer = false;
                    off_turn = true;
                    metrics.off_turn_commands += 1;
                } else if matches!(selected, Selected::Pass) {
                    metrics.off_turn_passes += 1;
                }
            }
            if !matches!(selected, Selected::Command(_)) {
                wire.actor = actor;
                wire.model = models[actor - 1].clone();
                if request.teacher == Some(actor) {
                    wire.id += 1;
                    wire.send(&json!({"type":"teacher","id":wire.id,"actor":actor,"optional":false,"commandIndex":count,"ply":state.ply,"observation":runtime::observe(&state, actor)?}))?;
                    let response = wire.read()?;
                    encoding::known(&response, &["id", "command", "cancel"])?;
                    if response["id"] != wire.id {
                        return Err("stale teacher response".into());
                    }
                    if response["cancel"] == true {
                        return Err("cancelled".into());
                    }
                    state = boundary::step(&state, actor, &response["command"], catalog)?;
                    count += 1;
                    offer = true;
                    return Ok(());
                }
                let (choice, trace, costs) = (if request.mc_context {
                    sampler::external_mc
                } else {
                    sampler::external
                })(
                    &observation,
                    actor,
                    wire,
                    &mut random,
                    catalog,
                    false,
                    &mut |c, p, s, r| accept(actor, false, c, p, s, r),
                )?;
                selected = choice;
                path = trace;
                metrics.merge(costs);
                offer = true;
            }
            let Selected::Command(command) = selected else {
                return Err("no complete legal command".into());
            };
            let next = committed.ok_or("accepted command missing state")?;
            metrics.step_ms += step_ms;
            metrics.record_ms += rejection_record_ms;
            let record = Instant::now();
            let after_hash = if request.memory {
                String::new()
            } else {
                records::state_hash(&next)?
            };
            if let Some(writer) = writer.as_mut() {
                writer.push(json!({"type":"sample","index":count,"actor":actor,"command":command,"path":path,"optional":off_turn,"before":current_hash,"after":after_hash,"model":wire.model,"samplerState":random.0}))?;
            } else if request.teacher.is_none() && !request.mc_context {
                // 只有权威接受的路径可以编码。与完整记录审核共享编码/路径验证，
                // 不传拒绝试探或每步权威快照；终局收益由 done 统一确认。
                let command = serde_json::to_value(&command).map_err(|e| e.to_string())?;
                records::examples(
                    &state,
                    actor,
                    &path,
                    off_turn,
                    &command,
                    catalog,
                    |step, stage, selected, input| {
                        wire.tensor(json!({"type":"example","index":count,"step":step,"stage":stage,"actor":actor,"model":models[actor-1],"selected":selected}), input)
                    },
                )?;
            }
            metrics.record_ms += record.elapsed().as_secs_f64() * 1000.0;
            state = next;
            // 哈希随已提交的拥有型状态推进；跨请求、载入和失败不共享缓存。
            current_hash = after_hash;
            count += 1;
            Ok(())
        })();
        if let Err(e) = result {
            reason = if e == "cancelled" {
                "cancelled"
            } else if e == sampler::DECODE_BUDGET {
                "decode-budget"
            } else {
                "error"
            };
            if reason != "decode-budget" {
                error = Some(e);
            }
            break;
        }
    }
    let outcome = records::outcome(&state, count, reason);
    if let Some(mut writer) = writer {
        writer.push(outcome.clone())?;
        writer.finish()?;
    }
    if request.memory {
        current_hash = records::state_hash(&state)?;
    }
    wire.model = request.model;
    let result = json!({"type":"done","outcome":outcome,"ply":state.ply,"error":error,"elapsedMs":start.elapsed().as_secs_f64()*1000.0,"finalHash":current_hash,"model":wire.model,"metrics":metrics});
    #[cfg(feature = "kernel-profile")]
    {
        drop(kernel);
        let mut result = result;
        result["kernelProfile"] = crate::profile::finish();
        Ok(result)
    }
    #[cfg(not(feature = "kernel-profile"))]
    Ok(result)
}
pub fn serve() -> Result<(), String> {
    let manifest: Value = serde_json::from_slice(include_bytes!("../../data/manifest.json"))
        .map_err(|e| e.to_string())?;
    if manifest["sha256"] != records::rules_hash() {
        return Err("embedded rules package hash mismatch".into());
    }
    let mut catalog = None;
    crate::handle(
        serde_json::from_slice(records::RULES).map_err(|e| e.to_string())?,
        &mut catalog,
        &mut vec![],
        &mut crate::Resident::default(),
    )?;
    let catalog = catalog.ok_or("missing rules")?;
    let mut wire = Wire {
        input: io::stdin().lock(),
        output: io::BufWriter::new(io::stdout().lock()),
        model: String::new(),
        actor: 1,
        id: 0,
        command_index: 0,
        ply: 0.0,
        decision_id: 0,
        emit_decisions: false,
        tensor_bytes: Vec::new(),
        control_bytes: Vec::new(),
    };
    wire.send(&json!({"type":"ready","protocol":PROTOCOL,"rulesHash":records::rules_hash(),"ruleset":crate::model::RULESET,"schema":catalog.encoding,"engine":crate::identity::describe(),"capabilities":["memory-examples-v1","mc-context-v2","actor-model-routing-v1","teacher-v2"]}))?;
    loop {
        let request = wire.read()?;
        let result = match request["op"].as_str() {
            Some("close") => return Ok(()),
            Some("sample") => sample(request, &catalog, &mut wire),
            Some("mc-exercise") => {
                wire.model = "0".repeat(64);
                wire.actor = 1;
                wire.command_index = 0;
                wire.emit_decisions = true;
                wire.decision_id = 0;
                crate::exercise::run(&request, &mut wire)
            }
            Some("audit") => {
                #[cfg(feature = "kernel-profile")]
                let kernel = {
                    crate::profile::start();
                    crate::profile::scope(crate::profile::Phase::Kernel)
                };
                encoding::known(&request, &["op", "record", "encode"])?;
                let path = request["record"].as_str().ok_or("missing record path")?;
                let result = records::audit(Path::new(path), &catalog, |meta, input| {
                    if request["encode"] == true {
                        if let Some(input) = input {
                            wire.tensor(meta, input)
                        } else {
                            wire.send(&meta)
                        }
                    } else {
                        Ok(())
                    }
                })
                .map(|report| json!({"type":"done","report":report}));
                #[cfg(feature = "kernel-profile")]
                {
                    drop(kernel);
                    let profile = crate::profile::finish();
                    result.map(|mut result| {
                        result["kernelProfile"] = profile;
                        result
                    })
                }
                #[cfg(not(feature = "kernel-profile"))]
                result
            }
            _ => Err("unknown training operation".into()),
        };
        wire.send(&result.unwrap_or_else(|error| json!({"type":"error","error":error})))?;
    }
}
