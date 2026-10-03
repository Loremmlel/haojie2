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
    id: u64,
    tensor_bytes: Vec<u8>,
}
impl<R: BufRead, W: Write> Wire<R, W> {
    fn read(&mut self) -> Result<Value, String> {
        let mut bytes = vec![];
        (&mut self.input)
            .take(16 * 1024 * 1024 + 1)
            .read_until(b'\n', &mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            return Err("host disconnected".into());
        }
        if bytes.len() > 16 * 1024 * 1024 || !bytes.ends_with(b"\n") {
            return Err("invalid control frame".into());
        }
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())
    }
    fn send(&mut self, value: &Value) -> Result<(), String> {
        serde_json::to_writer(&mut self.output, value).map_err(|e| e.to_string())?;
        self.output.write_all(b"\n").map_err(|e| e.to_string())?;
        self.output.flush().map_err(|e| e.to_string())
    }
    fn tensor(&mut self, mut meta: Value, input: &encoding::Input) -> Result<(), String> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Protocol);
        // 同步写完后才允许下一请求复用；公开张量的顺序、精度和二进制协议不变。
        let bytes = &mut self.tensor_bytes;
        bytes.clear();
        bytes.reserve(
            input.entities.len() * (64 * 4 + 8 + 1)
                + 32 * 4
                + input.candidates.len() * (64 * 4 + 8 * 2 + 1),
        );
        for row in &input.entities {
            for &n in row.iter() {
                bytes.extend_from_slice(&(n as f32).to_le_bytes());
            }
        }
        for &n in &input.globals {
            bytes.extend_from_slice(&(n as f32).to_le_bytes());
        }
        for row in &input.candidates {
            for &n in row {
                bytes.extend_from_slice(&(n as f32).to_le_bytes());
            }
        }
        for &n in &input.kinds {
            bytes.extend_from_slice(&(n as i64).to_le_bytes());
        }
        for list in [&input.sources, &input.targets] {
            for &n in list {
                bytes.extend_from_slice(&(n as i64).to_le_bytes());
            }
        }
        bytes.extend(input.entity_mask.iter().map(|b| u8::from(*b)));
        bytes.extend(input.candidate_mask.iter().map(|b| u8::from(*b)));
        meta["entities"] = json!(input.entities.len());
        meta["candidates"] = json!(input.candidates.len());
        meta["bytes"] = json!(bytes.len());
        self.send(&meta)?;
        self.output
            .write_all(&self.tensor_bytes)
            .map_err(|e| e.to_string())?;
        self.output.flush().map_err(|e| e.to_string())
    }
}
impl<R: BufRead, W: Write> Inference for Wire<R, W> {
    fn logits(&mut self, input: &encoding::Input) -> Result<Vec<f64>, String> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Protocol);
        self.id += 1;
        self.tensor(
            json!({"type":"infer","id":self.id,"model":self.model}),
            input,
        )?;
        let response = self.read()?;
        encoding::known(&response, &["id", "model", "logits", "cancel"])?;
        if response["id"] != self.id || response["model"] != self.model {
            return Err("stale inference/model response".into());
        }
        if response["cancel"] == true {
            return Err("cancelled".into());
        }
        let logits: Vec<f64> =
            serde_json::from_value(response["logits"].clone()).map_err(|e| e.to_string())?;
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
    record: String,
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
    if request.op != "sample"
        || request.model.len() != 64
        || !request.model.bytes().all(|c| c.is_ascii_hexdigit())
        || request.max_commands == 0
        || request.max_commands > 20000
        || request.max_plies == 0
        || request.max_plies > 1000
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
    let mut writer = Writer::new(Path::new(&request.record))?;
    writer.push(json!({"type":"game","format":records::FORMAT,"rulesHash":records::rules_hash(),"ruleset":crate::model::RULESET,"encoding":catalog.encoding["encoding"],"model":request.model,"start":request.start,"initialHash":records::state_hash(&state)?,"samplerSeed":request.sampler_seed,"policyKind":"model-gumbel-backtracking-v1","maxCommands":request.max_commands,"maxPlies":request.max_plies}))?;
    wire.model = request.model;
    let mut random = Random::new(request.sampler_seed);
    let mut count = 0;
    let mut offer = true;
    let mut reason = "commands";
    let mut error = None;
    let mut metrics = sampler::Metrics::default();
    while !state.extra.contains_key("winner") && count < request.max_commands {
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
            let public = observation.position();
            let optional = offer
                && public.pending.is_empty()
                && public.phase != "shrine-draft"
                && public.pieces().any(|u| u.owner == other && u.has("u7"));
            let mut selected = Selected::Empty;
            let mut path = vec![];
            let mut off_turn = false;
            metrics.observation_ms += observed.elapsed().as_secs_f64() * 1000.0;
            if optional {
                let observed = Instant::now();
                let other_observation = runtime::public_view(&state, other)?;
                metrics.observation_ms += observed.elapsed().as_secs_f64() * 1000.0;
                let (choice, trace, costs) =
                    sampler::external(&other_observation, other, wire, &mut random, catalog, true)?;
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
                let (choice, trace, costs) =
                    sampler::external(&observation, actor, wire, &mut random, catalog, false)?;
                selected = choice;
                path = trace;
                metrics.merge(costs);
                offer = true;
            }
            let Selected::Command(command) = selected else {
                return Err("no complete legal command".into());
            };
            let step = Instant::now();
            let next = boundary::step_typed(&state, actor, &command, catalog)?;
            metrics.step_ms += step.elapsed().as_secs_f64() * 1000.0;
            let record = Instant::now();
            writer.push(json!({"type":"sample","index":count,"actor":actor,"command":command,"path":path,"optional":off_turn,"before":records::state_hash(&state)?,"after":records::state_hash(&next)?,"model":wire.model,"samplerState":random.0}))?;
            metrics.record_ms += record.elapsed().as_secs_f64() * 1000.0;
            state = next;
            count += 1;
            Ok(())
        })();
        if let Err(e) = result {
            reason = if e == "cancelled" {
                "cancelled"
            } else {
                "error"
            };
            error = Some(e);
            break;
        }
    }
    let outcome = records::outcome(&state, count, reason);
    writer.push(outcome.clone())?;
    writer.finish()?;
    let result = json!({"type":"done","outcome":outcome,"error":error,"elapsedMs":start.elapsed().as_secs_f64()*1000.0,"finalHash":records::state_hash(&state)?,"model":wire.model,"metrics":metrics});
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
        id: 0,
        tensor_bytes: Vec::new(),
    };
    wire.send(&json!({"type":"ready","protocol":PROTOCOL,"rulesHash":records::rules_hash(),"ruleset":crate::model::RULESET,"schema":catalog.encoding,"engine":crate::identity::describe()}))?;
    loop {
        let request = wire.read()?;
        let result = match request["op"].as_str() {
            Some("close") => return Ok(()),
            Some("sample") => sample(request, &catalog, &mut wire),
            Some("audit") => {
                encoding::known(&request, &["op", "record", "encode"])?;
                let path = request["record"].as_str().ok_or("missing record path")?;
                records::audit(Path::new(path), &catalog, |meta, input| {
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
                .map(|report| json!({"type":"done","report":report}))
            }
            _ => Err("unknown training operation".into()),
        };
        wire.send(&result.unwrap_or_else(|error| json!({"type":"error","error":error})))?;
    }
}
