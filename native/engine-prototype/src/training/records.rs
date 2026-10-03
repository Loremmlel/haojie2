//! 新原生记录采用明确的规范哈希，不伪装旧 TS JSON 指纹。完整行错误不能视为截断。
use crate::{
    boundary, encoding,
    model::{Catalog, State},
    runtime,
    tree::Tree,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};

pub const FORMAT: &str = "haojie-native-record-v1";
pub const RULES: &[u8] = include_bytes!("../../data/rules.json");
pub fn bytes_hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn rules_hash() -> String {
    bytes_hash(RULES)
}
/// 类型标签；长度 u64 LE；数值统一有限 f64 LE（-0 归 0）；对象按 UTF-8 键排序。
pub fn hash(value: &Value) -> String {
    fn encode(v: &Value, h: &mut Sha256) {
        match v {
            Value::Null => h.update([0]),
            Value::Bool(b) => h.update([1, u8::from(*b)]),
            Value::Number(n) => {
                h.update([2]);
                let f = n.as_f64().unwrap();
                h.update((if f == 0.0 { 0.0 } else { f }).to_le_bytes());
            }
            Value::String(s) => {
                h.update([3]);
                h.update((s.len() as u64).to_le_bytes());
                h.update(s.as_bytes());
            }
            Value::Array(a) => {
                h.update([4]);
                h.update((a.len() as u64).to_le_bytes());
                for v in a {
                    encode(v, h);
                }
            }
            Value::Object(o) => {
                h.update([5]);
                h.update((o.len() as u64).to_le_bytes());
                let mut keys: Vec<_> = o.keys().collect();
                keys.sort();
                for k in keys {
                    encode(&json!(k), h);
                    encode(&o[k], h);
                }
            }
        }
    }
    let mut h = Sha256::new();
    encode(value, &mut h);
    format!("{:x}", h.finalize())
}
pub fn state_hash(s: &State) -> Result<String, String> {
    Ok(hash(&serde_json::to_value(s).map_err(|e| e.to_string())?))
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Played {
    pub actor: usize,
    pub command: Value,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Start {
    pub seed: u32,
    pub rules: String,
    #[serde(default)]
    pub prelude: Vec<Played>,
}
impl Start {
    pub fn state(&self, catalog: &Catalog) -> Result<State, String> {
        if self.seed == 0
            || !["classic", "shrine"].contains(&self.rules.as_str())
            || self.prelude.len() > 20000
        {
            return Err("invalid start".into());
        }
        let mut s = runtime::create(self.seed, self.rules == "shrine", catalog)?;
        s.events.clear();
        s.extra.insert("log".into(), json!([]));
        for p in &self.prelude {
            s = boundary::step(&s, p.actor, &p.command, catalog)?;
        }
        Ok(s)
    }
}
pub struct Writer {
    file: File,
    partial: PathBuf,
    target: PathBuf,
    previous: String,
}
impl Writer {
    pub fn new(path: &Path) -> Result<Self, String> {
        if path.exists() {
            return Err("record exists".into());
        }
        let partial = PathBuf::from(format!("{}.partial", path.display()));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            file,
            partial,
            target: path.into(),
            previous: String::new(),
        })
    }
    pub fn push(&mut self, body: Value) -> Result<(), String> {
        let mut envelope = json!({"previous":self.previous,"body":body});
        let digest = hash(&envelope);
        envelope["sha256"] = json!(digest);
        serde_json::to_writer(&mut self.file, &envelope).map_err(|e| e.to_string())?;
        self.file.write_all(b"\n").map_err(|e| e.to_string())?;
        self.file.flush().map_err(|e| e.to_string())?;
        self.previous = digest;
        Ok(())
    }
    pub fn finish(self) -> Result<(), String> {
        self.file.sync_all().map_err(|e| e.to_string())?;
        // 同目录硬链接的 create-new 语义防止并发覆盖已有证据；成功后只移除自己拥有的临时名字。
        std::fs::hard_link(&self.partial, &self.target).map_err(|e| e.to_string())?;
        std::fs::remove_file(&self.partial).map_err(|e| e.to_string())
    }
}
pub fn outcome(s: &State, commands: usize, reason: &str) -> Value {
    let winner = s.extra.get("winner").cloned().unwrap_or(Value::Null);
    let terminated = !winner.is_null();
    json!({"type":"outcome","commands":commands,"terminated":terminated,"reason":if terminated{"terminal"}else{reason},"winner":winner,
        "returns":if terminated {json!({"1":if winner=="draw"{0}else if winner==1{1}else{-1},"2":if winner=="draw"{0}else if winner==2{1}else{-1}})}else{Value::Null}})
}
pub fn add_pass(input: &mut encoding::Input) {
    let mut row = vec![0.0; 64];
    row[63] = 1.0;
    input.candidates.push(row);
    input.candidate_mask.push(true);
    input.sources.push(-1);
    input.targets.push(-1);
}
pub fn examples(
    s: &State,
    actor: usize,
    path: &[usize],
    optional: bool,
    command: &Value,
    catalog: &Catalog,
    mut emit: impl FnMut(usize, &str, usize, &encoding::Input) -> Result<(), String>,
) -> Result<(), String> {
    if path.is_empty() || path.len() > 256 {
        return Err("invalid decision path".into());
    }
    let observation = runtime::observe(s, actor)?;
    let mut tree = Tree::new(&observation, actor, catalog)?;
    for (step, &selected) in path.iter().enumerate() {
        let node = tree.node(&path[..step])?;
        let choice = node
            .choices
            .get(selected)
            .ok_or("decision index out of range")?;
        if (step + 1 == path.len()) != choice.next.is_none() {
            return Err("incomplete decision path".into());
        }
        if step + 1 == path.len() && choice.command != *command {
            return Err("decision command mismatch".into());
        }
        let mut input = encoding::encode_sampling(&tree, &node)?;
        if optional && step == 0 {
            add_pass(&mut input);
        }
        emit(step, node.stage, selected, &input)?;
    }
    Ok(())
}
pub fn audit(
    path: &Path,
    catalog: &Catalog,
    mut emit: impl FnMut(Value, Option<&encoding::Input>) -> Result<(), String>,
) -> Result<Value, String> {
    let mut source = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let mut state = None;
    let mut previous = String::new();
    let mut count = 0;
    let mut complete = false;
    let mut summary = Value::Null;
    let mut model = String::new();
    let mut max_commands = 0;
    let mut max_plies = 0.0;
    let mut initial_ply = 0.0;
    let mut incomplete_tail = false;
    loop {
        let mut line = vec![];
        (&mut source)
            .take(16 * 1024 * 1024 + 1)
            .read_until(b'\n', &mut line)
            .map_err(|e| e.to_string())?;
        if line.len() > 16 * 1024 * 1024 {
            return Err("record frame exceeds limit".into());
        }
        if line.is_empty() {
            break;
        }
        if !line.ends_with(b"\n") {
            if complete {
                return Err("bytes after completed outcome".into());
            }
            incomplete_tail = true;
            break;
        }
        let mut envelope: Value =
            serde_json::from_slice(&line).map_err(|e| format!("corrupt record JSON: {e}"))?;
        encoding::known(&envelope, &["previous", "body", "sha256"])?;
        let digest = envelope
            .as_object_mut()
            .ok_or("record envelope")?
            .remove("sha256")
            .ok_or("missing record hash")?;
        if envelope["previous"] != previous || digest != hash(&envelope) {
            return Err("record hash chain mismatch".into());
        }
        previous = digest.as_str().ok_or("invalid hash")?.into();
        let body = &envelope["body"];
        if complete {
            return Err("record after outcome".into());
        }
        match body["type"].as_str() {
            Some("game") => {
                encoding::known(
                    body,
                    &[
                        "type",
                        "format",
                        "rulesHash",
                        "ruleset",
                        "encoding",
                        "model",
                        "start",
                        "initialHash",
                        "samplerSeed",
                        "policyKind",
                        "maxCommands",
                        "maxPlies",
                    ],
                )?;
                if state.is_some()
                    || body["format"] != FORMAT
                    || body["rulesHash"] != rules_hash()
                    || body["ruleset"] != crate::model::RULESET
                    || body["encoding"] != catalog.encoding["encoding"]
                    || body["policyKind"] != "model-gumbel-backtracking-v1"
                {
                    return Err("record version mismatch".into());
                }
                let start: Start =
                    serde_json::from_value(body["start"].clone()).map_err(|e| e.to_string())?;
                let s = start.state(catalog)?;
                initial_ply = s.ply;
                max_commands = body["maxCommands"]
                    .as_u64()
                    .filter(|n| (1..=20000).contains(n))
                    .ok_or("invalid command limit")?;
                max_plies = body["maxPlies"]
                    .as_u64()
                    .filter(|n| (1..=1000).contains(n))
                    .ok_or("invalid ply limit")? as f64;
                if !body["samplerSeed"]
                    .as_u64()
                    .is_some_and(|n| n <= u32::MAX as u64)
                {
                    return Err("invalid sampler seed".into());
                }
                if body["initialHash"] != state_hash(&s)? {
                    return Err("initial state hash mismatch".into());
                }
                model = body["model"].as_str().ok_or("model hash missing")?.into();
                if model.len() != 64 || !model.bytes().all(|c| c.is_ascii_hexdigit()) {
                    return Err("invalid model hash".into());
                }
                emit(body.clone(), None)?;
                state = Some(s);
            }
            Some("sample") => {
                encoding::known(
                    body,
                    &[
                        "type",
                        "index",
                        "actor",
                        "command",
                        "path",
                        "optional",
                        "before",
                        "after",
                        "model",
                        "samplerState",
                    ],
                )?;
                let s = state.as_ref().ok_or("missing game")?;
                if count as u64 >= max_commands
                    || s.ply - initial_ply >= max_plies
                    || !body["samplerState"]
                        .as_u64()
                        .is_some_and(|n| n <= u32::MAX as u64)
                {
                    return Err("sample beyond boundary/invalid sampler state".into());
                }
                if body["index"] != count
                    || body["before"] != state_hash(s)?
                    || body["model"] != model
                {
                    return Err("sample identity/hash mismatch".into());
                }
                let actor = body["actor"].as_u64().ok_or("invalid actor")? as usize;
                let path: Vec<usize> =
                    serde_json::from_value(body["path"].clone()).map_err(|e| e.to_string())?;
                let optional = body["optional"].as_bool().ok_or("optional flag missing")?;
                let mut primary = runtime::viewer(s);
                let observation = runtime::observe(s, primary)?;
                if observation["phase"] == "shrine-draft"
                    && observation["shrineDraft"]["committed"][primary.to_string()] == true
                {
                    primary = 3 - primary;
                }
                if optional != (actor != primary) {
                    return Err("optional decision actor mismatch".into());
                }
                let next = boundary::step(s, actor, &body["command"], catalog)?;
                if body["after"] != state_hash(&next)? {
                    return Err("result hash mismatch".into());
                }
                examples(
                    s,
                    actor,
                    &path,
                    optional,
                    &body["command"],
                    catalog,
                    |step, stage, selected, input| {
                        emit(
                            json!({"type":"example","index":count,"step":step,"actor":actor,"command":body["command"]["type"],"stage":stage,"selected":selected}),
                            Some(input),
                        )
                    },
                )?;
                state = Some(next);
                count += 1;
            }
            Some("outcome") => {
                let s = state.as_ref().ok_or("missing game")?;
                let reason = body["reason"].as_str().ok_or("missing outcome reason")?;
                if (reason == "terminal") != s.extra.contains_key("winner") {
                    return Err("false terminal boundary".into());
                }
                if (reason == "commands" && count as u64 != max_commands)
                    || (reason == "plies" && s.ply - initial_ply < max_plies)
                {
                    return Err("false truncation boundary".into());
                }
                if !["terminal", "commands", "plies", "cancelled", "error"].contains(&reason)
                    || *body != outcome(s, count, reason)
                {
                    return Err("outcome mismatch".into());
                }
                summary = body.clone();
                emit(summary.clone(), None)?;
                complete = true;
            }
            _ => return Err("unknown record type".into()),
        }
    }
    let s = state.ok_or("empty record")?;
    if !complete {
        summary = outcome(&s, count, "interrupted");
        summary["terminated"] = json!(false);
        summary["returns"] = Value::Null;
        summary["winner"] = Value::Null;
        emit(summary.clone(), None)?;
    }
    Ok(
        json!({"outcome":summary,"complete":complete,"incompleteTail":incomplete_tail,"finalHash":state_hash(&s)?,"commands":count}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_record_vectors_cover_numbers_and_utf8_key_order() {
        let vectors: Vec<Value> =
            serde_json::from_str(include_str!("../../data/hash-vectors.json")).unwrap();
        for vector in vectors {
            let hex = vector["bytes"].as_str().unwrap();
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            assert_eq!(hash(&vector["value"]), bytes_hash(&bytes));
        }
        assert_ne!(hash(&json!({})), hash(&json!({"a": null})));
    }

    #[test]
    fn valid_hash_chain_cannot_claim_a_nonterminal_game_is_terminal() {
        let mut catalog = None;
        crate::handle(
            serde_json::from_slice(RULES).unwrap(),
            &mut catalog,
            &mut vec![],
            &mut crate::Resident::default(),
        )
        .unwrap();
        let catalog = catalog.unwrap();
        let start = Start {
            seed: 71,
            rules: "classic".into(),
            prelude: vec![],
        };
        let state = start.state(&catalog).unwrap();
        let path = std::env::temp_dir().join(format!(
            "haojie-false-terminal-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let mut writer = Writer::new(&path).unwrap();
        writer.push(json!({"type":"game","format":FORMAT,"rulesHash":rules_hash(),
            "ruleset":crate::model::RULESET,"encoding":catalog.encoding["encoding"],
            "model":"a".repeat(64),"start":start,"initialHash":state_hash(&state).unwrap(),
            "samplerSeed":91,"policyKind":"model-gumbel-backtracking-v1","maxCommands":10,"maxPlies":10})).unwrap();
        writer.push(outcome(&state, 0, "terminal")).unwrap();
        writer.finish().unwrap();
        let result = audit(&path, &catalog, |_, _| Ok(()));
        std::fs::remove_file(path).unwrap();
        assert_eq!(result.unwrap_err(), "false terminal boundary");
    }
}
