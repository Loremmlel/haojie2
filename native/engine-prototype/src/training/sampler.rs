//! 整个参数采样循环驻留原生进程；正式状态与策略随机流分开，仅 Observation 进入树和网络。
use crate::model::{Catalog, Command};
use crate::{
    actions, encoding,
    policy::{Random, TinyPolicy},
    runtime,
    tree::Tree,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Instant;

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    tree_ms: f64,
    encoding_ms: f64,
    inference_ms: f64,
    sampling_ms: f64,
    pub observation_ms: f64,
    pub step_ms: f64,
    pub record_ms: f64,
    nodes: usize,
    evaluations: usize,
    forced: usize,
    backtracks: usize,
    max_entities: usize,
    max_candidates: usize,
    pub off_turn_commands: usize,
    pub off_turn_passes: usize,
}
impl Metrics {
    pub fn merge(&mut self, other: Self) {
        self.tree_ms += other.tree_ms;
        self.encoding_ms += other.encoding_ms;
        self.inference_ms += other.inference_ms;
        self.sampling_ms += other.sampling_ms;
        self.nodes += other.nodes;
        self.evaluations += other.evaluations;
        self.forced += other.forced;
        self.backtracks += other.backtracks;
        self.max_entities = self.max_entities.max(other.max_entities);
        self.max_candidates = self.max_candidates.max(other.max_candidates);
    }
}
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
pub enum Selected {
    Command(Box<Command>),
    Pass,
    Empty,
}
struct Sampling<'a> {
    projections: Option<crate::policy::Decision<'a>>,
    random: &'a mut Random,
    metrics: &'a mut Metrics,
    optional: bool,
    nodes: usize,
    external: Option<&'a mut dyn Inference>,
    trace: Vec<usize>,
}
pub trait Inference {
    fn logits(&mut self, input: &encoding::Input) -> Result<Vec<f64>, String>;
}
impl Sampling<'_> {
    fn visit(&mut self, tree: &mut Tree<'_>, cursor: &[usize]) -> Result<Selected, String> {
        self.nodes += 1;
        if self.nodes > 4096 || cursor.len() > 256 {
            return Err("参数解码预算耗尽".into());
        }
        self.metrics.nodes += 1;
        let t = Instant::now();
        let node = tree.node(cursor)?;
        self.metrics.tree_ms += ms(t);
        let pass = self.optional && cursor.is_empty();
        if node.choices.is_empty() {
            return Ok(if pass {
                Selected::Pass
            } else {
                Selected::Empty
            });
        }
        let count = node.choices.len() + usize::from(pass);
        self.metrics.max_candidates = self.metrics.max_candidates.max(count);
        let mut order: Vec<usize> = (0..count).collect();
        if count == 1 {
            self.metrics.forced += 1;
        } else {
            let t = Instant::now();
            let mut input = encoding::encode_sampling(tree, &node)?;
            if pass {
                let mut row = vec![0.0; 64];
                row[63] = 1.0;
                input.candidates.push(row);
                input.candidate_mask.push(true);
                input.sources.push(-1);
                input.targets.push(-1);
            }
            self.metrics.encoding_ms += ms(t);
            self.metrics.max_entities = self.metrics.max_entities.max(input.entities.len());
            self.metrics.evaluations += 1;
            let t = Instant::now();
            let logits = if let Some(inference) = self.external.as_mut() {
                inference.logits(&input)?
            } else {
                self.projections
                    .as_mut()
                    .map(|p| p.logits(&input))
                    .unwrap_or_else(|| vec![0.0; count])
            };
            self.metrics.inference_ms += ms(t);
            if logits.len() != count || logits.iter().any(|v| !v.is_finite()) {
                return Err("invalid policy logits".into());
            }
            let t = Instant::now();
            let scores: Vec<f64> = logits
                .iter()
                .map(|v| v - (-self.random.next().ln()).ln())
                .collect();
            order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
            self.metrics.sampling_ms += ms(t);
        }
        for i in order {
            if pass && i == node.choices.len() {
                self.trace.insert(0, i);
                return Ok(Selected::Pass);
            }
            let c = &node.choices[i];
            if c.next.is_none() {
                self.trace.insert(0, i);
                return Ok(Selected::Command(Box::new(c.command.clone())));
            }
            let mut next = cursor.to_vec();
            next.push(i);
            let selected = self.visit(tree, &next)?;
            if !matches!(selected, Selected::Empty) {
                self.trace.insert(0, i);
                return Ok(selected);
            }
            self.metrics.backtracks += 1;
        }
        Ok(Selected::Empty)
    }
}
fn sample(
    observation: &runtime::PublicPosition,
    actor: usize,
    policy: Option<&TinyPolicy>,
    random: &mut Random,
    metrics: &mut Metrics,
    catalog: &Catalog,
    optional: bool,
) -> Result<Selected, String> {
    let t = Instant::now();
    if observation.viewer != actor {
        return Err("public view actor mismatch".into());
    }
    let mut tree = Tree::from_view(observation, catalog)?;
    metrics.tree_ms += ms(t);
    Sampling {
        random,
        metrics,
        optional,
        nodes: 0,
        projections: policy.map(TinyPolicy::decision),
        external: None,
        trace: vec![],
    }
    .visit(&mut tree, &[])
}
/// 正式模型沿用同一 Gumbel 排序与空分支回溯；不声称路径概率等于各节点概率乘积。
pub fn external(
    observation: &runtime::PublicPosition,
    actor: usize,
    inference: &mut dyn Inference,
    random: &mut Random,
    catalog: &Catalog,
    optional: bool,
) -> Result<(Selected, Vec<usize>, Metrics), String> {
    let start = Instant::now();
    if observation.viewer != actor {
        return Err("public view actor mismatch".into());
    }
    let mut tree = Tree::from_view(observation, catalog)?;
    let mut metrics = Metrics {
        tree_ms: ms(start),
        ..Metrics::default()
    };
    let mut sampling = Sampling {
        projections: None,
        random,
        metrics: &mut metrics,
        optional,
        nodes: 0,
        external: Some(inference),
        trace: vec![],
    };
    let selected = sampling.visit(&mut tree, &[])?;
    Ok((selected, sampling.trace, metrics))
}
/// 本地受控基准批次：固定命令/回合上限，失败返回已完成前缀与 unknown，绝不伪造终局收益。
/// 墙钟只计时，不参与选招。取消由宿主终止此独立进程；不会写文件或修改驻留规则会话。
pub fn game(request: &Value, catalog: &Catalog) -> Result<Value, String> {
    encoding::validate_schema(&catalog.encoding)?;
    let seed = request["seed"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= u32::MAX as u64)
        .ok_or("seed must be nonzero uint32")? as u32;
    let rules = actions::text(&request["rules"]);
    if !["classic", "shrine"].contains(&rules) {
        return Err("invalid rules".into());
    }
    let positive = |key| {
        request[key]
            .as_u64()
            .filter(|n| *n > 0 && *n <= 9007199254740991)
            .ok_or_else(|| format!("invalid {key}"))
    };
    let max_commands = positive("maxCommands")? as usize;
    let max_plies = positive("maxPlies")? as f64;
    let policy_seed = request["policySeed"].as_u64().unwrap_or(73129) as u32;
    let sampler_seed = request["samplerSeed"]
        .as_u64()
        .unwrap_or(0x6ab921d3u32.wrapping_add(2654435761) as u64) as u32;
    let policy = match request["policy"].as_str().unwrap_or("tiny") {
        "tiny" => Some(TinyPolicy::new(policy_seed)),
        "uniform" => None,
        _ => return Err("invalid policy".into()),
    };
    let start = Instant::now();
    // 固定工作集从宿主导入权威起点；策略仍只接收 observe 白名单。
    #[cfg(feature = "kernel-profile")]
    let importing = crate::profile::scope(crate::profile::Phase::StateImport);
    let mut state = if let Some(initial) = request.get("initialState") {
        serde_json::from_value(initial.clone()).map_err(|e| e.to_string())?
    } else {
        runtime::create(seed, rules == "shrine", catalog)?
    };
    state.events.clear();
    state.extra.insert("log".into(), json!([]));
    #[cfg(feature = "kernel-profile")]
    drop(importing);
    let initial_ply = state.ply;
    let mut random = Random::new(sampler_seed);
    let mut metrics = Metrics::default();
    let mut commands = vec![];
    let mut offer_interrupt = true;
    let mut error = None;
    while !state.extra.contains_key("winner")
        && commands.len() < max_commands
        && state.ply - initial_ply < max_plies
    {
        let selected = (|| -> Result<(usize, Command), String> {
            let t = Instant::now();
            let mut actor = runtime::viewer(&state);
            let mut observation = runtime::public_view(&state, actor)?;
            if observation.position().phase == "shrine-draft"
                && observation.position().extra["shrineDraft"]["committed"][actor.to_string()]
                    == true
            {
                actor = 3 - actor;
                observation = runtime::public_view(&state, actor)?;
            }
            metrics.observation_ms += ms(t);
            let other = 3 - actor;
            let mut selected = Selected::Empty;
            let t = Instant::now();
            let public = observation.position();
            let interrupt = offer_interrupt
                && public.pending.is_empty()
                && public.phase != "shrine-draft"
                && public.pieces().any(|u| u.owner == other && u.has("u7"));
            metrics.tree_ms += ms(t);
            if interrupt {
                let t = Instant::now();
                let off_observation = runtime::public_view(&state, other)?;
                metrics.observation_ms += ms(t);
                selected = sample(
                    &off_observation,
                    other,
                    policy.as_ref(),
                    &mut random,
                    &mut metrics,
                    catalog,
                    true,
                )?;
                match selected {
                    Selected::Command(_) => {
                        actor = other;
                        metrics.off_turn_commands += 1;
                        offer_interrupt = false;
                    }
                    Selected::Pass => metrics.off_turn_passes += 1,
                    Selected::Empty => return Err("没有完整合法回合外命令".into()),
                }
            }
            if !matches!(selected, Selected::Command(_)) {
                selected = sample(
                    &observation,
                    actor,
                    policy.as_ref(),
                    &mut random,
                    &mut metrics,
                    catalog,
                    false,
                )?;
                offer_interrupt = true;
            }
            match selected {
                Selected::Command(c) => Ok((actor, *c)),
                _ => Err("没有完整合法命令".into()),
            }
        })();
        let (actor, command) = match selected {
            Ok(v) => v,
            Err(e) => {
                error = Some(e);
                break;
            }
        };
        let t = Instant::now();
        let c = &command;
        if !actions::permitted_command(&state, actor, &command) {
            error = Some("selected unauthorized command".into());
            break;
        }
        match crate::apply_runtime(&state, c, catalog) {
            Ok(mut next) => {
                next.events.clear();
                next.extra.insert("log".into(), json!([]));
                state = next;
            }
            Err(e) => {
                error = Some(format!("selected command failed: {e:?}"));
                break;
            }
        }
        metrics.step_ms += ms(t);
        let t = Instant::now();
        commands.push(json!({"actor":actor,"command":command}));
        metrics.record_ms += ms(t);
    }
    let terminated = state.extra.contains_key("winner");
    let truncation = if terminated {
        None
    } else if commands.len() >= max_commands {
        Some("commands")
    } else if state.ply - initial_ply >= max_plies {
        Some("plies")
    } else {
        None
    };
    let winner = state.extra.get("winner").cloned().unwrap_or(Value::Null);
    let returns = if terminated {
        json!({"1":if winner=="draw"{0}else if winner==1{1}else{-1},"2":if winner=="draw"{0}else if winner==2{1}else{-1}})
    } else {
        Value::Null
    };
    let elapsed_ms = ms(start);
    #[cfg(feature = "kernel-profile")]
    let _exporting = crate::profile::scope(crate::profile::Phase::StateExport);
    Ok(
        json!({"seed":seed,"rules":rules,"policySeed":policy_seed,"samplerSeed":sampler_seed,"parameters":if policy.is_some(){12929}else{0},"commands":commands,"metrics":metrics,"elapsedMs":elapsed_ms,"status":{"commands":commands.len(),"ply":state.ply,"phase":state.phase,"terminated":terminated,"truncated":truncation.is_some(),"truncation":truncation,"winner":winner,"returns":returns},"error":error,"state":state,"observations":[runtime::observe(&state,1)?,runtime::observe(&state,2)?]}),
    )
}
