//! 完整命令级随机图搜索；在神经请求处挂起，单局模拟按原顺序串行推进。
use super::{
    ffi::{Bundle, Front, Host, Info},
    random::Random,
};
use crate::model::Command;
use indexmap::IndexMap;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

pub trait Environment {
    fn info(&self, id: u64) -> Result<Info, String>;
    fn encode(&mut self, id: u64) -> Result<(Info, Arc<Bundle>), String>;
    fn decode(&mut self, id: u64, prefix: usize) -> Result<Front, String>;
    fn sample(&mut self, id: u64, command: &Command, seed: u32) -> Result<Info, String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Graph {
        actions: usize,
        diamond: bool,
        cycle: bool,
        chain: bool,
        conditional: bool,
        encoded: HashMap<u64, usize>,
        maximum_position: u64,
    }
    impl Graph {
        fn new(actions: usize) -> Self {
            Self {
                actions,
                diamond: false,
                cycle: false,
                chain: false,
                conditional: false,
                encoded: HashMap::new(),
                maximum_position: 0,
            }
        }
        fn front(next: Vec<usize>) -> Front {
            Front {
                command: None,
                options: vec![0.0; next.len() * 704].into(),
                next,
                stage: 0,
                focus: "source",
            }
        }
    }
    impl Environment for Graph {
        fn info(&self, id: u64) -> Result<Info, String> {
            Ok(Info {
                id,
                actor: 1,
                ply: 1.0,
                winner: None,
                key: id.to_string(),
                metrics: [0.0; 4],
            })
        }
        fn encode(&mut self, id: u64) -> Result<(Info, Arc<Bundle>), String> {
            *self.encoded.entry(id).or_default() += 1;
            Ok((
                self.info(id)?,
                Arc::new(Bundle {
                    states: vec![0.0; 3 * 128].into(),
                    records: Arc::from([]),
                    owners: Arc::from([]),
                    relations: vec![0.0; 3 * 3 * 8].into(),
                }),
            ))
        }
        fn decode(&mut self, _: u64, prefix: usize) -> Result<Front, String> {
            if prefix == 0 {
                return Ok(Self::front((1..=self.actions).collect()));
            }
            if self.conditional && prefix <= 2 {
                let start = 3 + (prefix - 1) * 3;
                return Ok(Self::front((start..start + 3).collect()));
            }
            let mut command = Command::new("end");
            command.x = Some((prefix - if self.conditional { 3 } else { 1 }) as f64);
            Ok(Front {
                command: Some(command),
                next: vec![],
                options: Arc::from([]),
                stage: 2,
                focus: "target",
            })
        }
        fn sample(&mut self, id: u64, command: &Command, _: u32) -> Result<Info, String> {
            let choice = command.x.unwrap() as u64;
            let next = if self.cycle {
                0
            } else if self.chain {
                id + 1
            } else if id == 0 && self.actions <= 2 && !self.conditional {
                choice + 1
            } else if self.diamond && (id == 1 || id == 2) {
                3
            } else {
                let mut result = self.info(id)?;
                result.actor = 2;
                result.winner = Some(json!(if self.conditional
                    || if self.actions > 2 {
                        choice == 63
                    } else if self.diamond {
                        choice == 0
                    } else {
                        id == 1
                    } {
                    1
                } else {
                    2
                }));
                return Ok(result);
            };
            self.maximum_position = self.maximum_position.max(next);
            self.info(next)
        }
    }
    fn run(graph: &mut Graph, simulations: usize, depth: usize) -> SearchResult {
        let mut rng = super::super::random::testing();
        let mut search = Search::new(simulations, depth, false);
        let mut answer = None;
        loop {
            match search.advance(graph, &mut rng, answer).unwrap() {
                Progress::Done(result) => return result,
                Progress::Request(request) => {
                    answer = Some(Reply {
                        values: if request.kind == 0 {
                            vec![0.25, 0.0, 0.0, 0.0, 0.0]
                        } else {
                            vec![0.0; request.options.len() / 704]
                        },
                        version: 17,
                    });
                }
            }
        }
    }

    #[test]
    fn same_player_backup_and_transposition_encode_once() {
        let mut graph = Graph::new(2);
        let result = run(&mut graph, 80, 8);
        let winning = result
            .edges
            .iter()
            .position(|e| e.command.x == Some(0.0))
            .unwrap();
        assert!(result.policy[winning] > 0.8);
        assert_eq!(result.version, 17);
        let mut graph = Graph::new(2);
        graph.diamond = true;
        let result = run(&mut graph, 120, 8);
        assert_eq!(graph.encoded[&3], 1);
        assert_eq!(result.graph_nodes, 4);
        assert!(result.transpositions > 0);
    }

    #[test]
    fn progressive_widening_reaches_more_than_32_actions() {
        let mut graph = Graph::new(64);
        let result = run(&mut graph, 1100, 8);
        assert_eq!(result.edges.len(), 64);
        assert_eq!(graph.encoded, HashMap::from([(0, 1)]));
        assert!((result.edges.iter().map(|e| e.prior).sum::<f64>() - 1.0).abs() < 1e-7);
        assert_eq!(result.supervision.len(), 1);
        assert_eq!(result.supervision[0].target.len(), 64);
        assert!((result.supervision[0].weight - 1.0).abs() < 1e-7);
    }

    #[test]
    fn conditional_prefixes_do_not_consume_command_depth() {
        let mut graph = Graph::new(2);
        graph.conditional = true;
        let result = run(&mut graph, 40, 1);
        assert_eq!(result.edges.len(), 6);
        assert_eq!(result.graph_nodes, 1);
        assert!(
            result
                .edges
                .iter()
                .all(|e| (e.prior - 1.0 / 6.0).abs() < 1e-7)
        );
        assert!((result.supervision.iter().map(|s| s.weight).sum::<f64>() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn unique_command_cycle_and_depth_boundaries() {
        let mut graph = Graph::new(1);
        let mut rng = super::super::random::testing();
        let initial = rng.snapshot();
        let mut search = Search::new(64, 16, true);
        let Progress::Done(result) = search.advance(&mut graph, &mut rng, None).unwrap() else {
            panic!("唯一命令不应请求神经网络");
        };
        assert_eq!(initial, rng.snapshot());
        assert_eq!(result.graph_nodes, 0);
        let mut graph = Graph::new(2);
        graph.cycle = true;
        let result = run(&mut graph, 20, 8);
        assert_eq!(result.transpositions, 20);
        assert_eq!(result.graph_nodes, 1);
        assert_eq!(result.edges.iter().map(|e| e.visits).sum::<u64>(), 20);
        assert!((result.edges.iter().map(|e| e.total).sum::<f64>() - 5.0).abs() < 1e-8);
        let mut graph = Graph::new(2);
        graph.chain = true;
        let result = run(&mut graph, 40, 2);
        assert_eq!(graph.maximum_position, 2);
        assert_eq!(result.graph_nodes, 3);
    }
}
impl Environment for Host {
    fn info(&self, id: u64) -> Result<Info, String> {
        self.info_typed(id)
    }
    fn encode(&mut self, id: u64) -> Result<(Info, Arc<Bundle>), String> {
        self.encode_typed(id)
    }
    fn decode(&mut self, id: u64, prefix: usize) -> Result<Front, String> {
        self.decode_typed(id, prefix)
    }
    fn sample(&mut self, id: u64, command: &Command, seed: u32) -> Result<Info, String> {
        self.sample(id, command, seed)
    }
}

#[derive(Clone)]
pub struct Request {
    pub kind: u32,
    pub context: u64,
    pub key: String,
    pub bundle: Option<Arc<Bundle>>,
    pub options: Arc<[f32]>,
}
pub struct Reply {
    pub values: Vec<f32>,
    pub version: i64,
}
pub enum Progress {
    Request(Request),
    Done(SearchResult),
}

pub struct EdgeResult {
    pub command: Command,
    pub prior: f64,
    pub visits: u64,
    pub total: f64,
}
#[derive(Clone)]
pub struct Supervision {
    pub options: Arc<[f32]>,
    pub target: Arc<[f32]>,
    pub weight: f64,
}
pub struct SearchResult {
    pub info: Info,
    pub bundle: Option<Arc<Bundle>>,
    pub edges: Vec<EdgeResult>,
    pub policy: Vec<f32>,
    pub supervision: Vec<Supervision>,
    pub version: i64,
    pub graph_nodes: usize,
    pub transpositions: usize,
}

struct Prefix {
    next: Vec<usize>,
    options: Arc<[f32]>,
    prior: Vec<f64>,
    remaining: Vec<f64>,
}
struct Edge {
    command: Command,
    prior: f64,
    visits: u64,
    total: f64,
    paths: Vec<(f64, Vec<(usize, usize)>)>,
}
struct Node {
    info: Info,
    bundle: Arc<Bundle>,
    value: f64,
    version: i64,
    prefixes: IndexMap<usize, Prefix>,
    edges: Vec<Edge>,
    by_command: HashMap<String, usize>,
    visits: u64,
    available: f64,
}
impl Node {
    fn new(info: Info, bundle: Arc<Bundle>) -> Self {
        Self {
            info,
            bundle,
            value: 0.0,
            version: -1,
            prefixes: IndexMap::new(),
            edges: vec![],
            by_command: HashMap::new(),
            visits: 0,
            available: 1.0,
        }
    }
    fn request(&self, kind: u32, options: Arc<[f32]>) -> Request {
        Request {
            kind,
            context: self.info.id,
            key: self.info.key.clone(),
            bundle: (kind == 0).then(|| self.bundle.clone()),
            options,
        }
    }
    fn select(&self) -> Result<usize, String> {
        let mut best = None;
        let scale = (1.0 + self.visits as f64).sqrt();
        for (index, edge) in self.edges.iter().enumerate() {
            let score = edge.total / edge.visits.max(1) as f64
                + 1.5 * edge.prior * scale / (1.0 + edge.visits as f64);
            // 与 Python max 一致，平手保留第一个已插入的边。
            if best.is_none_or(|(_, previous)| score > previous) {
                best = Some((index, score));
            }
        }
        best.map(|(index, _)| index)
            .ok_or_else(|| "非终局没有合法完整命令".into())
    }
}

struct Expansion {
    node: usize,
    prefix: usize,
    trace: Vec<(usize, usize)>,
    logp: f64,
}
enum Pending {
    Encode {
        node: usize,
        root: bool,
    },
    Decode {
        node: usize,
        prefix: usize,
        next: Vec<usize>,
        options: Arc<[f32]>,
    },
}

pub struct Search {
    simulations: usize,
    depth: usize,
    noise: bool,
    started: bool,
    nodes: Vec<Node>,
    graph: HashMap<String, usize>,
    completed: usize,
    current: Option<usize>,
    level: usize,
    path: Vec<(usize, usize)>,
    seen: HashSet<String>,
    expansion: Option<Expansion>,
    pending: Option<Pending>,
    transpositions: usize,
    finished: bool,
}

// NumPy 对连续浮点数组使用八路分组的成对求和；保留原精度与分组顺序。
macro_rules! numpy_sum {
    ($name:ident, $ty:ty) => {
        pub fn $name(values: &[$ty]) -> $ty {
            let n = values.len();
            if n < 8 {
                return values.iter().copied().fold(-0.0, |a, b| a + b);
            }
            if n <= 128 {
                let mut parts: [$ty; 8] = values[..8].try_into().unwrap();
                let end = n - n % 8;
                for block in (8..end).step_by(8) {
                    for i in 0..8 {
                        parts[i] += values[block + i];
                    }
                }
                let mut total = ((parts[0] + parts[1]) + (parts[2] + parts[3]))
                    + ((parts[4] + parts[5]) + (parts[6] + parts[7]));
                for value in &values[end..] {
                    total += *value;
                }
                return total;
            }
            let middle = (n / 2) - (n / 2) % 8;
            $name(&values[..middle]) + $name(&values[middle..])
        }
    };
}
numpy_sum!(sum_f32, f32);
numpy_sum!(sum_f64, f64);

fn probabilities(logits: &[f32], rng: &Random) -> Result<Vec<f64>, String> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("条件策略的神经输出无效".into());
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // NumPy 的 SIMD exp 与系统 expf 可差一个最低位，足以改变 PUCT 近平手顺序。
    let centered: Vec<f32> = logits.iter().map(|value| *value - maximum).collect();
    let values = rng.exp(&centered)?;
    let total = sum_f32(&values);
    Ok(values.into_iter().map(|v| (v / total) as f64).collect())
}

impl Search {
    pub fn new(simulations: usize, depth: usize, noise: bool) -> Self {
        Self {
            simulations,
            depth,
            noise,
            started: false,
            nodes: vec![],
            graph: HashMap::new(),
            completed: 0,
            current: None,
            level: 0,
            path: vec![],
            seen: HashSet::new(),
            expansion: None,
            pending: None,
            transpositions: 0,
            finished: false,
        }
    }

    fn backup(&mut self, mut actor: usize, mut value: f64) {
        for &(node, edge) in self.path.iter().rev() {
            let parent = &mut self.nodes[node];
            if parent.info.actor != actor {
                value = -value;
            }
            actor = parent.info.actor;
            parent.edges[edge].visits += 1;
            parent.edges[edge].total += value;
            parent.visits += 1;
        }
        self.completed += 1;
        self.current = None;
    }

    fn add_prefix(
        &mut self,
        node: usize,
        prefix: usize,
        next: Vec<usize>,
        options: Arc<[f32]>,
        logits: Option<&[f32]>,
        rng: &mut Random,
    ) -> Result<(), String> {
        let mut prior = if let Some(logits) = logits {
            if logits.len() != next.len() {
                return Err("条件策略输出数量不符".into());
            }
            probabilities(logits, rng)?
        } else {
            vec![1.0]
        };
        if self.noise && node == 0 && prefix == 0 && prior.len() > 1 {
            for (probability, noise) in prior.iter_mut().zip(rng.dirichlet(next.len())) {
                *probability = 0.75 * *probability + 0.25 * noise;
            }
        }
        let length = prior.len();
        self.nodes[node].prefixes.insert(
            prefix,
            Prefix {
                next,
                options,
                prior,
                remaining: vec![1.0; length],
            },
        );
        Ok(())
    }

    fn finish(&mut self) -> Result<SearchResult, String> {
        self.finished = true;
        let root = &self.nodes[0];
        let total: u64 = root.edges.iter().map(|edge| edge.visits).sum();
        let policy = if self.simulations == 0 {
            vec![1.0]
        } else {
            if total == 0 {
                return Err("搜索没有产生访问量".into());
            }
            root.edges
                .iter()
                .map(|edge| (edge.visits as f64 / total as f64) as f32)
                .collect()
        };
        let mut targets: IndexMap<usize, Vec<f64>> = IndexMap::new();
        for (edge, &mass) in root.edges.iter().zip(&policy) {
            for (probability, path) in &edge.paths {
                for &(prefix, choice) in path {
                    let branch = &root.prefixes[&prefix];
                    if branch.prior.len() == 1 {
                        continue;
                    }
                    let target = targets
                        .entry(prefix)
                        .or_insert_with(|| vec![0.0; branch.prior.len()]);
                    target[choice] += mass as f64 * probability / edge.prior;
                }
            }
        }
        let supervision = targets
            .into_iter()
            .filter_map(|(prefix, target)| {
                let total = sum_f64(&target);
                (total > 0.0).then(|| Supervision {
                    options: root.prefixes[&prefix].options.clone(),
                    target: target
                        .into_iter()
                        .map(|v| (v / total) as f32)
                        .collect::<Vec<_>>()
                        .into(),
                    weight: total,
                })
            })
            .collect();
        Ok(SearchResult {
            info: root.info.clone(),
            bundle: Some(root.bundle.clone()),
            edges: root
                .edges
                .iter()
                .map(|e| EdgeResult {
                    command: e.command.clone(),
                    prior: e.prior,
                    visits: e.visits,
                    total: e.total,
                })
                .collect(),
            policy,
            supervision,
            version: root.version,
            graph_nodes: self.nodes.len(),
            transpositions: self.transpositions,
        })
    }

    pub fn advance<E: Environment>(
        &mut self,
        environment: &mut E,
        rng: &mut Random,
        answer: Option<Reply>,
    ) -> Result<Progress, String> {
        if self.finished {
            return Err("搜索已经完成".into());
        }
        if let Some(pending) = self.pending.take() {
            let answer = answer.ok_or("缺少神经响应")?;
            match pending {
                Pending::Encode { node, root } => {
                    self.nodes[node].value = *answer.values.first().ok_or("空价值响应")? as f64;
                    self.nodes[node].version = answer.version;
                    if !root {
                        let child = &self.nodes[node];
                        self.backup(child.info.actor, child.value);
                    }
                }
                Pending::Decode {
                    node,
                    prefix,
                    next,
                    options,
                } => {
                    self.add_prefix(node, prefix, next, options, Some(&answer.values), rng)?;
                }
            }
        } else if answer.is_some() {
            return Err("没有等待中的神经请求".into());
        }

        if !self.started {
            self.started = true;
            let mut prefix = 0;
            loop {
                let front = environment.decode(0, prefix)?;
                if let Some(command) = front.command {
                    self.finished = true;
                    return Ok(Progress::Done(SearchResult {
                        info: environment.info(0)?,
                        bundle: None,
                        edges: vec![EdgeResult {
                            command,
                            prior: 1.0,
                            visits: 0,
                            total: 0.0,
                        }],
                        policy: vec![1.0],
                        supervision: vec![],
                        version: -1,
                        graph_nodes: 0,
                        transpositions: 0,
                    }));
                }
                if front.next.len() != 1 {
                    break;
                }
                prefix = front.next[0];
            }
            let (info, bundle) = environment.encode(0)?;
            self.graph.insert(info.key.clone(), 0);
            self.nodes.push(Node::new(info, bundle));
            self.pending = Some(Pending::Encode {
                node: 0,
                root: true,
            });
            return Ok(Progress::Request(self.nodes[0].request(0, Arc::from([]))));
        }

        loop {
            if self.simulations > 0 && self.completed == self.simulations {
                return Ok(Progress::Done(self.finish()?));
            }
            if self.current.is_none() {
                self.current = Some(0);
                self.level = 0;
                self.path.clear();
                self.seen.clear();
                self.seen.insert(self.nodes[0].info.key.clone());
            }
            let node = self.current.unwrap();
            if self.simulations > 0 && self.level >= self.depth {
                self.backup(self.nodes[node].info.actor, self.nodes[node].value);
                continue;
            }
            if self.expansion.is_none() {
                let limit = (2.0 * (self.nodes[node].visits as f64 + 1.0).sqrt()).ceil() as usize;
                if self.nodes[node].available > 0.0 && self.nodes[node].edges.len() < limit.max(1) {
                    self.expansion = Some(Expansion {
                        node,
                        prefix: 0,
                        trace: vec![],
                        logp: 0.0,
                    });
                }
            }
            let mut chosen = None;
            while let Some(mut expansion) = self.expansion.take() {
                let node = expansion.node;
                let prefix = expansion.prefix;
                if !self.nodes[node].prefixes.contains_key(&prefix) {
                    let front = environment.decode(self.nodes[node].info.id, prefix)?;
                    if let Some(command) = front.command {
                        let parent = &mut self.nodes[node];
                        let mut available = 0.0;
                        for &(p, choice) in expansion.trace.iter().rev() {
                            let branch = parent.prefixes.get_mut(&p).unwrap();
                            branch.remaining[choice] = available;
                            available = branch
                                .prior
                                .iter()
                                .zip(&branch.remaining)
                                .map(|(p, r)| p * r)
                                .sum();
                        }
                        parent.available = available;
                        let prior = expansion.logp.exp();
                        let key = serde_json::to_string(&command).map_err(|e| e.to_string())?;
                        let index = if let Some(&index) = parent.by_command.get(&key) {
                            parent.edges[index].prior += prior;
                            parent.edges[index].paths.push((prior, expansion.trace));
                            index
                        } else {
                            let index = parent.edges.len();
                            parent.by_command.insert(key, index);
                            parent.edges.push(Edge {
                                command,
                                prior,
                                visits: 0,
                                total: 0.0,
                                paths: vec![(prior, expansion.trace)],
                            });
                            index
                        };
                        chosen = Some(index);
                        break;
                    }
                    if front.next.is_empty() {
                        return Err("合法解码前缀没有后继".into());
                    }
                    if front.next.len() > 1 {
                        let request = self.nodes[node].request(1, front.options.clone());
                        self.pending = Some(Pending::Decode {
                            node,
                            prefix,
                            next: front.next,
                            options: front.options,
                        });
                        self.expansion = Some(expansion);
                        return Ok(Progress::Request(request));
                    }
                    self.add_prefix(node, prefix, front.next, front.options, None, rng)?;
                }
                let branch = &self.nodes[node].prefixes[&prefix];
                let mut weights: Vec<_> = branch
                    .prior
                    .iter()
                    .zip(&branch.remaining)
                    .map(|(p, r)| p * r)
                    .collect();
                let total = sum_f64(&weights);
                if total <= 0.0 {
                    break;
                }
                for value in &mut weights {
                    *value /= total;
                }
                let choice = rng.choice(&weights)?;
                expansion.logp += branch.prior[choice].max(1e-300).ln();
                expansion.trace.push((prefix, choice));
                expansion.prefix = branch.next[choice];
                self.expansion = Some(expansion);
            }
            let edge = chosen
                .map(Ok)
                .unwrap_or_else(|| self.nodes[node].select())?;
            if self.simulations == 0 {
                return Ok(Progress::Done(self.finish()?));
            }
            let result = environment.sample(
                self.nodes[node].info.id,
                &self.nodes[node].edges[edge].command,
                rng.simulation_seed(),
            )?;
            self.path.push((node, edge));
            if let Some(winner) = &result.winner {
                let value = if winner == "draw" {
                    0.0
                } else if winner.as_u64() == Some(result.actor as u64) {
                    1.0
                } else {
                    -1.0
                };
                self.backup(result.actor, value);
                continue;
            }
            if let Some(&child) = self.graph.get(&result.key) {
                self.transpositions += 1;
                if self.seen.contains(&result.key) {
                    self.backup(self.nodes[child].info.actor, self.nodes[child].value);
                } else {
                    self.seen.insert(result.key);
                    self.current = Some(child);
                    self.level += 1;
                }
            } else {
                let (info, bundle) = environment.encode(result.id)?;
                let child = self.nodes.len();
                self.graph.insert(info.key.clone(), child);
                self.nodes.push(Node::new(info, bundle));
                self.pending = Some(Pending::Encode {
                    node: child,
                    root: false,
                });
                return Ok(Progress::Request(
                    self.nodes[child].request(0, Arc::from([])),
                ));
            }
        }
    }
}
