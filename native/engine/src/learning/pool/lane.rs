//! 逐局搜索和正式提交；随机数、手工计划与根版本归所属工作线程所有。
use super::{Array, Config, Record};
use crate::{
    learning::{
        ffi::{Bundle, Host, Info},
        random::{Functions, Random},
        search::{Progress, Reply, Request, Search, SearchResult, Supervision},
        timing::{self, Phase},
    },
    model::Command,
};
use serde_json::{Value, json};
use std::sync::Arc;

pub enum Advanced {
    Request(Request),
    Record(Record),
}

pub struct Lane {
    pub game: Value,
    pub steps: u64,
    pub info: Info,
    pub root_version: Option<i64>,
    pub done: bool,
    host: Host,
    rng: Random,
    root_rng: Option<Random>,
    root_teacher: Option<Value>,
    search: Option<Search>,
    answer: Option<Reply>,
    deep: bool,
    config: Arc<Config>,
}

impl Lane {
    pub fn new(input: &Value, config: Arc<Config>, functions: Functions) -> Result<Self, String> {
        let game = input["game"].clone();
        let mut host = Host::new(game["seed"].as_u64().ok_or("缺少对局seed")? as u32)?;
        let restored = &input["restored"];
        if !restored.is_null() {
            host.restore(&restored["environment"])?;
            if !restored["teacher"].is_null() {
                host.teacher_restore(&restored["teacher"]);
            }
        }
        let rng = Random::from_value(
            if restored.is_null() {
                &input["rng"]
            } else {
                &restored["rng"]
            },
            functions,
        )?;
        Ok(Self {
            game,
            steps: restored["steps"].as_u64().unwrap_or(0),
            info: host.info_typed(0)?,
            root_version: restored["root_version"].as_i64(),
            done: false,
            host,
            rng,
            root_rng: None,
            root_teacher: None,
            search: None,
            answer: None,
            deep: false,
            config,
        })
    }

    pub fn model(&self) -> i64 {
        if self.game["opponent"] == "history"
            && self.info.actor as u64 != self.game["learner"].as_u64().unwrap_or(1)
        {
            self.game["version"].as_i64().unwrap_or(-1)
        } else {
            self.root_version.unwrap_or(-1)
        }
    }

    pub fn reply(&mut self, answer: Reply) {
        self.root_version = Some(answer.version);
        self.answer = Some(answer);
    }

    fn teacher_turn(&self) -> bool {
        self.game["opponent"] == "demo"
            || (self.game["opponent"] == "teacher"
                && self.info.actor as u64 != self.game["learner"].as_u64().unwrap_or(1))
    }

    fn teacher_decisions(
        &mut self,
        command: &Command,
        weight: f64,
    ) -> Result<Vec<Supervision>, String> {
        let mut decisions = vec![];
        for (prefix, index) in self.host.trace_typed(0, command)? {
            let front = self.host.decode_typed(0, prefix)?;
            if front.next.len() > 1 {
                let mut target = vec![0.; front.next.len()];
                target[index] = 1.;
                decisions.push(Supervision {
                    options: front.options,
                    target: target.into(),
                    weight,
                });
            }
        }
        Ok(decisions)
    }

    pub fn advance(&mut self) -> Result<Advanced, String> {
        let evaluation = self.game["evaluation"].as_bool().unwrap_or(false);
        if self.search.is_none() {
            self.root_rng = Some(self.rng.clone());
            self.deep = if evaluation {
                true
            } else {
                self.rng.random() < self.config.full_fraction
            };
            if self.teacher_turn() {
                self.root_teacher = Some(self.host.teacher_snapshot());
                let difficulty = self.game["difficulty"]
                    .as_str()
                    .unwrap_or("easy")
                    .to_owned();
                let command = self.host.teacher_next(&difficulty)?;
                let sample = if self.game["opponent"] == "demo" && !evaluation {
                    let (info, bundle) = self.host.encode_typed(0)?;
                    Some((info, bundle, self.teacher_decisions(&command, 1.)?, -1, 0))
                } else {
                    None
                };
                return Ok(Advanced::Record(self.commit(command, sample, None)?));
            }
            self.search = Some(Search::new(
                if self.deep {
                    self.config.simulations
                } else {
                    self.config.fast_simulations
                },
                self.config.depth,
                !evaluation && self.game["opponent"] != "evaluation",
            ));
        }
        match self.search.as_mut().unwrap().advance(
            &mut self.host,
            &mut self.rng,
            self.answer.take(),
        )? {
            Progress::Request(request) => Ok(Advanced::Request(request)),
            Progress::Done(result) => {
                let index = if self.config.temperature == 0. {
                    let mut best = 0;
                    for i in 1..result.policy.len() {
                        if result.policy[i] > result.policy[best] {
                            best = i;
                        }
                    }
                    best
                } else {
                    let mut probability: Vec<f64> = result
                        .policy
                        .iter()
                        .map(|p| (*p as f64).powf(1. / self.config.temperature))
                        .collect();
                    let sum = crate::learning::search::sum_f64(&probability);
                    for p in &mut probability {
                        *p /= sum;
                    }
                    self.rng.choice(&probability)?
                };
                let command = result.edges[index].command.clone();
                let learning = self.game["opponent"] == "self"
                    || self.game["opponent"] == "demo"
                    || result.info.actor as u64 == self.game["learner"].as_u64().unwrap_or(1);
                // 落子已由搜索选定。按局号错开绝对步数，暂停恢复无需另存计数或消耗随机数。
                let mut sample = None;
                if learning
                    && !evaluation
                    && let Some(bundle) = &result.bundle
                {
                    let mut decisions = if self.deep {
                        result.supervision.clone()
                    } else {
                        vec![]
                    };
                    let interval = self.config.teacher_label_interval;
                    let mut teacher_prefixes = 0;
                    if interval > 0
                        && self
                            .steps
                            .wrapping_add(self.game["id"].as_u64().unwrap_or(0))
                            .is_multiple_of(interval)
                    {
                        let advice = self.host.teacher_advice()?;
                        let labels = self.teacher_decisions(&advice, 0.5)?;
                        teacher_prefixes = labels.len();
                        decisions.extend(labels);
                    }
                    if !decisions.is_empty() {
                        sample = Some((
                            result.info.clone(),
                            bundle.clone(),
                            decisions,
                            result.version,
                            teacher_prefixes,
                        ));
                    }
                }
                Ok(Advanced::Record(self.commit(
                    command,
                    sample,
                    Some(&result),
                )?))
            }
        }
    }

    fn commit(
        &mut self,
        command: Command,
        sample: Option<(Info, Arc<Bundle>, Vec<Supervision>, i64, usize)>,
        result: Option<&SearchResult>,
    ) -> Result<Record, String> {
        let _scope = timing::scope(Phase::CommitRecord);
        let info = self.host.apply(&command)?;
        self.steps += 1;
        let limit = self.game["ply_limit"].as_u64().unwrap_or(0);
        let truncation = if info.winner.is_some() {
            None
        } else if self.steps >= self.config.max_commands {
            Some("command_limit")
        } else if limit != 0 && info.ply > limit as f64 {
            Some("ply_limit")
        } else {
            None
        };
        self.done = info.winner.is_some() || truncation.is_some();
        let mut arrays = vec![];
        let sample = sample.map(|(root, bundle, decisions, version, teacher_prefixes)| {
            arrays.extend(Array::bundle(&bundle));
            let mut metadata = vec![];
            for decision in decisions {
                let index = arrays.len();
                let count = decision.target.len();
                arrays.push(Array { values: decision.options, shape: vec![count, 704] });
                arrays.push(Array { values: decision.target, shape: vec![count] });
                metadata.push(json!({"options":index,"target":index+1,"weight":decision.weight}));
            }
            json!({"actor":root.actor,"ply":root.ply,"version":version,"bundle":[0,1,2,3],"decisions":metadata,"teacher_prefixes":teacher_prefixes})
        });
        let mut metadata = json!({"game":self.game,"info":info.value(),"command":command,
            "steps":self.steps,"done":self.done,"truncation":truncation,"sample":sample});
        if truncation.is_some() {
            metadata["environment"] = self.host.snapshot();
        }
        if self.config.audit {
            let mut audit = json!({"rng_after":self.rng.snapshot()});
            if let Some(result) = result {
                audit["edges"] = json!(result.edges.iter().map(|e| json!({"command":e.command,"prior":e.prior,"visits":e.visits,"total":e.total})).collect::<Vec<_>>());
                audit["policy"] = json!(result.policy);
                audit["graph_nodes"] = json!(result.graph_nodes);
                audit["transpositions"] = json!(result.transpositions);
            }
            metadata["audit"] = audit;
        }
        self.info = info;
        self.search = None;
        self.root_rng = None;
        self.root_teacher = None;
        self.root_version = None;
        Ok(Record { metadata, arrays })
    }

    pub fn snapshot(&self) -> Value {
        let teacher = if self.game["opponent"] == "demo" || self.game["opponent"] == "teacher" {
            self.root_teacher
                .clone()
                .unwrap_or_else(|| self.host.teacher_snapshot())
        } else {
            Value::Null
        };
        json!({"game":self.game,"environment":self.host.snapshot(),"steps":self.steps,
            "rng":self.root_rng.as_ref().unwrap_or(&self.rng).snapshot(),"root_version":self.root_version,"teacher":teacher})
    }
}
