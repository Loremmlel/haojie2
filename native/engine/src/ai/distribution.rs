use super::keys::{self, Keys};
use crate::{
    chance,
    model::{Catalog, Command, Failure, State},
};
use std::collections::VecDeque;

pub struct Outcome {
    pub state: State,
    pub weight: f64,
}
impl Clone for Outcome {
    fn clone(&self) -> Self {
        Self {
            state: self.state.fork(),
            weight: self.weight,
        }
    }
}
pub struct Distribution {
    pub outcomes: Vec<Outcome>,
    pub sampled: bool,
    pub attempts: usize,
}
// 参数与 TS distribution 一一对应，避免仅为参数数量增加包装层。
#[allow(clippy::too_many_arguments)]
pub fn distribution(
    s: &State,
    c: &Command,
    limit: usize,
    samples: usize,
    salt: usize,
    sample_key: Option<&str>,
    keys: &Keys,
    catalog: &Catalog,
) -> Result<Distribution, String> {
    let mut result = Distribution {
        outcomes: vec![],
        sampled: false,
        attempts: 0,
    };
    let run = |tape: &[f64], fallback| {
        chance::using(tape, fallback, || {
            crate::resolution::without_presentation(|| crate::command::apply_runtime(s, c, catalog))
        })
    };
    let mut queue = VecDeque::from([(vec![], 1.)]);
    while let Some((tape, weight)) = queue.pop_front() {
        result.attempts += 1;
        let (next, cuts) = run(&tape, None);
        match next {
            Ok(mut state) => {
                state.events.clear();
                state.extra.insert("log".into(), serde_json::json!([]));
                result.outcomes.push(Outcome { state, weight });
            }
            Err(Failure::Uncertain) if cuts.is_some() => {
                let cuts = cuts.unwrap();
                if (cuts.len() > 5 && !(c.kind == "summon" && limit >= 40 && tape.is_empty()))
                    || queue.len() + result.outcomes.len() + cuts.len() > limit
                    || tape.len() >= 6
                {
                    result.sampled = true;
                    break;
                }
                for pair in cuts.windows(2) {
                    let mut tape = tape.clone();
                    tape.push((pair[0] + pair[1]) / 2.);
                    queue.push_back((tape, weight * (pair[1] - pair[0])));
                }
            }
            Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => {
                result.outcomes.clear();
                return Ok(result);
            }
            Err(e) => return Err(format!("AI distribution {e:?}")),
        }
    }
    if result.sampled {
        result.outcomes.clear();
        let fallback = || format!("{}{}{salt}", keys.position(s), keys::command(c));
        let seed = chance::hash(
            sample_key
                .map(str::to_owned)
                .unwrap_or_else(fallback)
                .as_str(),
        );
        for lane in 0..samples {
            result.attempts += 1;
            let (next, _) = run(&[], Some((seed, lane, samples)));
            match next {
                Ok(mut state) => {
                    state.events.clear();
                    state.extra.insert("log".into(), serde_json::json!([]));
                    result.outcomes.push(Outcome {
                        state,
                        weight: 1. / samples as f64,
                    });
                }
                Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => {
                    result.outcomes.clear();
                    return Ok(result);
                }
                Err(e) => return Err(format!("AI sample {e:?}")),
            }
        }
    }
    let mut grouped = indexmap::IndexMap::<String, Outcome>::new();
    for outcome in result.outcomes {
        let key = keys.position(&outcome.state);
        if let Some(old) = grouped.get_mut(&key) {
            old.weight += outcome.weight;
        } else {
            grouped.insert(key, outcome);
        }
    }
    result.outcomes = grouped.into_values().collect();
    Ok(result)
}
