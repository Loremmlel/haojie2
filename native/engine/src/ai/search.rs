use super::{
    candidates,
    distribution::{self, Distribution, Outcome},
    evaluation,
    keys::{self, Keys},
    ranking::command_priority,
    settings::Difficulty,
};
use crate::{
    chance, geometry,
    model::{Catalog, Command, State},
    runtime,
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct Step {
    before: String,
    command: Command,
}
#[derive(Clone)]
struct Node {
    outcomes: Vec<Outcome>,
    score: f64,
    path: Vec<Step>,
    root: String,
    expandable: bool,
    sampled: bool,
}
struct Context<'a> {
    keys: Keys,
    catalog: &'a Catalog,
    count: usize,
    candidates: usize,
    sampled: usize,
    replies: usize,
    depth: usize,
    max: usize,
    stop_at: usize,
    side: usize,
    difficulty: Difficulty,
    cache: HashMap<String, f64>,
    scenario_seed: u32,
}
impl Context<'_> {
    fn stopped(&self) -> bool {
        self.count >= self.stop_at
    }
    fn value(&mut self, s: &State) -> f64 {
        let key = self.keys.position(s);
        if let Some(v) = self.cache.get(&key) {
            return *v;
        }
        let v = evaluation::evaluate(s, self.side, self.catalog);
        if self.cache.len() > 3000 {
            self.cache.clear();
        }
        self.cache.insert(key, v);
        v
    }
    fn expectation(&mut self, out: &[Outcome]) -> f64 {
        let scores: Vec<_> = out.iter().map(|o| self.value(&o.state)).collect();
        let mean = out
            .iter()
            .zip(&scores)
            .map(|(o, s)| o.weight * s)
            .sum::<f64>();
        mean - if self.difficulty == Difficulty::Hard {
            0.06
        } else {
            0.
        } * out
            .iter()
            .zip(&scores)
            .map(|(o, s)| o.weight * (mean - s).max(0.))
            .sum::<f64>()
    }
    fn scenario(&self, s: &State, c: &Command, sample: usize) -> String {
        let u = c.unit_id.as_deref().and_then(|id| s.unit(id));
        let mut values = vec![
            json!(self.scenario_seed),
            json!(sample),
            json!(s.ply),
            json!(s.summon_slots),
            json!(c.kind),
            json!(c.unit_id),
            json!(c.card_id),
            json!(u.map(|u| u.shots)),
            json!(u.map(|u| u.moves)),
            json!(u.map(|u| u.operations)),
            json!(u.map(|u| u.free_used)),
        ];
        if let Some(value) = u.and_then(|u| u.extra.get("rerollUsedPly")) {
            values.push(value.clone());
        }
        keys::stringify(&json!(values))
    }
    fn simulate(
        &mut self,
        s: &State,
        c: &Command,
        limit: usize,
        samples: usize,
        salt: usize,
        key: Option<&str>,
    ) -> Result<Distribution, String> {
        let d =
            distribution::distribution(s, c, limit, samples, salt, key, &self.keys, self.catalog)?;
        self.count += d.attempts;
        if d.sampled {
            self.sampled += 1;
        }
        Ok(d)
    }
}
fn terminal(s: &State) -> bool {
    s.extra.contains_key("winner")
}
fn expand(
    ctx: &mut Context,
    s: &State,
    path: &[Step],
    level: Difficulty,
    lean: bool,
    salt: usize,
) -> Result<Vec<Node>, String> {
    let settings = level.settings();
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    let mut work: Vec<_> = candidates::groups(s, level, lean, ctx.catalog)?
        .into_iter()
        .map(|mut g| {
            if lean || !path.is_empty() {
                g.keep = 1;
            }
            (g, 0usize, 0usize)
        })
        .collect();
    if lean {
        let mut priorities = Vec::new();
        for (g, _, _) in &mut work {
            priorities.push(g.priority.unwrap_or_else(|| {
                command_priority(s, &g.commands(s, ctx.catalog)[0], ctx.catalog)
            }));
        }
        let mut order: Vec<_> = (0..work.len()).collect();
        order.sort_by(|a, b| priorities[*b].total_cmp(&priorities[*a]));
        work = order.into_iter().take(3).map(|i| work[i].clone()).collect();
    }
    let local_start = ctx.count;
    let local_limit = if lean {
        usize::MAX
    } else if !path.is_empty() {
        24
    } else {
        40.max((ctx.max as f64 * 0.16).floor() as usize)
    };
    let mut live = work.len();
    while live > 0 {
        let pass_start = result.len();
        live = 0;
        for (g, index, accepted) in &mut work {
            let length = g.commands(s, ctx.catalog).len();
            if *index >= length || *accepted >= g.keep {
                continue;
            }
            live += 1;
            if (ctx.stopped() || ctx.count - local_start >= local_limit)
                && !result.is_empty()
                && !(s.phase == "summon" && g.family == "summon" && *accepted < g.keep)
            {
                return Ok(result);
            }
            let command = g.commands[*index].clone();
            *index += 1;
            let key = keys::command(&command);
            if !seen.insert(key.clone()) {
                continue;
            }
            let scenario = lean.then(|| ctx.scenario(s, &command, salt));
            let d = ctx.simulate(
                s,
                &command,
                if lean {
                    2
                } else if command.kind == "summon" {
                    64
                } else {
                    settings.chance_limit
                },
                if lean { 1 } else { settings.samples },
                salt,
                scenario.as_deref(),
            )?;
            if !d.outcomes.is_empty() {
                ctx.candidates += 1;
                *accepted += 1;
                if !lean {
                    ctx.depth = ctx.depth.max(path.len() + 1);
                }
                let mut next_path = path.to_vec();
                next_path.push(Step {
                    before: ctx.keys.fingerprint(s),
                    command,
                });
                let only = if d.outcomes.len() == 1 {
                    Some(&d.outcomes[0].state)
                } else {
                    None
                };
                let expandable = only.is_some_and(|v| {
                    !terminal(v)
                        && v.active == s.active
                        && v.phase == s.phase
                        && runtime::viewer(v) == runtime::viewer(s)
                });
                let score = ctx.expectation(&d.outcomes)
                    - next_path
                        .iter()
                        .filter(|p| p.command.kind == "move")
                        .count() as f64
                        * 0.2;
                result.push(Node {
                    outcomes: d.outcomes,
                    sampled: d.sampled,
                    score,
                    path: next_path,
                    root: if path.is_empty() {
                        key
                    } else {
                        keys::command(&path[0].command)
                    },
                    expandable,
                });
            }
        }
        if !lean && path.is_empty() && !ctx.stopped() {
            let new = result[pass_start..].to_vec();
            result.extend(attack_continuations(ctx, &new, level)?);
            let new = result[pass_start..].to_vec();
            result.extend(deployment_continuations(ctx, s, &new)?);
        }
    }
    Ok(result)
}
fn deployment_continuations(
    ctx: &mut Context,
    before: &State,
    roots: &[Node],
) -> Result<Vec<Node>, String> {
    let rows = geometry::deployment_rows(before, before.active);
    let mut added = Vec::new();
    for root in roots {
        if ctx.stopped() {
            break;
        }
        if root.path.last().is_none_or(|p| p.command.kind != "move")
            || !root.expandable
            || root.sampled
            || root.outcomes.len() != 1
        {
            continue;
        }
        let state = &root.outcomes[0].state;
        if !state.pending.is_empty() {
            continue;
        }
        let gained: Vec<_> = geometry::deployment_rows(state, state.active)
            .into_iter()
            .filter(|y| !rows.contains(y))
            .collect();
        if gained.is_empty() {
            continue;
        }
        for command in candidates::deployment_candidates(state, &gained, ctx.catalog)
            .into_iter()
            .take(4)
        {
            if ctx.stopped() {
                break;
            }
            let d = ctx.simulate(state, &command, 1, 1, 0, None)?;
            if d.outcomes.is_empty() {
                continue;
            }
            ctx.candidates += 1;
            let expandable = d.outcomes.len() == 1
                && !terminal(&d.outcomes[0].state)
                && d.outcomes[0].state.pending.is_empty()
                && d.outcomes[0].state.active == state.active;
            let mut path = root.path.clone();
            path.push(Step {
                before: ctx.keys.fingerprint(state),
                command,
            });
            ctx.depth = ctx.depth.max(path.len());
            added.push(Node {
                score: ctx.expectation(&d.outcomes) - 0.2,
                outcomes: d.outcomes,
                sampled: d.sampled,
                root: root.root.clone(),
                path,
                expandable,
            });
        }
    }
    Ok(added)
}
fn attack_continuations(
    ctx: &mut Context,
    roots: &[Node],
    level: Difficulty,
) -> Result<Vec<Node>, String> {
    let mut added = Vec::new();
    for root in roots {
        let first = &root.path.last().unwrap().command;
        if first.kind != "attack" || first.unit_id.is_none() || !root.expandable {
            continue;
        }
        let mut current = root.clone();
        for _ in 0..5 {
            if ctx.stopped() || current.outcomes.len() != 1 {
                break;
            }
            let state = &current.outcomes[0].state;
            if !state.pending.is_empty() || terminal(state) {
                break;
            }
            let mut followups = candidates::attack_candidates(
                state,
                first.unit_id.as_deref().unwrap(),
                None,
                ctx.catalog,
            );
            if first
                .target_id
                .as_deref()
                .is_some_and(|id| state.unit(id).is_some() || id.starts_with("base-"))
            {
                for ally in &state.units {
                    if ally.owner == state.active
                        && Some(ally.id.as_str()) != first.unit_id.as_deref()
                    {
                        followups.extend(candidates::attack_candidates(
                            state,
                            &ally.id,
                            first.target_id.as_deref(),
                            ctx.catalog,
                        ));
                    }
                }
            }
            let mut ranked: Vec<_> = followups
                .into_iter()
                .map(|c| {
                    let priority = command_priority(state, &c, ctx.catalog);
                    (c, priority)
                })
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut best: Option<Node> = None;
            let mut focused: Option<Node> = None;
            for (command, _) in
                ranked
                    .into_iter()
                    .take(if level == Difficulty::Hard { 6 } else { 3 })
            {
                if ctx.stopped() {
                    break;
                }
                let d = ctx.simulate(
                    state,
                    &command,
                    level.settings().chance_limit,
                    level.settings().samples,
                    0,
                    None,
                )?;
                if d.outcomes.is_empty() {
                    continue;
                }
                ctx.candidates += 1;
                let expandable = d.outcomes.len() == 1
                    && !terminal(&d.outcomes[0].state)
                    && d.outcomes[0].state.active == state.active
                    && runtime::viewer(&d.outcomes[0].state) == runtime::viewer(state);
                let target = command.target_id == first.target_id;
                let mut path = current.path.clone();
                path.push(Step {
                    before: ctx.keys.fingerprint(state),
                    command,
                });
                ctx.depth = ctx.depth.max(path.len());
                let node = Node {
                    score: ctx.expectation(&d.outcomes),
                    outcomes: d.outcomes,
                    sampled: current.sampled || d.sampled,
                    root: root.root.clone(),
                    path,
                    expandable,
                };
                if target && focused.as_ref().is_none_or(|v| node.score > v.score) {
                    focused = Some(node.clone());
                }
                if best.as_ref().is_none_or(|v| node.score > v.score) {
                    best = Some(node);
                }
            }
            let Some(best) = focused.or(best) else {
                break;
            };
            current = best.clone();
            if best.score > root.score + 0.01 {
                added.push(best.clone());
            }
            if !best.expandable {
                break;
            }
        }
    }
    Ok(added)
}
fn diverse(ctx: &Context, nodes: &[Node], width: usize, seen: &mut HashSet<String>) -> Vec<Node> {
    let mut sorted = nodes.to_vec();
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut chosen = Vec::new();
    let mut roots = HashMap::<String, usize>::new();
    for node in sorted {
        if !node.expandable {
            continue;
        }
        let key = ctx.keys.position(&node.outcomes[0].state);
        if seen.contains(&key) || *roots.get(&node.root).unwrap_or(&0) >= 2.max(width.div_ceil(3)) {
            continue;
        }
        *roots.entry(node.root.clone()).or_default() += 1;
        seen.insert(key);
        chosen.push(node);
        if chosen.len() == width {
            break;
        }
    }
    chosen
}
fn rollout(
    ctx: &mut Context,
    initial: &State,
    owner: usize,
    scenario: usize,
) -> Result<State, String> {
    let mut state = initial.fork();
    for _ in 0..100 {
        if terminal(&state) || ctx.stopped() || state.active != owner && state.pending.is_empty() {
            break;
        }
        let mover = runtime::viewer(&state);
        let mut nodes = expand(ctx, &state, &[], Difficulty::Easy, true, scenario)?;
        nodes.sort_by(|a, b| {
            if mover == ctx.side {
                b.score.total_cmp(&a.score)
            } else {
                a.score.total_cmp(&b.score)
            }
        });
        let Some(choice) = nodes.first() else {
            break;
        };
        let mut draw = (f64::from(chance::hash(&ctx.scenario(
            &state,
            &choice.path[0].command,
            scenario,
        ))) + 0.5)
            / 4294967296.;
        let mut selected = choice.outcomes.last().unwrap();
        for out in &choice.outcomes {
            draw -= out.weight;
            if draw <= 0. {
                selected = out;
                break;
            }
        }
        state = selected.state.fork();
    }
    Ok(state)
}
fn at_turn_end(ctx: &mut Context, s: &State, node: &Node) -> Result<Option<Node>, String> {
    if node.sampled {
        return Ok(None);
    }
    let mut outcomes = Vec::new();
    for out in &node.outcomes {
        if terminal(&out.state) {
            outcomes.push(out.clone());
            continue;
        }
        if !out.state.pending.is_empty() {
            return Ok(None);
        }
        if out.state.ply == s.ply + 1. && out.state.active != s.active {
            outcomes.push(out.clone());
            continue;
        }
        if out.state.ply != s.ply || out.state.active != s.active || ctx.stopped() {
            return Ok(None);
        }
        let d = ctx.simulate(
            &out.state,
            &Command::new("end"),
            ctx.difficulty.settings().chance_limit,
            3,
            0,
            None,
        )?;
        if !d.outcomes.is_empty() {
            ctx.candidates += 1;
        }
        if d.sampled
            || d.outcomes.is_empty()
            || d.outcomes.iter().any(|o| {
                !terminal(&o.state)
                    && (!o.state.pending.is_empty()
                        || o.state.ply != s.ply + 1.
                        || o.state.active == s.active)
            })
        {
            return Ok(None);
        }
        outcomes.extend(d.outcomes.into_iter().map(|mut o| {
            o.weight *= out.weight;
            o
        }));
    }
    Ok(Some(Node {
        score: ctx.expectation(&outcomes),
        outcomes,
        expandable: false,
        ..node.clone()
    }))
}
fn reconsider(ctx: &mut Context, s: &State, nodes: &[Node]) -> Result<(Vec<Node>, usize), String> {
    let Some(end) = nodes
        .iter()
        .find(|n| n.path.len() == 1 && n.path[0].command.kind == "end")
    else {
        return Ok((vec![], 0));
    };
    let Some(baseline) = at_turn_end(ctx, s, end)? else {
        return Ok((vec![], 0));
    };
    let mut best = baseline.clone();
    let mut improved = false;
    let mut checks = 0;
    let tactical = ["attack", "skill", "charge", "cast", "equip"];
    let mut alternatives: Vec<_> = nodes
        .iter()
        .filter(|n| {
            !n.sampled
                && !n.outcomes.is_empty()
                && n.path
                    .iter()
                    .all(|p| tactical.contains(&p.command.kind.as_str()))
        })
        .collect();
    alternatives.sort_by(|a, b| {
        command_priority(s, &b.path[0].command, ctx.catalog)
            .total_cmp(&command_priority(s, &a.path[0].command, ctx.catalog))
            .then(b.path.len().cmp(&a.path.len()))
    });
    let mut seen = HashSet::new();
    for node in alternatives {
        if ctx.stopped() {
            break;
        }
        let key = keys::stringify(&json!(
            node.path
                .iter()
                .map(|p| keys::command_value(&p.command))
                .collect::<Vec<_>>()
        ));
        if !seen.insert(key) {
            continue;
        }
        let candidate = at_turn_end(ctx, s, node)?;
        checks += 1;
        if let Some(candidate) = candidate
            && candidate.score > best.score + 0.01
        {
            best = candidate;
            improved = true;
        }
    }
    Ok((
        if improved {
            vec![best, baseline]
        } else {
            vec![]
        },
        checks,
    ))
}
pub fn decide(
    observation: &Value,
    side: usize,
    difficulty: Difficulty,
    budget: Option<usize>,
    catalog: &Catalog,
) -> Result<Value, String> {
    let s = crate::options::position(observation)?;
    if s.extra.contains_key("mode") {
        return Err("Rust AI supports classic only".into());
    }
    let cfg = difficulty.settings();
    let max = budget.unwrap_or(cfg.nodes).max(40);
    let reserve = if s.phase == "play" && s.pending.is_empty() {
        160.min(max / 5)
    } else {
        0
    };
    let search_max = max - reserve;
    let keys = Keys::new(observation);
    let scenario_seed = chance::hash(&keys.position(&s));
    let mut ctx = Context {
        keys,
        catalog,
        count: 0,
        candidates: 0,
        sampled: 0,
        replies: 0,
        depth: 1,
        max,
        stop_at: search_max,
        side,
        difficulty,
        cache: HashMap::new(),
        scenario_seed,
    };
    if terminal(&s) || runtime::viewer(&s) != side {
        return Ok(
            json!({"command":null,"plan":[],"stats":{"simulations":0,"candidates":0,"depth":0,"replies":0,"sampled":0,"exhausted":false}}),
        );
    }
    if difficulty == Difficulty::Hard {
        ctx.stop_at = search_max.min(40.max((max as f64 * 0.4).floor() as usize));
    }
    let synthesis = s.phase == "synthesis" && s.pending.is_empty();
    let roots = expand(&mut ctx, &s, &[], difficulty, false, 0)?;
    let mut by_root = indexmap::IndexMap::<String, Node>::new();
    for n in &roots {
        if by_root.get(&n.root).is_none_or(|old| n.score > old.score) {
            by_root.insert(n.root.clone(), n.clone());
        }
    }
    ctx.stop_at = if difficulty == Difficulty::Hard {
        search_max.min(ctx.count.max((max as f64 * 0.4).floor() as usize))
    } else {
        search_max
    };
    let mut seen = HashSet::new();
    let mut beam = diverse(&ctx, &roots, cfg.width, &mut seen);
    for _ in 2..=cfg.depth {
        if synthesis || beam.is_empty() || ctx.stopped() {
            break;
        }
        let mut children = Vec::new();
        for parent in &beam {
            if ctx.stopped() {
                break;
            }
            let next = expand(
                &mut ctx,
                &parent.outcomes[0].state,
                &parent.path,
                difficulty,
                false,
                0,
            )?;
            for node in &next {
                if by_root
                    .get(&node.root)
                    .is_none_or(|old| node.score > old.score + 0.01)
                {
                    by_root.insert(node.root.clone(), node.clone());
                }
            }
            children.extend(next);
        }
        beam = diverse(&ctx, &children, cfg.width, &mut seen);
    }
    let mut choices: Vec<_> = by_root.into_values().collect();
    choices.sort_by(|a, b| b.score.total_cmp(&a.score));
    let end_can_win = choices
        .iter()
        .take(if difficulty == Difficulty::Hard {
            cfg.reply_roots
        } else if difficulty == Difficulty::Easy {
            3
        } else {
            1
        })
        .any(|n| n.path[0].command.kind == "end");
    ctx.stop_at = if end_can_win { search_max } else { max };
    let mut reply_candidates = 0;
    let mut reply_samples = 0;
    if difficulty == Difficulty::Hard
        && !synthesis
        && s.phase != "summon"
        && choices.len() > 1
        && !ctx.stopped()
    {
        let shortlist: Vec<_> = choices.iter().take(cfg.reply_roots).cloned().collect();
        let mut sums = vec![0.; shortlist.len()];
        for sample in 0..2 {
            if ctx.stopped() {
                break;
            }
            let mut round = Vec::new();
            for node in &shortlist {
                if ctx.stopped() {
                    break;
                }
                let mut draw = (f64::from(chance::hash(&format!(
                    "{}:reply:{sample}",
                    ctx.keys.fingerprint(&s)
                ))) + 0.5)
                    / 4294967296.;
                let mut state = node.outcomes.last().unwrap().state.fork();
                for out in &node.outcomes {
                    draw -= out.weight;
                    if draw <= 0. {
                        state = out.state.fork();
                        break;
                    }
                }
                if node.outcomes.len() > 1 {
                    ctx.sampled += 1;
                }
                if !terminal(&state) && state.active == side {
                    state = rollout(&mut ctx, &state, side, sample)?;
                }
                if !terminal(&state) && state.active != side {
                    state = rollout(&mut ctx, &state, state.active, sample)?;
                }
                if terminal(&state)
                    || (state.active == side && state.ply >= s.ply + 2. && state.pending.is_empty())
                {
                    ctx.replies += 1;
                    round.push(ctx.value(&state));
                } else {
                    break;
                }
            }
            if round.len() != shortlist.len() {
                break;
            }
            reply_samples += 1;
            for (i, score) in round.iter().enumerate() {
                sums[i] += score;
            }
            choices = shortlist
                .iter()
                .enumerate()
                .map(|(i, n)| Node {
                    score: sums[i] / reply_samples as f64,
                    ..n.clone()
                })
                .collect();
            choices.sort_by(|a, b| b.score.total_cmp(&a.score));
            reply_candidates = shortlist.len();
        }
    }
    let mut selected = choices.first().cloned();
    if difficulty == Difficulty::Easy
        && !synthesis
        && s.phase != "summon"
        && selected.as_ref().is_some_and(|n| n.score.abs() < 90000.)
    {
        let near: Vec<_> = choices
            .iter()
            .filter(|n| selected.as_ref().unwrap().score - n.score <= cfg.noise)
            .take(3)
            .collect();
        selected =
            Some(near[chance::hash(&ctx.keys.fingerprint(&s)) as usize % near.len()].clone());
    }
    let exhausted = ctx.stopped();
    let node_limit = ctx.count >= ctx.stop_at;
    ctx.stop_at = max;
    let mut checks = 0;
    let mut improved = false;
    if selected
        .as_ref()
        .is_some_and(|n| n.path[0].command.kind == "end")
        && reserve > 0
        && !ctx.stopped()
    {
        let (audit, count) = reconsider(&mut ctx, &s, &roots)?;
        checks = count;
        if !audit.is_empty() {
            selected = Some(audit[0].clone());
            improved = true;
            reply_candidates = 0;
            reply_samples = 0;
        }
    }
    let plan = selected.map(|n| n.path).unwrap_or_default();
    Ok(
        json!({"command":plan.first().map(|p|keys::command_value(&p.command)),"plan":plan.iter().map(|p|json!({"before":p.before,"command":keys::command_value(&p.command)})).collect::<Vec<_>>(),
        "stats":{"simulations":ctx.count,"candidates":ctx.candidates,"depth":ctx.depth,"replies":ctx.replies,"sampled":ctx.sampled,"exhausted":exhausted||ctx.stopped(),"mode":"work","stopReason":if node_limit||ctx.count>=max{"nodes"}else{"complete"},"replyCandidates":reply_candidates,"replySamples":reply_samples,"selectedDepth":plan.len(),"endTurnChecks":checks,"endTurnImproved":improved}}),
    )
}
/// 与 TS budget + plan-cache 对照；跨命令保留已用工作量和确定性计划。
pub fn teacher_next(
    observation: &Value,
    difficulty: Difficulty,
    record: &mut Value,
    catalog: &Catalog,
) -> Result<Value, String> {
    let state = crate::options::position(observation)?;
    let side = runtime::viewer(&state);
    let key = side.to_string();
    if record.is_null() {
        *record = json!({"budgets":{"1":{"ply":-1,"nodes":0,"commands":0},"2":{"ply":-1,"nodes":0,"commands":0}},"plans":{"1":[],"2":[]}});
    }
    if record["budgets"][&key]["ply"] != state.ply {
        record["budgets"][&key] = json!({"ply":state.ply,"nodes":0,"commands":0});
        record["plans"][&key] = json!([]);
    }
    let plan = record["plans"][&key]
        .as_array()
        .ok_or("invalid teacher plans")?;
    let cached = plan.first().filter(|p| {
        p["command"]["type"] != "end" && p["before"] == keys::public_fingerprint(observation)
    });
    let decision = if let Some(next) = cached {
        json!({"command":next["command"],"plan":plan,"stats":{"simulations":0,"candidates":0,"depth":0,"replies":0,"sampled":0,"exhausted":false,"cached":true}})
    } else {
        let active = state
            .units
            .iter()
            .filter(|u| {
                let st = crate::stats::stats(&state, u, catalog);
                u.owner == side
                    && !u.any(&["grave", "wall"])
                    && !st.sleeping
                    && !st.frozen
                    && !st.stunned
                    && (st.operations_left > 0. || u.mode == "attack" || u.mode == "move")
            })
            .count();
        let settings = difficulty.settings();
        let remaining =
            settings.turn_nodes as f64 - record["budgets"][&key]["nodes"].as_f64().unwrap_or(0.);
        let minimum = match difficulty {
            Difficulty::Hard => 240.,
            Difficulty::Medium => 120.,
            Difficulty::Easy => 60.,
        };
        let budget = (remaining / active.div_ceil(3).max(1) as f64)
            .min(settings.nodes as f64)
            .max(minimum)
            .floor() as usize;
        decide(observation, side, difficulty, Some(budget), catalog)?
    };
    if decision["command"].is_null() {
        return Err("hand AI has no command".into());
    }
    record["plans"][&key] = json!(&decision["plan"].as_array().unwrap()[1..]);
    let nodes = record["budgets"][&key]["nodes"].as_u64().unwrap_or(0)
        + decision["stats"]["simulations"].as_u64().unwrap_or(0);
    let commands = record["budgets"][&key]["commands"].as_u64().unwrap_or(0) + 1;
    record["budgets"][&key]["nodes"] = json!(nodes);
    record["budgets"][&key]["commands"] = json!(commands);
    Ok(decision)
}
