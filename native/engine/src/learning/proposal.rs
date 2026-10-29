use crate::{
    geometry,
    inspection::Queries,
    model::{Catalog, Command, Point, State},
    options::{self, Action, Step},
};
use std::borrow::Cow;
use std::collections::HashSet;
use std::rc::Rc;

pub fn random(seed: &mut u32) -> u32 {
    if *seed == 0 {
        *seed = 0x9e3779b9;
    }
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

/// 枚举一个参数的规则域；最后参数执行完整预检，随机边界仍保留。
fn expand(
    s: &State,
    c: &Command,
    action: &Action,
    step: Step,
    last: bool,
    catalog: &Catalog,
    queries: &mut Queries,
) -> Result<Vec<(Command, bool)>, String> {
    let actor = crate::runtime::viewer(s);
    if step.kind == "point"
        && last
        && let Some(points) = options::points(s, actor, c, catalog, queries)?
    {
        return Ok(points
            .into_iter()
            .map(|(at, _)| {
                (
                    Command {
                        x: Some(at.x),
                        y: Some(at.y),
                        ..c.clone()
                    },
                    false,
                )
            })
            .collect());
    }
    let mut values = Vec::new();
    match step.kind {
        "chosen" => {
            values.push((c.clone(), false));
            for kind in &action.chosen {
                let mut next = c.clone();
                next.chosen_kind = Some(kind.clone());
                values.push((next, false));
            }
        }
        "material" => {
            let selected = c.material_ids.as_deref().unwrap_or(&[]);
            // 合成材料是集合；只生成原数组顺序下递增的组合。
            let after = selected
                .last()
                .and_then(|id| action.materials.iter().position(|v| v == id))
                .map_or(0, |i| i + 1);
            for id in &action.materials[after..] {
                let mut next = c.clone();
                next.material_ids
                    .get_or_insert_with(Vec::new)
                    .push(id.clone());
                values.push((next, false));
            }
        }
        "target" => {
            for t in geometry::targets(s) {
                if step.unit_only && t.unit.is_none() {
                    continue;
                }
                let side = t.unit.as_ref().map_or(t.owner, |u| u.side());
                if (step.relation == "friend" && side != actor)
                    || (step.relation == "enemy" && side == actor)
                {
                    continue;
                }
                let mut next = c.clone();
                match step.field {
                    "secondId" => next.second_id = Some(t.id),
                    "sacrificeIds" => {
                        if next.sacrifice_ids.as_deref().unwrap_or(&[]).contains(&t.id) {
                            continue;
                        }
                        next.sacrifice_ids.get_or_insert_with(Vec::new).push(t.id);
                    }
                    _ => next.target_id = Some(Rc::from(t.id.as_str())),
                }
                values.push((next, false));
            }
        }
        "point" => {
            let points: Vec<Point> = if c.kind == "synthesize" {
                let recipe = catalog
                    .recipes
                    .iter()
                    .find(|r| r["id"].as_str() == c.recipe_id.as_deref())
                    .ok_or("unknown recipe")?;
                crate::synthesis::destinations(
                    s,
                    recipe,
                    c.material_ids.as_deref().unwrap_or(&[]),
                    catalog,
                )
            } else if action.id == "giant" && c.target_id.is_some() {
                s.unit(c.target_id.as_deref().unwrap())
                    .map(|u| geometry::expansion_anchors(s, u, catalog))
                    .unwrap_or_default()
            } else {
                geometry::all_cells().collect()
            };
            for at in points {
                values.push((
                    Command {
                        x: Some(at.x),
                        y: Some(at.y),
                        ..c.clone()
                    },
                    false,
                ));
            }
        }
        "death" => {
            for record in s.deaths.iter() {
                let mut next = c.clone();
                next.death_id = record["id"].as_str().map(str::to_owned);
                values.push((next, false));
            }
        }
        "row" | "column" => {
            for value in 1..=if step.kind == "row" { 13 } else { 9 } {
                let mut next = c.clone();
                if step.kind == "row" {
                    next.row = Some(value as f64)
                } else {
                    next.column = Some(value as f64)
                };
                values.push((next, false));
            }
        }
        "direction" => {
            values.push((c.clone(), false));
            if let (Some(u), Some(target)) = (
                c.unit_id.as_deref().and_then(|id| s.unit(id)),
                c.target_id
                    .as_deref()
                    .and_then(|id| geometry::find_target(s, id)),
            ) {
                let range = queries.attributes(s, u, catalog).range;
                for direction in queries
                    .attacks
                    .directions(s, u, &target, range, u.piercing())
                {
                    let mut next = c.clone();
                    next.direction = Some(Cow::Borrowed(direction));
                    values.push((next, false));
                }
            }
        }
        "path" => {
            if let Some(u) = c.unit_id.as_deref().and_then(|id| s.unit(id)) {
                let path = c.path.as_deref().unwrap_or(&[]);
                if !path.is_empty() {
                    values.push((c.clone(), false));
                }
                let points: Vec<_> = path
                    .last()
                    .map(|p| geometry::neighbors(*p).collect())
                    .unwrap_or_else(|| geometry::cells(u, u.at()));
                let range = queries.attributes(s, u, catalog).range;
                for at in points {
                    let mut route = path.to_vec();
                    route.push(at);
                    if geometry::valid_attack_route(u, &route, range) {
                        let mut next = c.clone();
                        next.path = Some(route);
                        values.push((next, true));
                    }
                }
            }
        }
        _ => return Err(format!("unsupported classic parameter {}", step.kind)),
    }
    if last && step.kind != "path" {
        let mut valid = Vec::new();
        for value in values {
            if options::inspect(s, actor, &value.0, catalog, queries)? != "invalid" {
                valid.push(value);
            }
        }
        Ok(valid)
    } else {
        Ok(values)
    }
}

struct Walk<'a> {
    state: &'a State,
    catalog: &'a Catalog,
    queries: Queries,
    seed: u32,
    remaining: usize,
}
impl Walk<'_> {
    fn complete(
        &mut self,
        action: &Action,
        command: Command,
        steps: &[Step],
        offset: usize,
        depth: usize,
    ) -> Result<Option<Command>, String> {
        if self.remaining == 0 || depth > 256 {
            return Ok(None);
        }
        self.remaining -= 1;
        if offset == steps.len() {
            return Ok((options::inspect(
                self.state,
                crate::runtime::viewer(self.state),
                &command,
                self.catalog,
                &mut self.queries,
            )? != "invalid")
                .then_some(command));
        }
        let mut choices = expand(
            self.state,
            &command,
            action,
            steps[offset],
            offset + 1 == steps.len(),
            self.catalog,
            &mut self.queries,
        )?;
        // 无放回随机遍历，并回溯空分支；参数选择不进入 MCTS 深度。
        while !choices.is_empty() && self.remaining > 0 {
            let index = (random(&mut self.seed) as usize) % choices.len();
            let (next, repeat) = choices.swap_remove(index);
            if let Some(command) = self.complete(
                action,
                next,
                steps,
                offset + usize::from(!repeat),
                depth + 1,
            )? {
                return Ok(Some(command));
            }
        }
        Ok(None)
    }
}

/// 随机有限候选集是策略输入的一部分；不按手工评分裁剪完整合法域。
pub fn generate(
    state: &State,
    catalog: &Catalog,
    seed: u32,
    count: usize,
) -> Result<Vec<Command>, String> {
    if state.extra.contains_key("mode") {
        return Err("training supports classic only".into());
    }
    let actor = crate::runtime::viewer(state);
    let mut walk = Walk {
        state,
        catalog,
        queries: Queries::default(),
        seed,
        remaining: 20000,
    };
    let actions = options::actions(state, actor, catalog, &mut walk.queries)?;
    let simple = actions.iter().all(|a| {
        a.steps.is_empty()
            && a.materials.is_empty()
            && a.chosen.is_empty()
            && a.command.kind != "attack"
    });
    let mut commands = Vec::new();
    let mut seen = HashSet::new();
    let mut order: Vec<_> = (0..actions.len()).collect();
    for i in (1..order.len()).rev() {
        let j = random(&mut walk.seed) as usize % (i + 1);
        order.swap(i, j);
    }
    for attempt in 0..count.saturating_mul(8).max(actions.len()) {
        if simple && attempt >= actions.len() {
            break;
        }
        if commands.len() >= count || actions.is_empty() || walk.remaining == 0 {
            break;
        }
        let index = if attempt < order.len() {
            order[attempt]
        } else {
            random(&mut walk.seed) as usize % actions.len()
        };
        let action = &actions[index];
        let mut steps = Vec::new();
        if !action.materials.is_empty() {
            steps.extend(std::iter::repeat_n(Step::new("material"), 3));
        }
        steps.extend_from_slice(&action.steps);
        if !action.chosen.is_empty() {
            steps.push(Step::new("chosen"));
        }
        if action.command.kind == "attack" && action.command.path.is_none() {
            steps.push(Step::new("direction"));
        }
        if let Some(command) = walk.complete(action, action.command.clone(), &steps, 0, 0)? {
            let key = serde_json::to_string(&command).map_err(|e| e.to_string())?;
            if seen.insert(key) {
                commands.push(command);
            }
        }
    }
    if commands.is_empty() && !state.extra.contains_key("winner") {
        return Err(format!(
            "no complete command; stage={}, budget={}",
            state.phase, walk.remaining
        ));
    }
    Ok(commands)
}
