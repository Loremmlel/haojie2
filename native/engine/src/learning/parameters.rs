use crate::{
    geometry,
    inspection::Queries,
    model::{Catalog, Command, Point, State},
    options::{self, Action, Step},
};
use std::borrow::Cow;
use std::rc::Rc;

/// 枚举一个参数的规则域；最后参数执行完整预检，随机边界仍保留。
pub(super) fn expand(
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
