use super::spatial::stats_for;
use super::{
    evaluation as eval, hp, passive,
    ranking::{self, ghost},
    settings::Difficulty,
    spatial::hit_distance,
    threats,
};
use crate::{
    geometry::{self, Target},
    inspection::Queries,
    model::{Catalog, Command, Point, State, Unit},
    options::{self, Action, Step},
};
use std::rc::Rc;

#[derive(Clone)]
pub struct Group {
    pub family: String,
    pub commands: Vec<Command>,
    pub keep: usize,
    pub priority: Option<f64>,
    deferred: Option<(Action, usize)>,
}
impl Group {
    pub fn commands(&mut self, s: &State, c: &Catalog) -> &[Command] {
        if let Some((a, p)) = self.deferred.take() {
            self.commands = expand(s, &a, p, None, None, c);
        }
        &self.commands
    }
}
fn group(
    family: impl Into<String>,
    commands: Vec<Command>,
    keep: usize,
    priority: Option<f64>,
) -> Group {
    Group {
        family: family.into(),
        commands,
        keep,
        priority,
        deferred: None,
    }
}
fn summon_choices(s: &State, cmd: Command, pool: &str, c: &Catalog) -> Vec<Command> {
    let available = options::array(
        &s.extra.get("auras").unwrap_or(&serde_json::Value::Null)[s.active.to_string()],
    )
    .iter()
    .any(|a| a["kind"] == "laoqian" && a["usedPly"] != s.ply);
    if !available {
        return vec![cmd];
    }
    let score = |k: &crate::model::Kind| {
        let d = c.by_kind(k);
        d.health.max(0.) * 0.45
            + d.attack.max(0.) * d.actions.max(1.) * 0.7
            + d.range.min(8.) * 3.
            + if k.is("u25") {
                115.
            } else if k.is("u12") {
                150.
            } else if k.is("u3") {
                40.
            } else {
                0.
            }
    };
    let mut kinds = c.pools[pool].clone();
    kinds.sort_by(|a, b| score(b).total_cmp(&score(a)));
    let mut result = vec![cmd.clone()];
    result.extend(kinds.into_iter().map(|k| {
        let mut next = cmd.clone();
        next.chosen_kind = Some(k);
        next
    }));
    result
}
fn actor<'a>(s: &'a State, cmd: &Command) -> Option<&'a Unit> {
    cmd.unit_id
        .as_deref()
        .and_then(|id| s.unit(id))
        .or_else(|| {
            if cmd.kind == "react" {
                s.pending.first().map(|r| &r.source)
            } else {
                None
            }
        })
}
pub fn points(
    s: &State,
    a: &Action,
    cmd: &Command,
    rows: Option<&[usize]>,
    rank: bool,
    c: &Catalog,
) -> Vec<Point> {
    super::spatial::with_cache(s, || points_inner(s, a, cmd, rows, rank, c))
}
fn points_inner(
    s: &State,
    a: &Action,
    cmd: &Command,
    rows: Option<&[usize]>,
    rank: bool,
    c: &Catalog,
) -> Vec<Point> {
    let u = actor(s, cmd);
    let id = a.id.split(':').next().unwrap();
    if (cmd.kind == "move" && u.is_some_and(|u| u.runner() || u.size > 1.)) || id == "bounce" {
        return u
            .map(|u| geometry::neighbors(u.at()).collect())
            .unwrap_or_default();
    }
    if id == "giant" {
        return cmd
            .target_id
            .as_deref()
            .and_then(|id| s.unit(id))
            .map(|u| geometry::expansion_anchors(s, u, c))
            .unwrap_or_default();
    }
    let card = if cmd.kind == "deploy" {
        crate::preparation::card(s, cmd)
    } else {
        None
    };
    let raw = card.as_ref().map(|v| {
        ghost(
            s,
            &crate::preparation::kind(&v["kind"]).key(),
            s.active,
            s.turn(s.active),
            Point { x: 1., y: 1. },
            c,
        )
    });
    let st = u.map(|u| stats_for(s, u, c));
    let limit = st.as_ref().map_or(0., |st| {
        if st.movement.fract() != 0. {
            st.movement * 2.
        } else {
            st.movement
        }
    });
    let straight = u.is_some_and(|u| u.has("13") && !u.silenced);
    let mut all: Vec<_> = geometry::all_cells()
        .filter(|p| {
            if rows.is_some_and(|r| !r.contains(&(p.y as usize))) {
                return false;
            }
            if id == "hut-spawn" {
                let Some(r) = s.pending.first() else {
                    return false;
                };
                let Some(hut) = s.unit(&r.source.id) else {
                    return false;
                };
                let unit = ghost(s, "20", r.owner, s.turn(r.owner), Point { x: 1., y: 1. }, c);
                return passive(s, hut)
                    && hut.owner == r.owner
                    && hut.max_hp >= 10.
                    && geometry::can_place(s, &unit, *p, c)
                    && geometry::attack_path(
                        s,
                        hut,
                        &geometry::point_target(*p),
                        stats_for(s, hut, c).range,
                        None,
                        false,
                    )
                    .is_some();
            }
            if cmd.kind == "deploy" {
                return raw
                    .as_ref()
                    .is_some_and(|u| geometry::can_deploy(s, u, *p, c));
            }
            if cmd.kind == "move" {
                return u.is_some_and(|u| {
                    geometry::distance(u.at(), *p) <= limit
                        && geometry::movement_path(s, u, *p, limit, straight, c).is_some()
                });
            }
            if id == "dash" {
                return u.is_some_and(|u| {
                    geometry::distance(u.at(), *p) <= 6.
                        && geometry::movement_path(s, u, *p, 6., false, c).is_some()
                });
            }
            if id == "blast" {
                return p.x < 9. && p.y < 13.;
            }
            if ["cross", "ice-mark"].contains(&id) {
                return u.is_some_and(|u| (p.x - u.x).abs() <= 5. && (p.y - u.y).abs() <= 5.);
            }
            if id == "hook" {
                return u.is_some_and(|u| {
                    cmd.target_id
                        .as_deref()
                        .and_then(|id| s.unit(id))
                        .is_some_and(|v| {
                            let mut moved = v.fork();
                            moved.x = p.x;
                            moved.y = p.y;
                            v.at() != *p
                                && geometry::can_place(s, v, *p, c)
                                && geometry::attack_path(
                                    s,
                                    u,
                                    &Target::from(&moved),
                                    st.as_ref().unwrap().range,
                                    None,
                                    false,
                                )
                                .is_some()
                        })
                });
            }
            if id == "wall" {
                return u.is_some_and(|u| {
                    geometry::can_place(s, &ghost(s, "wall", u.owner, 0., *p, c), *p, c)
                        && geometry::attack_path(
                            s,
                            u,
                            &geometry::point_target(*p),
                            st.as_ref().unwrap().range,
                            None,
                            false,
                        )
                        .is_some()
                });
            }
            if id == "revive" {
                return u.is_some_and(|u| {
                    geometry::distance(*p, u.at()) <= st.as_ref().unwrap().range + 1.
                });
            }
            true
        })
        .collect();
    if rank {
        let scores: Vec<_> = all
            .iter()
            .map(|p| ranking::point_rank(s, a, cmd, *p, u, c))
            .collect();
        let mut order: Vec<_> = (0..all.len()).collect();
        order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
        all = order.into_iter().map(|i| all[i]).collect();
    }
    all
}
fn choices(
    s: &State,
    a: &Action,
    cmd: &Command,
    step: Step,
    rows: Option<&[usize]>,
    target_id: Option<&str>,
    c: &Catalog,
) -> Vec<Command> {
    let id = a.id.split(':').next().unwrap();
    let owner = crate::runtime::viewer(s);
    if step.kind == "path" {
        return vec![];
    }
    if step.kind == "target" {
        let u = actor(s, cmd);
        let moved = u.map(|u| {
            let mut v = u.fork();
            if id == "dash"
                && let Some(x) = cmd.x
            {
                v.x = x;
                v.y = cmd.y.unwrap();
            }
            v
        });
        let view = moved.as_ref();
        let range = cmd.kind != "cast"
            && cmd.kind != "equip"
            && cmd.kind != "react"
            && id != "giant"
            && id != "superhook";
        let mut targets: Vec<_> = geometry::targets(s)
            .into_iter()
            .filter(|t| {
                if target_id.is_some_and(|id| t.id != id) || step.unit_only && t.unit.is_none() {
                    return false;
                }
                let side = t.unit.as_ref().map_or(t.owner, |u| u.side());
                if step.relation == "friend" && side != owner
                    || step.relation == "enemy" && side == owner
                {
                    return false;
                }
                if cmd
                    .sacrifice_ids
                    .as_ref()
                    .is_some_and(|ids| ids.contains(&t.id))
                    || cmd.target_id.as_deref() == Some(&t.id)
                {
                    return false;
                }
                if step.field == "sacrificeIds"
                    && t.unit
                        .as_ref()
                        .is_none_or(|u| u.kind.is("u25") || u.hp * 2. < u.max_hp)
                {
                    return false;
                }
                if id == "giant"
                    && (t.unit.is_none()
                        || u.is_some_and(|u| u.id == t.id)
                        || geometry::expansion_anchors(s, t.unit.as_ref().unwrap(), c).is_empty())
                {
                    return false;
                }
                if id == "sacrifice-summon"
                    && (u.is_some_and(|u| u.id == t.id)
                        || t.unit.as_ref().is_none_or(|u| !u.kind.is("14")))
                {
                    return false;
                }
                if id == "sacrifice"
                    && (u.is_some_and(|u| u.id == t.id)
                        || t.unit.as_ref().is_some_and(|u| u.kind.is("u25")))
                {
                    return false;
                }
                if cmd.kind == "attack" || cmd.kind == "react" {
                    if !geometry::top_target(s, t, c)
                        && !view.is_some_and(|u| {
                            crate::combat::healing_attack(u, c)
                                && t.id == u.id
                                && cmd.mode.as_deref() != Some("damage")
                        })
                    {
                        return false;
                    }
                    if view.is_some_and(|u| {
                        side == owner
                            && t.unit.as_ref().is_some_and(|v| v.hp >= v.max_hp)
                            && !u.silenced
                            && crate::combat::healing_attack(u, c)
                            && cmd.mode.as_deref() != Some("damage")
                    }) {
                        return false;
                    }
                    if let Some(u) = view {
                        let st = stats_for(s, u, c);
                        if u.kind.is("10")
                            && passive(s, u)
                            && t.unit.as_ref().is_some_and(|v| {
                                v.kind.is("u18") && passive(s, v) && v.hp == v.max_hp
                            })
                            && st.attack <= 10.
                            && !s.effect(u, "execute")
                            && !s.effect(u, "convert")
                            && !u.piercing()
                        {
                            return false;
                        }
                        if side != owner
                            && st.attack == 0.
                            && !(passive(s, u) && u.any(&["10", "u6"]))
                            && !s.effect(u, "execute")
                            && !s.effect(u, "convert")
                            && !u.weapon("u5")
                            && !u.piercing()
                            && !t.unit.as_ref().is_some_and(|v| {
                                v.effects.iter().any(|e| {
                                    e.kind == "mark" && e.owner == owner && e.until > s.ply
                                })
                            })
                        {
                            return false;
                        }
                        if u.kind.is("10")
                            && !u.silenced
                            && side != owner
                            && st.attack == 0.
                            && !s.effect(u, "execute")
                            && !s.effect(u, "convert")
                            && !t
                                .unit
                                .as_ref()
                                .map_or(hp(s, t.owner) == 300., |v| v.hp == v.max_hp)
                            && threats::mark_follow(s, t, owner, s.ply + 2., c) == 0.
                        {
                            return false;
                        }
                    }
                    if side == owner
                        && !view.is_some_and(|u| {
                            t.unit.as_ref().is_some_and(|v| {
                                c.by_kind(&u.kind).attack < 0.
                                    || u.signed()
                                    || (!u.silenced && u.any(&["2", "u21", "sage"]))
                                    || (!v.silenced && v.has("16"))
                                    || cmd.mode.as_deref() == Some("heal")
                                        && crate::combat::healing_attack(u, c)
                            })
                        })
                    {
                        return false;
                    }
                }
                if range
                    && view.is_some_and(|u| {
                        if u.piercing() {
                            !hit_distance(s, u, t, "", c).is_finite()
                        } else {
                            geometry::attack_path(s, u, t, stats_for(s, u, c).range, None, false)
                                .is_none()
                        }
                    })
                {
                    return false;
                }
                true
            })
            .map(|t| {
                let score = ranking::target_rank(s, a, cmd, &t, c);
                (t, score)
            })
            .collect();
        targets.sort_by(|a, b| b.1.total_cmp(&a.1));
        return targets
            .into_iter()
            .map(|(t, _)| {
                let mut next = cmd.clone();
                match step.field {
                    "sacrificeIds" => next.sacrifice_ids.get_or_insert_with(Vec::new).push(t.id),
                    "secondId" => next.second_id = Some(t.id),
                    _ => next.target_id = Some(Rc::from(t.id.as_str())),
                };
                super::keys::append(&mut next, &[step.field]);
                next
            })
            .collect();
    }
    if step.kind == "point" {
        return points(s, a, cmd, rows, true, c)
            .into_iter()
            .map(|p| {
                let mut next = Command {
                    x: Some(p.x),
                    y: Some(p.y),
                    ..cmd.clone()
                };
                super::keys::append(&mut next, &["x", "y"]);
                next
            })
            .collect();
    }
    if step.kind == "death" {
        let mut records: Vec<_> = s
            .deaths
            .iter()
            .filter(|d| {
                d["kind"] != "grave"
                    && d["owner"] == s.active
                    && d["revived"] != true
                    && d["ply"]
                        .as_f64()
                        .is_some_and(|p| s.ply - p <= 4. && s.ply > p)
            })
            .collect();
        records.sort_by(|a, b| {
            c.by_kind(&crate::preparation::kind(&b["kind"]))
                .attack
                .total_cmp(&c.by_kind(&crate::preparation::kind(&a["kind"])).attack)
        });
        return records
            .into_iter()
            .map(|d| {
                let mut next = cmd.clone();
                next.death_id = Some(d["id"].as_str().unwrap().into());
                super::keys::append(&mut next, &["deathId"]);
                next
            })
            .collect();
    }
    let mut lines: Vec<_> = (1..=if step.kind == "row" { 13 } else { 9 })
        .map(|line| {
            let mut next = cmd.clone();
            if step.kind == "row" {
                next.row = Some(line as f64);
            } else {
                next.column = Some(line as f64);
            }
            super::keys::append(&mut next, &[step.kind]);
            let score = geometry::targets(s)
                .iter()
                .map(|t| {
                    let hits = t
                        .footprint()
                        .iter()
                        .filter(|p| {
                            if step.kind == "row" {
                                p.y == line as f64
                            } else {
                                p.x == line as f64
                            }
                        })
                        .count() as f64;
                    if hits == 0. || (t.owner == owner && cmd.kind == "cast") {
                        return 0.;
                    }
                    (if t.owner == owner { -1. } else { 1. })
                        * t.unit.as_ref().map_or(25., |u| {
                            (20. * hits).min(u.hp)
                                + if u.hp <= 20. * hits {
                                    eval::material(s, u, c)
                                } else {
                                    0.
                                }
                        })
                })
                .sum::<f64>();
            (next, score)
        })
        .collect();
    lines.sort_by(|a, b| b.1.total_cmp(&a.1));
    lines.into_iter().map(|v| v.0).collect()
}
pub fn expand(
    s: &State,
    a: &Action,
    partial: usize,
    rows: Option<&[usize]>,
    target: Option<&str>,
    c: &Catalog,
) -> Vec<Command> {
    let mut base = a.command.clone();
    super::keys::tag(&mut base);
    let mut drafts = vec![base];
    for (i, step) in a.steps.iter().enumerate() {
        let last = i + 1 == a.steps.len();
        drafts = drafts
            .iter()
            .flat_map(|cmd| {
                let mut next = choices(
                    s,
                    a,
                    cmd,
                    *step,
                    if last { rows } else { None },
                    if last { target } else { None },
                    c,
                );
                if !last {
                    next.truncate(partial);
                }
                next
            })
            .collect();
    }
    if a.command.kind == "attack" {
        drafts = drafts
            .into_iter()
            .flat_map(|cmd| {
                let routes = cmd
                    .unit_id
                    .as_deref()
                    .and_then(|id| s.unit(id))
                    .and_then(|u| {
                        cmd.target_id
                            .as_deref()
                            .and_then(|id| geometry::find_target(s, id))
                            .map(|t| {
                                if u.piercing() && !u.weapon("u28")
                                    || !(u.piercing()
                                        || u.has("u20") && !u.silenced
                                        || t.unit
                                            .as_ref()
                                            .is_some_and(|v| v.has("24") && !v.silenced))
                                {
                                    return vec![];
                                }
                                geometry::AttackQuery::default().directions(
                                    s,
                                    u,
                                    &t,
                                    stats_for(s, u, c).range,
                                    u.piercing(),
                                )
                            })
                    })
                    .unwrap_or_default();
                if routes.len() > 1 {
                    routes
                        .into_iter()
                        .map(|d| {
                            let mut next = Command {
                                direction: Some(std::borrow::Cow::Borrowed(d)),
                                ..cmd.clone()
                            };
                            super::keys::append(&mut next, &["direction"]);
                            next
                        })
                        .collect()
                } else {
                    vec![cmd]
                }
            })
            .collect();
    }
    drafts
}
fn make(
    s: &State,
    mut a: Action,
    d: Difficulty,
    defer: bool,
    c: &Catalog,
) -> Result<Option<Group>, String> {
    if !options::allowed(
        s,
        crate::runtime::viewer(s),
        &mut a,
        c,
        &mut Queries::default(),
    )? {
        return Ok(None);
    }
    let family = format!(
        "{}:{}",
        a.command
            .unit_id
            .as_deref()
            .or(a.command.card_id.as_deref())
            .unwrap_or("system"),
        a.id
    );
    let settings = d.settings();
    if defer
        && a.steps.len() == 1
        && a.steps[0].kind == "point"
        && ["deploy", "skill", "cast"].contains(&a.command.kind.as_str())
    {
        if points(s, &a, &a.command, None, false, c).is_empty() {
            return Ok(None);
        }
        let mut g = group(
            family,
            vec![],
            settings.per_action,
            Some(ranking::command_priority(s, &a.command, c)),
        );
        g.deferred = Some((a, settings.partial));
        return Ok(Some(g));
    }
    let commands: Vec<_> = expand(s, &a, settings.partial, None, None, c)
        .into_iter()
        .flat_map(|cmd| {
            if let Some(pool) = options::pool(s, &cmd, c) {
                summon_choices(s, cmd, pool, c)
            } else {
                vec![cmd]
            }
        })
        .collect();
    if commands.is_empty() {
        return Ok(None);
    }
    let priority = ranking::command_priority(s, &commands[0], c);
    Ok(Some(group(
        family,
        commands,
        if a.command.kind == "attack" {
            settings.per_action.max(8)
        } else {
            settings.per_action
        },
        Some(priority),
    )))
}
pub fn groups(s: &State, d: Difficulty, defer: bool, c: &Catalog) -> Result<Vec<Group>, String> {
    super::spatial::with_cache(s, || groups_inner(s, d, defer, c))
}
fn groups_inner(s: &State, d: Difficulty, defer: bool, c: &Catalog) -> Result<Vec<Group>, String> {
    if s.extra.contains_key("winner") {
        return Ok(vec![]);
    }
    let mut result = Vec::new();
    let settings = d.settings();
    let owner = crate::runtime::viewer(s);
    if !s.pending.is_empty() {
        let actions = options::actions(s, owner, c, &mut Queries::default())?;
        if let Some(a) = actions.first()
            && let Some(g) = make(s, a.clone(), d, defer, c)?
        {
            result.push(g);
        }
        let r = &s.pending[0];
        let skip = r.kind != "bounce"
            && !(r.kind == "hut-spawn"
                && r.source.has("citadel")
                && actions
                    .first()
                    .is_some_and(|a| !points(s, a, &a.command, None, false, c).is_empty()));
        if skip {
            result.push(group("reaction-skip", vec![Command::new("react")], 1, None));
        }
        return Ok(result);
    }
    if s.phase == "synthesis" {
        result.push(group(
            "synthesis-skip",
            vec![Command::new("skip-synthesis")],
            1,
            Some(1000.),
        ));
        for recipe in &c.recipes {
            let mut ids = crate::synthesis::materials(s, recipe);
            if ids.len() < 3 {
                continue;
            }
            let cost = |id: &String| {
                s.unit(id)
                    .map_or(0., |u| eval::unit(s, u, &|v| eval::material(s, v, c), c))
            };
            ids.sort_by(|a, b| cost(a).total_cmp(&cost(b)));
            let mut sets = indexmap::IndexMap::new();
            for id in &ids {
                let mut trio = vec![id.clone()];
                trio.extend(ids.iter().filter(|v| *v != id).take(2).cloned());
                let mut key = trio.clone();
                key.sort();
                sets.insert(key.join("|"), trio);
            }
            let mut accepted = 0;
            for material in sets.values() {
                if c.by_kind(&crate::preparation::kind(&recipe["result"])).aura {
                    let mut cmd = Command::new("synthesize");
                    cmd.recipe_id = Some(recipe["id"].as_str().unwrap().into());
                    cmd.material_ids = Some(material.clone());
                    result.push(group(
                        format!("synthesis:{}", recipe["id"].as_str().unwrap()),
                        vec![cmd],
                        1,
                        Some(80.),
                    ));
                    break;
                }
                let mut destinations = crate::synthesis::destinations(s, recipe, material, c);
                if destinations.is_empty() {
                    continue;
                }
                let mut view = s.fork();
                if recipe["source"] == "board" {
                    view.units.retain(|u| !material.contains(&u.id));
                }
                let raw = ghost(
                    s,
                    &crate::preparation::kind(&recipe["result"]).key(),
                    s.active,
                    s.turn(s.active),
                    destinations[0],
                    c,
                );
                let mut ranked: Vec<_> = destinations
                    .drain(..)
                    .map(|p| (p, eval::placement(&view, &raw, p, false, None, c)))
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                let recipe_id = recipe["id"].as_str().unwrap();
                let commands = ranked
                    .into_iter()
                    .map(|(p, _)| {
                        let mut cmd = Command::new("synthesize");
                        cmd.recipe_id = Some(recipe_id.into());
                        cmd.material_ids = Some(material.clone());
                        cmd.x = Some(p.x);
                        cmd.y = Some(p.y);
                        cmd
                    })
                    .collect();
                result.push(group(
                    format!("synthesis:{recipe_id}:{accepted}"),
                    commands,
                    settings.per_action,
                    Some(80.),
                ));
                accepted += 1;
                if accepted >= if d == Difficulty::Easy { 1 } else { 3 } {
                    break;
                }
            }
        }
        return Ok(result);
    }
    let summons = || {
        let mut normal = Command::new("summon");
        normal.ultimate = Some(false);
        let mut list = summon_choices(s, normal, "normal", c);
        if s.extra["heads"][s.active.to_string()]
            .as_f64()
            .unwrap_or(0.)
            >= 2.
        {
            let mut ultimate = Command::new("summon");
            ultimate.ultimate = Some(true);
            list.extend(summon_choices(s, ultimate, "ultimate", c));
        }
        list
    };
    if s.phase == "summon" {
        result.push(group(
            "summon",
            if s.summon_slots > 0. {
                summons()
            } else {
                vec![Command::new("begin")]
            },
            2,
            None,
        ));
        for card in crate::preparation::hand(s) {
            for a in options::card_actions(s, card, c) {
                if a.command.kind == "reroll"
                    && let Some(g) = make(s, a, d, defer, c)?
                {
                    result.push(g);
                }
            }
        }
        return Ok(result);
    }
    if let Some(transit) = s.units.iter().find(|u| {
        u.owner == s.active
            && u.has("u12p")
            && !u.has("u12")
            && u.mode == "move"
            && !geometry::can_place(s, u, u.at(), c)
    }) {
        for a in options::unit_actions(s, transit, c, &stats_for(s, transit, c)) {
            if a.id == "move"
                && let Some(g) = make(s, a, d, defer, c)?
            {
                result.push(g);
            }
        }
        return Ok(result);
    }
    if s.summon_slots > 0. {
        result.push(group("bonus-summon", summons(), 2, Some(1000.)));
    }
    let mut actions: Vec<_> = s
        .units
        .iter()
        .filter(|u| u.owner == s.active)
        .flat_map(|u| options::unit_actions(s, u, c, &stats_for(s, u, c)))
        .collect();
    actions.sort_by_key(|a| a.command.kind != "attack");
    for a in actions {
        if let Some(g) = make(s, a, d, defer, c)? {
            result.push(g);
        }
    }
    for card in crate::preparation::hand(s) {
        for a in options::card_actions(s, card, c) {
            if let Some(g) = make(s, a, d, defer, c)? {
                result.push(g);
            }
        }
    }
    result.push(group("end", vec![Command::new("end")], 1, None));
    Ok(result)
}
pub fn attack_candidates(s: &State, id: &str, target: Option<&str>, c: &Catalog) -> Vec<Command> {
    if !s.pending.is_empty() {
        return vec![];
    }
    let Some(u) = s.unit(id) else {
        return vec![];
    };
    let Some(mut a) = options::unit_actions(s, u, c, &stats_for(s, u, c))
        .into_iter()
        .find(|a| a.command.kind == "attack")
    else {
        return vec![];
    };
    if options::allowed(
        s,
        crate::runtime::viewer(s),
        &mut a,
        c,
        &mut Queries::default(),
    ) != Ok(true)
    {
        return vec![];
    }
    let filter =
        if a.steps.len() == 1 && a.steps[0].kind == "target" && a.steps[0].field == "targetId" {
            target
        } else {
            None
        };
    expand(s, &a, 1, None, filter, c)
        .into_iter()
        .filter(|cmd| target.is_none_or(|id| cmd.target_id.as_deref() == Some(id)))
        .collect()
}
pub fn deployment_candidates(s: &State, rows: &[usize], c: &Catalog) -> Vec<Command> {
    crate::preparation::hand(s)
        .iter()
        .flat_map(|card| {
            options::card_actions(s, card, c)
                .into_iter()
                .filter(|a| a.command.kind == "deploy")
                .flat_map(|a| {
                    let filter = if a.steps.len() == 1 && a.steps[0].kind == "point" {
                        Some(rows)
                    } else {
                        None
                    };
                    expand(s, &a, 1, filter, None, c)
                        .into_iter()
                        .filter(|cmd| cmd.y.is_some_and(|y| rows.contains(&(y as usize))))
                        .take(2)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
