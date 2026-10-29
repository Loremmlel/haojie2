use super::spatial::stats_for;
use super::{evaluation as eval, hp, spatial::hit_distance, stored, threats};
use crate::{
    geometry::{self, Target},
    model::{Catalog, Command, Effect, Point, State, Unit},
    options::Action,
};

pub fn ghost(s: &State, kind: &str, owner: usize, born: f64, p: Point, c: &Catalog) -> Unit {
    let mut u = crate::resolution::template(s, kind, owner, p, "preview", c);
    u.born = born;
    u
}
fn lane_release(s: &State, blocker: &Unit, c: &Catalog) -> f64 {
    let owner = crate::runtime::viewer(s);
    let target = geometry::find_target(s, &format!("base-{}", 3 - owner)).unwrap();
    let mut value: f64 = 0.;
    for ally in &s.units {
        if ally.owner == owner
            && threats::ready(s, ally, c)
            && !hit_distance(s, ally, &target, "", c).is_finite()
            && hit_distance(s, ally, &target, &blocker.id, c).is_finite()
        {
            let damage = threats::pressure(s, ally, &target, &blocker.id, c);
            value = value.max(if damage >= hp(s, 3 - owner) {
                200.
            } else {
                (damage * 0.8).min(80.)
            });
        }
    }
    value
}
pub fn target_rank(s: &State, a: &Action, cmd: &Command, t: &Target, c: &Catalog) -> f64 {
    let owner = crate::runtime::viewer(s);
    let friend = t.unit.as_ref().map_or(t.owner, |u| u.side()) == owner;
    let caster = cmd
        .unit_id
        .as_deref()
        .and_then(|id| s.unit(id))
        .or_else(|| {
            if cmd.kind == "react" {
                s.pending.first().map(|r| &r.source)
            } else {
                None
            }
        });
    let card = crate::preparation::card(s, cmd);
    if caster.is_some_and(|u| u.kind.is("10") && !u.silenced) && !friend {
        let follow = threats::mark_follow(s, t, owner, s.ply + 2., c);
        let immediate = t
            .unit
            .as_ref()
            .map_or(hp(s, t.owner) == 300., |u| u.hp == u.max_hp);
        return if immediate { 5. } else { 0. }
            + follow
                * t.unit
                    .as_ref()
                    .map_or(15., |u| 12. + eval::material(s, u, c) * 0.25)
            + if immediate {
                t.unit
                    .as_ref()
                    .map_or(0., |u| eval::material(s, u, c) * 0.08)
            } else {
                0.
            };
    }
    if cmd.kind == "cast"
        && card
            .as_ref()
            .is_some_and(|v| v["kind"] == 18 || v["kind"] == 22)
        && let Some(carrier) = t.unit.as_ref()
    {
        let kind = if card.unwrap()["kind"] == 18 {
            "execute"
        } else {
            "convert"
        };
        let mut carrier = carrier.fork();
        carrier.effects.push(Effect {
            kind: kind.into(),
            owner,
            from: s.ply + 2.,
            until: s.ply + 3.,
            global: Some(true),
            amount: None,
            source_id: None,
        });
        let mut view = s.fork();
        for u in &mut view.units {
            if u.id == carrier.id {
                *u = carrier.fork();
            }
        }
        return threats::payload(&view, &carrier, kind, &|v| eval::material(&view, v, c), c);
    }
    let Some(u) = &t.unit else {
        return if friend {
            (300. - hp(s, t.owner)) * 0.6
        } else {
            80. + 300. - hp(s, t.owner)
        };
    };
    let value = eval::unit(s, u, &|v| eval::material(s, v, c), c);
    if cmd.kind == "attack" || cmd.kind == "react" {
        if cmd.mode.as_deref() == Some("heal") || (friend && cmd.mode.as_deref() != Some("damage"))
        {
            return (u.max_hp - u.hp) * 2.;
        }
        if friend {
            return -value;
        }
        let shots = caster.map_or(1., |u| stats_for(s, u, c).remaining);
        let packets = caster
            .map(|u| threats::hit_packets(s, u, t, None, c))
            .unwrap_or_default();
        let burst = if packets.len() == 1 && caster.is_none_or(|u| !u.any(&["9", "10"])) {
            shots
        } else {
            1.
        };
        let kill = packets
            .iter()
            .filter(|p| p.damage * burst >= u.hp)
            .map(|p| p.probability)
            .sum::<f64>();
        return value + kill * (value * 0.7 + 20.) + lane_release(s, u, c);
    }
    match a.id.split(':').next().unwrap() {
        "siphon" => {
            if cmd.target_id.is_some() {
                if friend {
                    (u.max_hp - u.hp) * 3. + 10.
                } else {
                    -value
                }
            } else if friend {
                -value
            } else {
                value + 30.
            }
        }
        "sacrifice-summon" => 30. - value,
        "reforge-two" => -value,
        "sacrifice" => -value + stats_for(s, u, c).attack,
        _ => {
            if friend {
                value
            } else {
                value + 30.
            }
        }
    }
}
pub fn command_priority(s: &State, cmd: &Command, c: &Catalog) -> f64 {
    let u = cmd.unit_id.as_deref().and_then(|id| s.unit(id));
    let t = cmd
        .target_id
        .as_deref()
        .and_then(|id| geometry::find_target(s, id));
    let owner = crate::runtime::viewer(s);
    match cmd.kind.as_str() {
        "end" | "finish-mode" => -0.1,
        "skip-synthesis" | "summon" | "begin" | "react" => 1000.,
        "deploy" => 80.,
        "attack" if u.is_some() && t.is_some() => {
            let u = u.unwrap();
            let t = t.unwrap();
            if t.owner == owner
                && let Some(target) = t.unit.as_ref()
            {
                return if cmd.mode.as_deref() == Some("heal")
                    || (cmd.mode.as_deref() != Some("damage")
                        && crate::combat::healing_attack(u, c))
                {
                    (target.max_hp - target.hp)
                        .min(c.by_kind(&u.kind).attack.abs())
                        .max(0.)
                        * 0.8
                } else {
                    -eval::material(s, target, c) * 0.8
                };
            }
            let hp = t.unit.as_ref().map_or(hp(s, t.owner), |u| u.hp);
            let packets = threats::hit_packets(s, u, &t, cmd.direction.as_deref(), c);
            let damage = packets
                .iter()
                .map(|p| p.probability * p.damage.min(hp))
                .sum::<f64>();
            let kill = packets
                .iter()
                .filter(|p| p.damage >= hp)
                .map(|p| p.probability)
                .sum::<f64>();
            damage * if t.unit.is_some() { 0.65 } else { 3.1 }
                + kill * t.unit.as_ref().map_or(100000., |v| eval::material(s, v, c))
                + if s.effect(u, "execute") || s.effect(u, "convert") {
                    100.
                } else {
                    0.
                }
                + if u.kind.is("10") {
                    threats::mark_follow(s, &t, owner, s.ply + 2., c) * 5.
                } else {
                    0.
                }
        }
        "move" if u.is_some() && cmd.x.is_some() && cmd.y.is_some() => {
            let u = u.unwrap();
            eval::placement(
                s,
                u,
                Point {
                    x: cmd.x.unwrap(),
                    y: cmd.y.unwrap(),
                },
                false,
                None,
                c,
            ) - eval::placement(s, u, u.at(), false, None, c)
                + 0.5
        }
        "skill"
            if u.is_some_and(|u| u.kind.is("14"))
                && t.as_ref()
                    .and_then(|t| t.unit.as_ref())
                    .is_some_and(|u| u.kind.is("14")) =>
        {
            30.
        }
        "charge" => 8.,
        "cast" | "equip" => 12.,
        _ => 6.,
    }
}
pub fn point_rank(
    s: &State,
    a: &Action,
    cmd: &Command,
    p: Point,
    u: Option<&Unit>,
    c: &Catalog,
) -> f64 {
    let id = a.id.split(':').next().unwrap();
    let owner = crate::runtime::viewer(s);
    let value_of = |v: &Unit| eval::material(s, v, c);
    if ["move", "deploy"].contains(&cmd.kind.as_str())
        || ["dash", "revive", "hut-spawn"].contains(&id)
    {
        let kind = if id == "hut-spawn" {
            "20".to_owned()
        } else if let Some(id) = &cmd.death_id {
            s.deaths
                .iter()
                .find(|v| v["id"] == *id)
                .map(|v| crate::preparation::kind(&v["kind"]).key())
                .unwrap_or("1".into())
        } else {
            crate::preparation::card(s, cmd)
                .map(|v| crate::preparation::kind(&v["kind"]).key())
                .unwrap_or("1".into())
        };
        let raw = ghost(s, &kind, owner, 0., p, c);
        let actor = u.unwrap_or(&raw);
        let mut moved = actor.fork();
        moved.x = p.x;
        moved.y = p.y;
        let mut value = eval::placement(s, actor, p, false, None, c)
            + threats::base_cover(s, &moved, c)
            + threats::payload_cover(s, &moved, &value_of, c);
        if cmd.kind == "move"
            && let Some(u) = u
        {
            let cards: Vec<_> = s.extra["hands"][u.owner.to_string()]
                .as_array()
                .unwrap()
                .iter()
                .filter(|v| !stored(c, &crate::preparation::kind(&v["kind"])))
                .collect();
            if !cards.is_empty() {
                let before = geometry::deployment_rows(s, u.owner);
                let mut view = s.fork();
                for v in &mut view.units {
                    if v.id == u.id {
                        v.x = p.x;
                        v.y = p.y;
                    }
                }
                let gained: Vec<_> = geometry::deployment_rows(&view, u.owner)
                    .into_iter()
                    .filter(|y| !before.contains(y))
                    .collect();
                if !gained.is_empty()
                    && cards.iter().any(|card| {
                        geometry::all_cells().any(|at| {
                            gained.contains(&(at.y as usize))
                                && geometry::can_deploy(
                                    &view,
                                    &ghost(
                                        &view,
                                        &crate::preparation::kind(&card["kind"]).key(),
                                        u.owner,
                                        0.,
                                        at,
                                        c,
                                    ),
                                    at,
                                    c,
                                )
                        })
                    })
                {
                    value += cards.len().min(2) as f64 * 8.;
                }
            }
        }
        if actor.runner() {
            value += s
                .units
                .iter()
                .filter(|v| v.id != actor.id && v.owner != actor.owner && geometry::covers(v, p))
                .map(|v| v.hp.min(30.) + 10.)
                .sum::<f64>();
        }
        return value;
    }
    if id == "wall" {
        let wall = ghost(s, "wall", owner, 0., p, c);
        return threats::base_cover(s, &wall, c) + threats::payload_cover(s, &wall, &value_of, c)
            - u.map_or(0., |u| geometry::distance(p, u.at()) * 0.1);
    }
    if id == "hook"
        && let Some(v) = cmd.target_id.as_deref().and_then(|id| s.unit(id))
    {
        return -eval::placement(s, v, p, false, None, c);
    }
    if ["blast", "cross", "ice-mark"].contains(&id) {
        let mut value = 0.;
        for t in geometry::targets(s) {
            if t.unit.as_ref().map_or(t.owner, |v| v.side()) == owner {
                continue;
            }
            let amount = t
                .footprint()
                .iter()
                .map(|q| match id {
                    "blast" => {
                        if q.x >= p.x && q.x <= p.x + 1. && q.y >= p.y && q.y <= p.y + 1. {
                            20.
                        } else {
                            0.
                        }
                    }
                    "cross" => {
                        if (q.x == p.x || q.y == p.y)
                            && u.is_some_and(|u| (q.x - u.x).abs() <= 5. && (q.y - u.y).abs() <= 5.)
                        {
                            if *q == p { 40. } else { 20. }
                        } else {
                            0.
                        }
                    }
                    _ => {
                        if *q == p {
                            20.
                        } else {
                            0.
                        }
                    }
                })
                .sum::<f64>();
            if amount != 0. {
                value += t
                    .unit
                    .as_ref()
                    .map_or(amount * 2., |u| u.hp.min(amount) + 15.);
            }
        }
        return value;
    }
    u.map_or(0., |u| -geometry::distance(p, u.at()))
}
