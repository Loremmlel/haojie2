use super::spatial::stats_for;
use super::{
    hp, passive,
    spatial::{action_window, hit_distance},
    stored, threats,
};
use crate::{
    geometry::{self, Target},
    model::{Catalog, Point, State, Unit},
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

pub fn material(s: &State, u: &Unit, c: &Catalog) -> f64 {
    let st = stats_for(s, u, c);
    let ratio = (u.hp / u.max_hp).min(1.);
    let attack = if passive(s, u) && u.kind.is("u2") {
        (st.attack - u.reserve.charge * 15.).max(0.)
    } else if passive(s, u) && u.kind.is("15") {
        (st.attack - u.reserve.charge * c.rule("/accumulator/attack")).max(0.)
    } else if passive(s, u) && u.kind.is("4") {
        st.attack / 3.
    } else if c.by_kind(&u.kind).actions == 0.5 {
        st.attack / 2.
    } else {
        st.attack.max(0.)
    };
    let mut value = 14.
        + u.hp * 0.38
        + u.max_hp * 0.12
        + attack * st.actions.sqrt() * (0.65 + 0.3 * ratio)
        + (st.range
            - if passive(s, u) && u.kind.is("15") {
                u.reserve.charge * c.rule("/accumulator/range")
            } else {
                0.
            })
        .min(8.)
            * 2.;
    if passive(s, u) {
        for k in u.kinds() {
            value += match k.key_ref().as_ref() {
                "2" => 16.,
                "3" => 40.,
                "4" => 10.,
                "6" => 12.,
                "7" => 18.,
                "10" => 12.,
                "11" => 15.,
                "12" => 12.,
                "14" => 6.,
                "16" => 8.,
                "19" | "20" | "21" => 12.,
                "23" => 4.,
                "24" | "26" => 14.,
                "u3" => 26.,
                "u4" => 30.,
                "u6" => 32.,
                "u8" => 28.,
                "u13" => 18.,
                "u14" => 38.,
                "u15" => 30.,
                "u18" => 28.,
                "u19" => 20.,
                "u20" => 25.,
                "u21" => 24.,
                "u22" => 20.,
                "u23" => 18.,
                "u24" => 17.,
                "firelord" => 65.,
                "sage" => 55.,
                "formless" => 38.,
                "slayer" => 60.,
                "citadel" | "archmage" => 55.,
                _ => 0.,
            };
        }
    }
    if passive(s, u) && u.kind.is("sage") {
        value += s
            .units
            .iter()
            .filter(|v| v.id != u.id && v.side() == u.owner)
            .map(|v| (v.max_hp - v.hp) * 0.18)
            .sum::<f64>()
            .min(60.);
    }
    if u.any(&["grave", "wall"]) {
        value = 4. + u.hp * 0.16;
    }
    if u.kind.is("u25") {
        value -= 7.;
    }
    if u.any(&["u12", "u12p"]) {
        value += 30.;
    }
    if u.extra.contains_key("expiresAt") {
        value *= 0.35;
    }
    if st.frozen {
        value *= 0.5;
    } else if st.sleeping || st.stunned {
        value *= 0.88;
    }
    let nearest = s
        .units
        .iter()
        .filter(|v| v.owner != u.owner)
        .map(|v| geometry::distance(u.at(), v.at()))
        .fold(
            geometry::distance(u.at(), geometry::base_point(3 - u.owner)),
            f64::min,
        );
    let discount = 1. / (1. + (nearest - st.range.min(6.)).max(0.) * 0.16);
    value += if u.kind.is("4") {
        u.reserve.charge.min(2.) * 12. + (u.reserve.charge - 2.).max(0.) * 2.
    } else {
        u.reserve.charge
            * if u.kind.is("u2") {
                6.75
            } else if u.kind.is("u6") {
                16.
            } else if u.reserve.charge_type.as_str() == "move" {
                7.
            } else {
                6.
            }
    } * discount;
    value += if u.weapon("u5") {
        20.
    } else if u.weapon("u16") {
        16.
    } else if u.weapon("u28") {
        22.
    } else {
        0.
    };
    for (i, e) in u.effects.iter().enumerate() {
        if e.kind == "immune" {
            let mut bare = u.fork();
            bare.effects.remove(i);
            value += (threats::incoming(s, &bare, false, c) * 0.35).min(24.);
        }
        if e.kind == "attack" {
            value += e.amount.unwrap_or(0.).min(15.) * 0.65 * discount;
        }
        if e.kind == "burn" {
            value -= u.hp.min(12.) * 0.5;
        }
    }
    value.max(1.)
}
pub fn unit(s: &State, u: &Unit, value_of: &dyn Fn(&Unit) -> f64, c: &Catalog) -> f64 {
    let mut value = value_of(u)
        + threats::payload(s, u, "execute", value_of, c)
        + threats::payload(s, u, "convert", value_of, c);
    for e in &u.effects {
        if e.kind == "mark" {
            value -= 5. * 0.9 * threats::mark_follow(s, &Target::from(u), e.owner, e.until, c);
        }
    }
    value
}
pub fn card(s: &State, card: &Value, owner: usize, c: &Catalog) -> f64 {
    let kind = crate::preparation::kind(&card["kind"]);
    if !stored(c, &kind) {
        let mut raw = crate::resolution::template(
            s,
            &kind.key(),
            owner,
            Point {
                x: 5.,
                y: if owner == 1 { 6. } else { 8. },
            },
            "",
            c,
        );
        raw.born = 0.;
        return material(s, &raw, c) * 0.76;
    }
    let remaining = card["expiresAt"].as_f64().unwrap_or(s.turn(owner) + 20.) - s.turn(owner);
    let value = match kind.key_ref().as_ref() {
        "8" => 22.,
        "17" => 14.,
        "18" => 27.,
        "22" => 34.,
        "25" => 38.,
        "u5" => 32.,
        "u9" => 42.,
        "u11" | "u16" | "u17" => 30.,
        "u26" | "u28" => 34.,
        _ => 20.,
    };
    value * (0.4 + (remaining - 1.).max(0.) * 0.17).min(1.)
}
pub fn head_value(heads: f64) -> f64 {
    heads.min(2.) * 3. + (heads - 2.).clamp(0., 2.) * 2. + (heads - 4.).clamp(0., 4.)
}
pub fn placement(
    s: &State,
    u: &Unit,
    p: Point,
    next: bool,
    value_of: Option<&dyn Fn(&Unit) -> f64>,
    c: &Catalog,
) -> f64 {
    let mut moved = u.fork();
    moved.x = p.x;
    moved.y = p.y;
    let st = stats_for(s, &moved, c);
    let foe = geometry::base_point(3 - u.owner);
    let mut nearest = geometry::cell_iter(&moved, p)
        .map(|q| geometry::distance(q, foe))
        .fold(f64::INFINITY, f64::min);
    for v in s
        .units
        .iter()
        .filter(|v| v.id != u.id && v.side() != u.owner)
    {
        for q in geometry::cell_iter(v, v.at()) {
            for a in geometry::cell_iter(&moved, p) {
                nearest = nearest.min(geometry::distance(a, q));
            }
        }
    }
    let support = u.any(&[
        "2", "3", "6", "u3", "u13", "u14", "u15", "u19", "u21", "u22", "sage", "citadel",
        "archmage",
    ]);
    let contact = (nearest - st.range).max(0.) / st.movement.clamp(0.5, 3.);
    let mut score = 20. / (1. + contact * 0.4)
        + 8. / (1. + (geometry::distance(p, foe) - st.range).max(0.) * 0.15);
    if st.attack > 0. || u.any(&["u6", "u12", "u12p", "firelord"]) {
        score += (st.range - nearest + 1.).clamp(0., 3.) * 2.;
    }
    if support {
        score += s
            .units
            .iter()
            .filter(|v| {
                v.id != u.id && v.side() == u.owner && geometry::distance(v.at(), p) <= st.range
            })
            .map(|v| 2. + ((v.max_hp - v.hp) * 0.12).min(5.))
            .sum::<f64>()
            .min(16.);
        score -= (2. - nearest).max(0.) * 3.;
    }
    let danger = threats::incoming(s, &moved, next, c);
    score -= moved.hp.min(danger)
        * if next {
            if u.any(&["grave", "wall"]) {
                0.06
            } else {
                0.15
            }
        } else {
            0.42
        };
    if danger >= moved.hp {
        score -= if next {
            (value_of.map_or_else(|| material(s, &moved, c), |f| f(&moved)) * 0.45).min(55.)
        } else {
            (14. + st.attack * 0.35).min(45.)
        };
    }
    let stacked = s
        .units
        .iter()
        .filter(|v| v.id != u.id && v.owner == u.owner && geometry::covers(v, p))
        .count();
    if !next && u.kind.is("u25") && stacked > 0 {
        score -= (stacked as f64).powf(1.35) * 2.;
    }
    score
}
fn formation(s: &State, side: usize, c: &Catalog) -> f64 {
    let view = action_window(s, side, true);
    let home = geometry::base_point(side);
    let foe = geometry::base_point(3 - side);
    let mut lanes = HashSet::new();
    let mut stacks = HashMap::<String, usize>::new();
    let mut value = 0.;
    let enemies: Vec<_> = geometry::targets(&view)
        .into_iter()
        .filter(|t| t.owner != side)
        .collect();
    for u in view.units.iter().filter(|u| u.side() == side) {
        let st = stats_for(&view, u, c);
        let contacts = if threats::ready(&view, u, c) {
            enemies
                .iter()
                .filter(|t| hit_distance(&view, u, t, "", c).is_finite())
                .count()
        } else {
            0
        };
        if contacts > 0 {
            lanes.insert(u.x as i32);
            value += contacts.min(3) as f64 * 1.6;
        }
        if u.kind.is("u25") {
            *stacks.entry(format!("{},{}", u.x, u.y)).or_default() += 1;
        }
        if st.movement <= 0. || st.frozen || u.any(&["grave", "wall"]) {
            continue;
        }
        let points: Vec<_> = if u.kind.is("13") && !u.silenced {
            [
                Point {
                    x: u.x + 3.,
                    y: u.y,
                },
                Point {
                    x: u.x - 3.,
                    y: u.y,
                },
                Point {
                    x: u.x,
                    y: u.y + 3.,
                },
                Point {
                    x: u.x,
                    y: u.y - 3.,
                },
            ]
            .into_iter()
            .filter(|p| geometry::movement_path(&view, u, *p, 3., true, c).is_some())
            .collect()
        } else {
            geometry::neighbors(u.at())
                .filter(|p| geometry::can_place(&view, u, *p, c))
                .collect()
        };
        if points.is_empty() && contacts == 0 {
            value -= 12.;
        } else if contacts == 0 {
            let forward = points
                .iter()
                .filter(|p| geometry::distance(**p, foe) < geometry::distance(u.at(), foe))
                .count();
            value += forward.min(2) as f64 * 1.5;
            if forward == 0 && geometry::distance(u.at(), foe) > st.range + 1. {
                value -= 4.;
            }
        }
        if geometry::distance(u.at(), home) < 3. && contacts == 0 && st.attack > 0. {
            value -= 4.;
        }
    }
    for n in stacks.values() {
        value -= (*n as f64 - 1.).max(0.).powf(1.6) * 3.;
    }
    value + lanes.len().min(5) as f64 * 3.
}
pub fn breakdown(s: &State, side: usize, c: &Catalog) -> Value {
    super::spatial::with_cache(s, || breakdown_inner(s, side, c))
}
fn breakdown_inner(s: &State, side: usize, c: &Catalog) -> Value {
    let mut bases = 0.;
    let mut resources = 0.;
    let mut force = 0.;
    let mut position = 0.;
    let mut formation_value = 0.;
    let mut effects = 0.;
    if let Some(winner) = s.extra.get("winner") {
        bases = if winner == "draw" {
            0.
        } else if winner == side {
            100000.
        } else {
            -100000.
        };
        return json!({"bases":bases,"resources":0.,"force":0.,"position":0.,"formation":0.,"effects":0.,"total":bases});
    }
    let mut dangers = [0.; 3];
    for p in [1, 2] {
        let sign = if p == side { 1. } else { -1. };
        let health = hp(s, p);
        let danger = threats::base_threat(s, p, c);
        dangers[p] = danger;
        if s.aura(p, "laoqian") {
            effects += sign * 155.;
        }
        bases += sign
            * (health * 3.1
                - danger * 0.65
                - if danger >= health {
                    900. + (danger - health) * 4.
                } else if danger > health * 0.6 {
                    120.
                } else {
                    0.
                });
        let cards = s.extra["hands"][p.to_string()].as_array().unwrap();
        resources += sign
            * (head_value(s.extra["heads"][p.to_string()].as_f64().unwrap_or(0.))
                + s.extra["bonus"][p.to_string()].as_f64().unwrap_or(0.) * 28.
                + (if s.phase == "play" && s.active == p {
                    s.summon_slots.max(0.) * 28.
                } else {
                    0.
                })
                + cards.iter().map(|v| card(s, v, p, c)).sum::<f64>());
        let reinforcements = cards
            .iter()
            .filter(|v| !stored(c, &crate::preparation::kind(&v["kind"])))
            .count()
            .min(2);
        position += sign
            * geometry::deployment_rows(s, p)
                .iter()
                .filter(|y| if p == 1 { **y > 8 } else { **y < 6 })
                .count() as f64
            * (2. + reinforcements as f64 * 8.);
        formation_value += sign * formation(s, p, c);
        for e in &s.base_effects[&p.to_string()] {
            if e.kind == "mark" {
                let t = geometry::find_target(s, &format!("base-{p}")).unwrap();
                effects -= sign * 5. * 0.9 * threats::mark_follow(s, &t, e.owner, e.until, c);
            }
        }
    }
    let value_of = |u: &Unit| material(s, u, c);
    let mut assets = [0.; 3];
    for u in &s.units {
        if u.side() != 0 {
            assets[u.owner] += value_of(u);
        }
    }
    for u in &s.units {
        if u.side() == 0 {
            continue;
        }
        let sign = if u.owner == side { 1. } else { -1. };
        force += sign * unit(s, u, &value_of, c);
        position += sign * placement(s, u, u.at(), true, Some(&value_of), c);
        let st = stats_for(s, u, c);
        let advantage = ((assets[u.owner] - assets[3 - u.owner])
            / (assets[u.owner] + assets[3 - u.owner]).max(100.))
        .max(0.);
        if st.movement > 0. && st.attack > 0. && !st.frozen && hp(s, u.owner) > dangers[u.owner] {
            position += sign * advantage * 24.
                / (1.
                    + (geometry::distance(u.at(), geometry::base_point(3 - u.owner)) - st.range)
                        .max(0.)
                        * 0.25);
        }
        for h in s.extra["hazards"].as_array().into_iter().flatten() {
            if geometry::cell_iter(u, u.at()).any(|p| {
                if h["axis"] == "row" {
                    Some(p.y) == h["line"].as_f64()
                } else {
                    Some(p.x) == h["line"].as_f64()
                }
            }) {
                effects -= sign * u.hp.min(20.) * 0.5;
            }
        }
        for m in s.extra["iceMarks"].as_array().into_iter().flatten() {
            if m["owner"] != u.owner
                && geometry::cell_iter(u, u.at())
                    .any(|p| Some(p.x) == m["x"].as_f64() && Some(p.y) == m["y"].as_f64())
            {
                effects -= sign * 18.;
            }
        }
    }
    for link in &s.siphons {
        let from_id = link["fromId"].as_str().unwrap_or("");
        let to_id = link["toId"].as_str().unwrap_or("");
        let from = s.unit(from_id);
        let to = s.unit(to_id);
        let owner = |id: &str, u: Option<&Unit>| {
            u.map_or_else(
                || {
                    id.strip_prefix("base-")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0)
                },
                |v| v.owner,
            )
        };
        let a = owner(from_id, from);
        let b = owner(to_id, to);
        if a > 0 {
            effects +=
                if a == side { -1. } else { 1. } * from.map_or(300., |v| v.hp).min(20.) * 0.9;
        }
        if b > 0 {
            effects += if b == side { 1. } else { -1. }
                * to.map_or(300. - hp(s, b), |v| v.max_hp - v.hp).min(20.);
        }
    }
    json!({"bases":bases,"resources":resources,"force":force,"position":position,"formation":formation_value,"effects":effects,"total":bases+resources+force+position+formation_value+effects})
}
pub fn evaluate(s: &State, side: usize, c: &Catalog) -> f64 {
    breakdown(s, side, c)["total"].as_f64().unwrap()
}
