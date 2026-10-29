use super::spatial::stats_for;
use super::{
    hp, passive,
    spatial::{action_window, front_hit, hit_distance},
};
use crate::{
    geometry::{self, Target},
    model::{Catalog, Point, State, Unit},
};

#[derive(Clone, Copy)]
pub struct Packet {
    pub damage: f64,
    pub probability: f64,
}
fn packet(damage: f64, probability: f64) -> Packet {
    Packet {
        damage,
        probability,
    }
}
pub fn ready(s: &State, u: &Unit, c: &Catalog) -> bool {
    let st = stats_for(s, u, c);
    st.remaining > 0.
        && !st.sleeping
        && !st.frozen
        && !st.stunned
        && !(passive(s, u) && ((u.has("4") && u.reserve.ready_charge < 2.) || u.has("firelord")))
}
pub fn hit_packets(
    s: &State,
    u: &Unit,
    t: &Target,
    direction: Option<&str>,
    c: &Catalog,
) -> Vec<Packet> {
    if t.unit.as_ref().is_some_and(|v| s.effect(v, "immune")) {
        return vec![packet(0., 1.)];
    }
    let attack = stats_for(s, u, c).attack;
    if attack < 0. {
        return vec![packet(0., 1.)];
    }
    let mut packets = vec![packet(attack, 1.)];
    if passive(s, u) {
        for k in u.kinds() {
            packets = packets
                .into_iter()
                .flat_map(|p| {
                    let choices = match k.key_ref().as_ref() {
                        "1" => vec![
                            packet(
                                p.damage + c.rule("/charger/heavyBonus"),
                                c.rule("/charger/heavyChance"),
                            ),
                            packet(
                                p.damage + c.rule("/charger/bonus"),
                                c.rule("/charger/criticalChance") - c.rule("/charger/heavyChance"),
                            ),
                            packet(p.damage, 1. - c.rule("/charger/criticalChance")),
                        ],
                        "u1" => vec![
                            packet(
                                c.rule("/superCritical/lethalDamage"),
                                c.rule("/superCritical/lethalChance"),
                            ),
                            packet(p.damage * 2., c.rule("/superCritical/doubleChance")),
                            packet(
                                p.damage,
                                1. - c.rule("/superCritical/lethalChance")
                                    - c.rule("/superCritical/doubleChance"),
                            ),
                        ],
                        "u8" => {
                            let rate = (c.rule("/vampire/base")
                                + c.rule("/vampire/perKill") * u.kills)
                                .min(1.);
                            vec![packet(p.damage * 2., rate), packet(p.damage, 1. - rate)]
                        }
                        "u27" if t.unit.is_none() => vec![packet(c.rule("/minerBaseDamage"), 1.)],
                        _ => vec![packet(p.damage, 1.)],
                    };
                    choices
                        .into_iter()
                        .map(move |q| packet(q.damage, p.probability * q.probability))
                })
                .collect();
        }
        if u.has("10") {
            let full = t
                .unit
                .as_ref()
                .map_or(hp(s, t.owner) == 300., |v| v.hp == v.max_hp);
            packets = vec![packet(
                if attack > 0. {
                    attack
                } else if full {
                    c.rule("/catapultMarkDamage")
                } else {
                    0.
                },
                1.,
            )];
        }
    }
    if let Some(v) = &t.unit {
        if passive(s, v) {
            if v.has("u18") {
                for p in &mut packets {
                    if p.damage <= c.rule("/kingAttackImmunity") {
                        p.damage = 0.;
                    }
                }
            }
            let frontal = if let Some(d) = direction {
                d == if t.owner == 1 { "up" } else { "down" }
            } else {
                let ignored = s
                    .units
                    .iter()
                    .any(|v| v.id == t.id && (v.x != t.at.x || v.y != t.at.y));
                front_hit(s, u, t, if ignored { &t.id } else { "" }, c)
            };
            if v.has("24") && frontal {
                for p in &mut packets {
                    p.damage = p.damage.min(c.rule("/frontDamageCap"));
                }
            }
        }
        if v.has("17p") && !v.silenced {
            packets = packets
                .into_iter()
                .flat_map(|p| {
                    if p.damage > 0. {
                        vec![
                            packet(0., p.probability * c.rule("/littleGoldImmunity")),
                            packet(
                                p.damage,
                                p.probability * (1. - c.rule("/littleGoldImmunity")),
                            ),
                        ]
                    } else {
                        vec![p]
                    }
                })
                .collect();
        }
    }
    packets
}
pub fn pressure(s: &State, u: &Unit, t: &Target, ignore: &str, c: &Catalog) -> f64 {
    if u.has("firelord") {
        if !passive(s, u)
            || s.effect(u, "freeze")
            || t.unit.as_ref().is_some_and(|v| s.effect(v, "immune"))
        {
            return 0.;
        }
        let mut view = s.fork();
        if let Some(v) = &t.unit {
            view.units.retain(|v| v.id != t.id);
            view.units.push(v.fork());
        }
        let enemies: Vec<_> = geometry::targets(&view)
            .into_iter()
            .filter(|v| v.unit.as_ref().map_or(v.owner, |v| v.side()) != u.owner)
            .collect();
        let inside = |p: Point| {
            geometry::inside(p)
                && (p.x - u.x).abs() <= c.rule("/firelord/radius")
                && (p.y - u.y).abs() <= c.rule("/firelord/radius")
        };
        let mut candidates: Vec<_> = enemies
            .iter()
            .filter(|v| v.footprint().into_iter().any(inside))
            .collect();
        candidates.sort_by(|a, b| {
            b.unit
                .as_ref()
                .map_or(hp(s, b.owner), |v| v.hp)
                .total_cmp(&a.unit.as_ref().map_or(hp(s, a.owner), |v| v.hp))
                .then(a.at.y.total_cmp(&b.at.y))
                .then(a.at.x.total_cmp(&b.at.x))
                .then(a.id.cmp(&b.id))
        });
        let Some(target) = candidates.first() else {
            return 0.;
        };
        let mut cells: Vec<_> = target
            .footprint()
            .into_iter()
            .filter(|p| inside(*p))
            .collect();
        cells.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
        let impact = cells[0];
        return if t.footprint().contains(&impact) {
            c.rule("/firelord/damage")
        } else if t
            .footprint()
            .iter()
            .any(|p| geometry::neighbors(impact).any(|q| q == *p))
        {
            c.rule("/firelord/splash")
        } else {
            0.
        };
    }
    if stats_for(s, u, c).attack < 0.
        || !ready(s, u, c)
        || !hit_distance(s, u, t, ignore, c).is_finite()
    {
        return 0.;
    }
    if u.has("9") && passive(s, u) && u.attacked.contains(&t.id) {
        return 0.;
    }
    if t.unit.as_ref().is_some_and(|v| s.effect(v, "immune")) {
        return 0.;
    }
    let mut amount = hit_packets(s, u, t, None, c)
        .iter()
        .map(|p| p.damage * p.probability)
        .sum::<f64>();
    let count = if passive(s, u) && u.any(&["9", "4", "10"]) {
        1.
    } else {
        stats_for(s, u, c).remaining
    };
    if u.has("15") && !u.silenced && u.reserve.charge > 0. {
        let mut empty = u.fork();
        empty.reserve.charge = 0.;
        empty.reserve.ready_charge = 0.;
        let follow = if hit_distance(s, &empty, t, ignore, c).is_finite() {
            hit_packets(s, &empty, t, None, c)
                .iter()
                .map(|p| p.damage * p.probability)
                .sum()
        } else {
            0.
        };
        amount += (count - 1.).max(0.) * follow;
    } else {
        amount *= count;
    }
    if t.unit.is_some() {
        if passive(s, u) && u.has("u4") {
            amount += 12.;
        }
        if u.weapon("u5") {
            amount += 15.;
        }
        if passive(s, u) && u.has("u6") && !u.weapon("u5") {
            amount += 15.;
        }
    }
    amount
}
pub fn incoming(s: &State, target: &Unit, next: bool, c: &Catalog) -> f64 {
    let view = action_window(s, 3 - target.owner, next);
    let t = Target::from(target);
    let ignore = if s
        .unit(&target.id)
        .is_some_and(|u| u.x != target.x || u.y != target.y)
    {
        target.id.as_str()
    } else {
        ""
    };
    view.units
        .iter()
        .filter(|v| v.owner != target.owner && v.id != target.id)
        .map(|v| pressure(&view, v, &t, ignore, c))
        .sum()
}
pub fn base_threat(s: &State, defender: usize, c: &Catalog) -> f64 {
    let view = action_window(s, 3 - defender, false);
    let t = geometry::find_target(&view, &format!("base-{defender}")).unwrap();
    view.units
        .iter()
        .filter(|u| u.owner != defender)
        .map(|u| pressure(&view, u, &t, "", c))
        .sum()
}
pub fn mark_follow(s: &State, t: &Target, owner: usize, until: f64, c: &Catalog) -> f64 {
    let view = action_window(s, owner, false);
    if view.ply >= until {
        return 0.;
    }
    let mut best: f64 = 0.;
    for u in &view.units {
        if u.owner != owner
            || u.has("10")
            || !ready(&view, u, c)
            || (u.has("9") && !u.silenced && u.attacked.contains(&t.id))
            || !hit_distance(&view, u, t, "", c).is_finite()
        {
            continue;
        }
        let hp = t.unit.as_ref().map_or(hp(&view, t.owner), |v| v.hp);
        let guarded = t
            .unit
            .as_ref()
            .is_some_and(|v| crate::resolution::guardian(&view, v, c).is_some());
        let packets = hit_packets(&view, u, t, None, c);
        let mut probability = packets
            .iter()
            .filter(|p| p.damage < hp || guarded)
            .map(|p| p.probability)
            .sum::<f64>();
        if probability == 0. {
            continue;
        }
        let hit = packets
            .iter()
            .map(|p| p.probability * p.damage)
            .sum::<f64>();
        if t.unit.is_some()
            && hit + 5. < hp
            && incoming(&view, u, false, c) >= u.hp
            && stats_for(&view, u, c).movement >= 1.
            && geometry::neighbors(u.at()).any(|p| {
                if !geometry::can_place(&view, u, p, c) {
                    return false;
                }
                let mut moved = u.fork();
                moved.x = p.x;
                moved.y = p.y;
                incoming(&view, &moved, false, c) < u.hp
            })
        {
            probability *= 0.25;
        }
        best = best.max(probability);
    }
    best
}
pub fn payload(
    s: &State,
    u: &Unit,
    kind: &str,
    value_of: &dyn Fn(&Unit) -> f64,
    c: &Catalog,
) -> f64 {
    payload_detail(s, u, kind, value_of, c).0
}
fn payload_detail(
    s: &State,
    u: &Unit,
    kind: &str,
    value_of: &dyn Fn(&Unit) -> f64,
    c: &Catalog,
) -> (f64, Option<String>) {
    let Some(effect) = u.effects.iter().find(|e| e.kind == kind) else {
        return (0., None);
    };
    if kind == "convert" && u.has("5") {
        return (0., None);
    }
    let clock = s.ply
        + if effect.global == Some(true) {
            0.
        } else {
            u.offset
        };
    let view = action_window(s, u.owner, effect.from > clock);
    let Some(carrier) = view.unit(&u.id) else {
        return (0., None);
    };
    if !view.active_effect(effect, Some(carrier)) || !ready(&view, carrier, c) {
        return (0., None);
    }
    let mut best: f64 = 0.;
    let mut best_id = None;
    for victim in &view.units {
        let target = Target::from(victim);
        if victim.side() == u.owner
            || (kind == "convert" && victim.side() == 0)
            || !geometry::top_target(&view, &target, c)
            || view.effect(victim, "immune")
            || !hit_distance(&view, carrier, &target, "", c).is_finite()
        {
            continue;
        }
        if view.units.iter().any(|v| {
            v.has("u15")
                && v.owner == victim.side()
                && passive(&view, v)
                && geometry::attack_path(
                    &view,
                    v,
                    &target,
                    stats_for(&view, v, c).range,
                    None,
                    false,
                )
                .is_some()
        }) {
            continue;
        }
        let guarded = victim.hp > 1. && crate::resolution::guardian(&view, victim, c).is_some();
        let probability = if kind == "execute" {
            1.
        } else {
            hit_packets(&view, carrier, &target, None, c)
                .iter()
                .filter(|p| p.damage > 0. && (p.damage < victim.hp || guarded))
                .map(|p| p.probability)
                .sum()
        };
        let value = value_of(victim) * probability * if kind == "convert" { 1.3 } else { 0.85 };
        if value > best {
            best = value;
            best_id = Some(victim.id.clone());
        }
    }
    let survival = (1. - 0.75 * (incoming(s, u, false, c) / u.hp.max(1.)).min(1.)).max(0.12);
    (
        best * if effect.from > clock { 0.55 } else { 0.8 } * survival,
        best_id,
    )
}
pub fn payload_cover(
    s: &State,
    defender: &Unit,
    value_of: &dyn Fn(&Unit) -> f64,
    c: &Catalog,
) -> f64 {
    let mut route = Vec::new();
    let mut carriers = Vec::new();
    for u in &s.units {
        if u.owner == defender.owner || u.weapon("u28") {
            continue;
        }
        for kind in ["execute", "convert"] {
            let (value, victim) = payload_detail(s, u, kind, value_of, c);
            let Some(victim) = victim.and_then(|id| s.unit(&id)) else {
                continue;
            };
            if value == 0. {
                continue;
            }
            let view = action_window(s, u.owner, false);
            let Some(carrier) = view.unit(&u.id) else {
                continue;
            };
            route.extend(
                geometry::attack_path(
                    &view,
                    carrier,
                    &Target::from(victim),
                    stats_for(&view, carrier, c).range,
                    None,
                    false,
                )
                .unwrap_or_default(),
            );
            carriers.push((u.id.clone(), kind, value));
        }
    }
    if !geometry::cell_iter(defender, defender.at()).any(|p| route.contains(&p)) {
        return 0.;
    }
    let mut view = s.fork();
    view.units.retain(|u| u.id != defender.id);
    view.units.push(defender.fork());
    carriers
        .iter()
        .map(|(id, kind, value)| {
            (value - payload(&view, view.unit(id).unwrap(), kind, value_of, c)).max(0.)
        })
        .sum()
}
pub fn base_cover(s: &State, defender: &Unit, c: &Catalog) -> f64 {
    let view = action_window(s, 3 - defender.owner, false);
    let target = geometry::find_target(&view, &format!("base-{}", defender.owner)).unwrap();
    let mut before = 0.;
    let mut route = Vec::new();
    for u in view.units.iter().filter(|v| v.owner != defender.owner) {
        let amount = pressure(&view, u, &target, "", c);
        if amount == 0. {
            continue;
        }
        before += amount;
        if !u.weapon("u28") {
            route.extend(
                geometry::attack_path(&view, u, &target, stats_for(&view, u, c).range, None, false)
                    .unwrap_or_default(),
            );
        }
    }
    if before == 0. || !geometry::cell_iter(defender, defender.at()).any(|p| route.contains(&p)) {
        return 0.;
    }
    let mut moved = view.fork();
    moved.units.retain(|v| v.id != defender.id);
    moved.units.push(defender.fork());
    let after = moved
        .units
        .iter()
        .filter(|u| u.owner != defender.owner)
        .map(|u| pressure(&moved, u, &target, "", c))
        .sum::<f64>();
    let hp = hp(s, defender.owner);
    (hp.min(before) - hp.min(after)).max(0.) * 3.1
        + if before >= hp && after < hp { 180. } else { 0. }
}
