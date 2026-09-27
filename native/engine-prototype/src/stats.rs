use crate::geometry::{Target, attack_path, cells, distance};
use crate::model::{Catalog, Kind, State, Unit, extra_number, number};
use crate::movement::{movement_stats, reserve};
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub attack: f64,
    pub range: f64,
    pub actions: f64,
    pub remaining: f64,
    #[serde(rename = "move")]
    pub movement: f64,
    pub mode: String,
    pub operation_limit: f64,
    pub operations_left: f64,
    pub frozen: bool,
    pub stunned: bool,
    pub sleeping: bool,
}
pub fn attack_charge(u: &Unit, catalog: &Catalog) -> Option<Kind> {
    u.kinds()
        .into_iter()
        .find(|k| catalog[&k.key()].actions == 0.5)
}
pub fn banner_count(s: &State, owner: usize) -> usize {
    s.landmarks()
        .iter()
        .filter(|u| u.live() && u.owner == owner && !u.silenced && u.has("s8"))
        .count()
}

/// 完整属性查询与 TS getStats 对照；攻击光环只读印刷射程，禁止递归求攻击力。
pub fn stats(s: &State, u: &Unit, catalog: &Catalog) -> Stats {
    let d = &catalog[&u.kind.key()];
    let enabled = !u.silenced;
    let age = s.turns[&u.owner.to_string()] + u.offset / 2.0 - u.born;
    let frozen = s.effect(u, "freeze");
    let stunned = s.effect(u, "stun");
    let sleeping = if d.landmark.is_some() {
        u.hp <= 0.0
    } else {
        age <= 0.0 || (enabled && u.has("23") && age < 2.0)
    };
    let (locked, left, movement) = movement_stats(s, u, catalog);
    let mut attack = d.attack + extra_number(u, "attackBonus");
    let mut range = d.range + extra_number(u, "rangeBonus");
    let mut actions = if d.landmark.is_some() || (d.actions > 0.0 && d.actions < 1.0) {
        1.0
    } else {
        d.actions
    };
    if enabled && u.has("3p") {
        let n = s
            .units
            .iter()
            .filter(|v| {
                cells(v, v.at())
                    .iter()
                    .any(|p| (p.x - u.x).abs().max((p.y - u.y).abs()) <= 1.0)
            })
            .count() as f64;
        attack = (40.0 - 5.0 * n).max(0.0) + extra_number(u, "attackBonus");
        range = n + extra_number(u, "rangeBonus");
    }
    if enabled && u.has("4") && number(&reserve(u, &Kind::Number(4), catalog)["charge"]) >= 5.0 {
        range += 1.0;
    }
    if enabled && u.has("15") {
        let r = reserve(u, &Kind::Number(15), catalog);
        if r["chargeType"] == "attack" {
            attack += number(&r["charge"]) * catalog.rule("/accumulator/attack");
            range += number(&r["charge"]) * catalog.rule("/accumulator/range");
        }
    }
    if enabled && u.has("u2") {
        attack += number(&reserve(u, &Kind::Text("u2".into()), catalog)["charge"]) * 15.0;
    }
    if enabled && u.has("u6") && u.weapon("u5") {
        attack = 10.0 + extra_number(u, "attackBonus");
    }
    if u.weapon("u5") || u.weapon("s16") {
        range += 1.0;
    }
    if u.weapon("s2") || u.weapon("s15") {
        attack += 20.0;
    }
    if u.weapon("u11") {
        attack += 5.0;
    }
    if enabled
        && u.has("12")
        && s.units.iter().any(|v| {
            v.side() != u.owner
                && v.id != u.id
                && cells(u, u.at())
                    .iter()
                    .any(|&p| cells(v, v.at()).iter().any(|&q| distance(p, q) == 1.0))
        })
    {
        actions -= 1.0;
    }
    attack += u
        .effects
        .iter()
        .filter(|e| e["type"] == "attack" && s.active_effect(e, Some(u)))
        .map(|e| number(&e["amount"]))
        .sum::<f64>();
    if !u.any(&["10", "s7"]) && u.side() == u.owner {
        let sources = s
            .units
            .iter()
            .filter(|v| {
                v.has("sage")
                    && v.owner == u.owner
                    && !v.silenced
                    && attack_path(
                        s,
                        v,
                        &Target::from(u),
                        catalog[&v.kind.key()].range
                            + extra_number(v, "rangeBonus")
                            + if v.weapon("u5") || v.weapon("s16") {
                                1.0
                            } else {
                                0.0
                            },
                        None,
                        false,
                    )
                    .is_some()
            })
            .count();
        attack += sources as f64 * catalog.rule("/sageAuraAttack");
        attack += banner_count(s, u.owner) as f64 * 10.0;
    }
    if s.effect(u, "inner-fire") {
        attack = u.hp;
    }
    let negative = d.attack < 0.0 || (enabled && u.has("s14"));
    if negative {
        attack = -20.0;
    }
    let reserve = attack_charge(u, catalog).map(|k| reserve(u, &k, catalog));
    let half_locked = reserve
        .as_ref()
        .is_some_and(|r| r["chargeType"] != "attack" || number(&r["readyCharge"]) < 1.0);
    let remaining = if locked || half_locked || (enabled && u.has("firelord")) {
        0.0
    } else if u.mode == "attack" {
        (actions - u.shots).max(0.0)
    } else if u.mode == "none" && (left > 0.0 || u.bonus_attacks > 0.0) {
        actions
    } else {
        0.0
    };
    Stats {
        attack: if negative { -20.0 } else { attack.max(0.0) },
        range: range.max(0.0),
        actions: actions.max(0.0),
        remaining,
        movement,
        mode: u.mode.clone(),
        operation_limit: (if enabled && u.has("u27") && age == 1.0 {
            2.0
        } else {
            1.0
        }) + extra_number(u, "extraOperations"),
        operations_left: left,
        frozen,
        stunned,
        sleeping,
    }
}

pub fn prune_siphons(s: &mut State, catalog: &Catalog) {
    s.siphons = s
        .siphons
        .iter()
        .filter(|link| {
            let Some(u) = s
                .units
                .iter()
                .find(|u| Some(u.id.as_str()) == link["sourceId"].as_str())
            else {
                return false;
            };
            if u.silenced {
                return false;
            }
            let a = link["fromId"]
                .as_str()
                .and_then(|id| crate::geometry::find_target(s, id));
            let b = link["toId"]
                .as_str()
                .and_then(|id| crate::geometry::find_target(s, id));
            let range = stats(s, u, catalog).range;
            a.zip(b).is_some_and(|(a, b)| {
                attack_path(s, u, &a, range, None, false).is_some()
                    && attack_path(s, u, &b, range, None, false).is_some()
            })
        })
        .cloned()
        .collect();
}
