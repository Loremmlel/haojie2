//! 实际回合与个人时钟分别推进。结束效果保留反应队列，清空后才切换玩家。
use crate::damage::{Source, alive, area_damage, damage, freeze, heal, kill};
use crate::geometry::{Target, can_deploy, covers, empty_for, find_target, neighbors, targets};
use crate::model::{Catalog, Failure, Point, State, Unit, ensure, number};
use crate::preparation::{hand, hand_mut, kind};
use crate::resolution::{Resolution, faction, reset_unit, template};
use crate::stats::prune_siphons;
use serde_json::{Value, json};

pub fn all_cells() -> impl Iterator<Item = Point> {
    (1..=13).flat_map(|y| {
        (1..=9).map(move |x| Point {
            x: x as f64,
            y: y as f64,
        })
    })
}
pub fn stored(catalog: &Catalog, k: &str) -> bool {
    let d = &catalog[k];
    d.spell.is_some() || d.weapon.is_some() || d.aura || d.tier == "shrine"
}
pub fn ice_marks(
    s: &mut State,
    only: Option<&str>,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    for mark in s.extra["iceMarks"].as_array().unwrap().clone() {
        if only.is_some_and(|id| mark["sourceId"] != id) {
            continue;
        }
        let u = s.units.iter().find(|u| mark["sourceId"] == u.id).cloned();
        if u.as_ref()
            .is_some_and(|u| number(&mark["due"]) > s.ply + u.offset)
        {
            continue;
        }
        s.extra
            .get_mut("iceMarks")
            .unwrap()
            .as_array_mut()
            .unwrap()
            .retain(|m| m["id"] != mark["id"]);
        let Some(u) = u.filter(|u| !u.silenced) else {
            continue;
        };
        ctx.token += 1;
        let p = Point {
            x: number(&mark["x"]),
            y: number(&mark["y"]),
        };
        if let Some(v) = s
            .units
            .iter()
            .find(|v| covers(v, p))
            .filter(|v| v.side() != u.owner)
            .cloned()
        {
            freeze(
                s,
                &Target::from(&v),
                &Source::new(&u, "skill"),
                5.0,
                catalog,
                ctx,
            )?;
        }
    }
    Ok(())
}
pub fn advance(
    s: &mut State,
    id: &str,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let ply = s.ply;
    let u = s.unit_mut(id).unwrap();
    u.offset += 2.0;
    reset_unit(ply, u);
    let u = u.clone();
    ice_marks(s, Some(id), catalog, ctx)?;
    ctx.emit(
        s,
        json!({"type":"skill","to":u.actor_event(),"owner":u.owner,"text":"独立推进回合"}),
        Some(format!(
            "{}提前进入自己的下一回合",
            catalog.by_kind(&u.kind).name
        )),
    );
    Ok(())
}
pub fn begin(s: &mut State, catalog: &Catalog, ctx: &mut Resolution) -> Result<(), Failure> {
    let owner = s.active;
    let key = owner.to_string();
    *s.turns.get_mut(&key).unwrap() += 1.0;
    crate::shrines::rebuild(s, catalog, ctx);
    for old in s.units.clone() {
        let ply = s.ply;
        let u = s.unit_mut(&old.id).unwrap();
        let offset = u.offset;
        u.effects
            .retain(|e| e.until > ply + if e.global == Some(true) { 0.0 } else { offset });
        let u = u.clone();
        if u.extra.get("expiresAt").is_some_and(|v| number(v) <= ply) {
            kill(s, &u, &Source::effect(0, "expire"), catalog, ctx)?;
        }
        if old.owner == owner
            && let Some(u) = s.unit_mut(&old.id)
        {
            reset_unit(ply, u);
        }
    }
    for p in ["1", "2"] {
        s.base_effects
            .get_mut(p)
            .unwrap()
            .retain(|e| e.until > s.ply);
    }
    for c in hand(s).to_vec() {
        if c.get("expiresAt")
            .is_some_and(|v| number(v) <= s.turns[&key])
        {
            ctx.emit(
                s,
                json!({"type":"skill","owner":owner,"text":"储存到期"}),
                Some(format!("{}已过期", catalog[&kind(&c["kind"]).key()].name)),
            );
        }
    }
    let turn = s.turns[&key];
    hand_mut(s).retain(|c| c.get("expiresAt").is_none_or(|v| number(v) > turn));
    for h in s.extra["hazards"].as_array().unwrap().clone() {
        if number(&h["due"]) > s.ply {
            continue;
        }
        s.extra
            .get_mut("hazards")
            .unwrap()
            .as_array_mut()
            .unwrap()
            .retain(|v| v["id"] != h["id"]);
        ctx.token += 1;
        let owner = number(&h["owner"]) as usize;
        let hit = |p: Point| {
            if h["axis"] == "row" {
                p.y == number(&h["line"])
            } else {
                p.x == number(&h["line"])
            }
        };
        let victims: Vec<_> = targets(s)
            .into_iter()
            .filter(|t| {
                t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != owner
                    && t.footprint().into_iter().any(&hit)
            })
            .collect();
        let prev=ctx.facts.replace(json!({"action":"storm","stage":"trigger","ability":"u9","actor":find_target(s,&format!("base-{owner}")).unwrap().actor(),"area":all_cells().filter(|p|hit(*p)).collect::<Vec<_>>()}));
        ctx.emit(
            s,
            json!({"type":"skill","owner":owner,"text":"烈焰风暴 · 再临"}),
            None,
        );
        area_damage(
            s,
            &victims,
            |p| if hit(p) { 20.0 } else { 0.0 },
            &Source::effect(owner, "spell"),
            catalog,
            ctx,
        )?;
        ctx.facts = prev;
    }
    ice_marks(s, None, catalog, ctx)?;
    s.deploy_rows[key.clone()] = json!(crate::geometry::deployment_rows(s, owner));
    prune_siphons(s, catalog);
    s.phase = if crate::synthesis::available(s, catalog) {
        "synthesis"
    } else {
        "summon"
    }
    .into();
    s.summon_slots = 2.0 + number(&s.extra["bonus"][&key]);
    s.extra.get_mut("bonus").unwrap()[&key] = json!(0);
    if s.extra.get("mode") == Some(&json!("shrine")) {
        s.extra.insert("regularSummons".into(), json!(2));
    }
    crate::movement::sync_banners(s, catalog);
    crate::shrines::capture_clock(s);
    ctx.emit(s,json!({"type":"turn","owner":owner,"text":format!("{} · {}",faction(owner),if s.phase=="synthesis"{"合成阶段"}else{"召唤阶段"})}),Some(format!("{}第{}回合开始，可召唤{}次",faction(owner),s.turns[&key],s.summon_slots)));
    Ok(())
}
pub fn switch(s: &mut State, catalog: &Catalog, ctx: &mut Resolution) -> Result<(), Failure> {
    s.active = 3 - s.active;
    s.ply += 1.0;
    begin(s, catalog, ctx)
}
pub fn prepare_end(s: &State, catalog: &Catalog) -> Result<Vec<Value>, Failure> {
    ensure(s.phase == "play", "请先完成召唤并进入行动阶段。")?;
    for u in &s.units {
        ensure(
            !(u.has("u12") || u.has("u12p")) || u.mode != "move" || empty_for(s, u, catalog),
            "冲撞棋子必须离开棋子、地标或基地占位后才能结束回合。",
        )?;
    }
    let discarded = hand(s)
        .iter()
        .filter(|c| !stored(catalog, &kind(&c["kind"]).key()))
        .cloned()
        .collect::<Vec<_>>();
    for c in &discarded {
        let k = kind(&c["kind"]).key();
        let ghost = template(
            s,
            &k,
            s.active,
            Point { x: 1.0, y: 1.0 },
            "preview",
            catalog,
        );
        if all_cells().any(|p| can_deploy(s, &ghost, p, catalog)) {
            return Err(Failure::InvalidOwned(format!(
                "请先部署{}，随从不能储存。",
                catalog[&k].name
            )));
        }
    }
    Ok(discarded)
}
pub fn end(s: &mut State, catalog: &Catalog, ctx: &mut Resolution) -> Result<(), Failure> {
    for c in prepare_end(s, catalog)? {
        let k = kind(&c["kind"]).key();
        ctx.emit(
            s,
            json!({"type":"skill","owner":s.active,"text":"无处部署"}),
            Some(format!("{}无合法格，自动弃置", catalog[&k].name)),
        );
    }
    hand_mut(s).retain(|c| stored(catalog, &kind(&c["kind"]).key()));
    for u in s.pieces().cloned().collect::<Vec<_>>() {
        for e in &u.effects {
            if (e.kind == "burn" || e.kind == "freeze") && s.active_effect(e, Some(&u)) {
                ctx.token += 1;
                let mut source = Source::effect(e.owner, "status");
                source.unit = s
                    .units
                    .iter()
                    .find(|v| e.source_id.as_deref() == Some(v.id.as_str()))
                    .cloned();
                damage(
                    s,
                    &Target::from(&u),
                    e.amount.unwrap_or(5.0),
                    &source,
                    catalog,
                    ctx,
                )?;
            }
        }
    }
    prune_siphons(s, catalog);
    for link in s.siphons.clone() {
        let source = s.units.iter().find(|u| link["sourceId"] == u.id).cloned();
        let from = find_target(s, link["fromId"].as_str().unwrap());
        let to = find_target(s, link["toId"].as_str().unwrap());
        if let (Some(source), Some(from), Some(to)) = (source, from, to) {
            ctx.token += 1;
            let prev=ctx.facts.replace(json!({"action":"siphon","stage":"trigger","actor":from.actor(),"subject":to.actor()}));
            let mut packet = Source::new(&source, "skill");
            packet.owner = number(&link["owner"]) as usize;
            damage(s, &from, 20.0, &packet, catalog, ctx)?;
            heal(s, &to, 20.0, catalog, ctx)?;
            ctx.facts = prev;
        }
    }
    for lord in s.units.clone() {
        if !alive(s, &lord.id)
            || lord.owner != s.active
            || !lord.has("firelord")
            || lord.silenced
            || s.effect(&lord, "freeze")
        {
            continue;
        }
        let radius = catalog.rule("/firelord/radius");
        let in_area = |p: Point| (p.x - lord.x).abs() <= radius && (p.y - lord.y).abs() <= radius;
        let enemies: Vec<_> = targets(s)
            .into_iter()
            .filter(|t| t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != lord.owner)
            .collect();
        let mut candidates: Vec<_> = enemies
            .iter()
            .filter(|t| t.footprint().into_iter().any(&in_area))
            .collect();
        let hp = |t: &Target| {
            t.unit
                .as_ref()
                .map(|u| u.hp)
                .unwrap_or_else(|| number(&s.bases[t.owner.to_string()]))
        };
        candidates.sort_by(|a, b| {
            hp(b)
                .total_cmp(&hp(a))
                .then(a.at.y.total_cmp(&b.at.y))
                .then(a.at.x.total_cmp(&b.at.x))
                .then(a.id.cmp(&b.id))
        });
        let Some(target) = candidates.first() else {
            continue;
        };
        let mut points: Vec<_> = target
            .footprint()
            .into_iter()
            .filter(|p| in_area(*p))
            .collect();
        points.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
        let impact = points[0];
        let area: Vec<_> = std::iter::once(impact).chain(neighbors(impact)).collect();
        let mut victims: Vec<_> = enemies
            .iter()
            .filter(|t| t.footprint().contains(&impact))
            .cloned()
            .collect();
        for t in &enemies {
            if !victims.iter().any(|v| v.id == t.id)
                && t.footprint().iter().any(|p| area[1..].contains(p))
            {
                victims.push(t.clone());
            }
        }
        ctx.token += 1;
        let prev = ctx
            .facts
            .replace(json!({"action":"judgement","actor":lord.actor_event(),"area":area}));
        ctx.emit(s,json!({"type":"attack","from":lord.actor_event(),"to":impact,"owner":lord.owner,"text":"末日审判","ultimate":true}),None);
        area_damage(
            s,
            &victims,
            |p| {
                if p == impact {
                    catalog.rule("/firelord/damage")
                } else if area.contains(&p) {
                    catalog.rule("/firelord/splash")
                } else {
                    0.0
                }
            },
            &Source::new(&lord, "skill"),
            catalog,
            ctx,
        )?;
        ctx.facts = prev;
    }
    crate::shrines::end(s, catalog, ctx)?;
    if !s.pending.is_empty() {
        s.phase = "play".into();
        s.summon_slots = -1.0;
        ctx.emit(s, json!({"type":"turn","text":"回合结束效果结算"}), None);
        return Ok(());
    }
    if number(&s.bases["1"]) > 0.0 && number(&s.bases["2"]) > 0.0 {
        switch(s, catalog, ctx)?;
    }
    Ok(())
}
