use crate::damage::{Source, alive, damage, heal, protected, set_effects};
use crate::geometry::{Target, attack_path, find_target, top_target};
use crate::model::{Catalog, Command, Failure, Kind, State, Unit, ensure, extra_number, number};
use crate::movement::{actor, consume, finish, reserve, stage, sync};
use crate::resolution::{Resolution, active_target, add_effect, normalize_guards, terminal};
use crate::stats::{Stats, attack_charge, stats};
use serde_json::{Value, json};

#[derive(Default, Clone)]
pub struct Options {
    pub reactive: bool,
    pub unlimited: bool,
    pub force_hostile: bool,
    pub mode: Option<String>,
    pub direction: Option<String>,
    pub path: Option<Vec<crate::model::Point>>,
    pub no_pierce: bool,
    pub amount: Option<f64>,
    pub weapon_first: Option<bool>,
}
struct Prepared {
    ally: bool,
    healing: bool,
    stats: Stats,
    path: Vec<crate::model::Point>,
    can_pierce: bool,
}
pub fn healing_attack(u: &Unit, catalog: &Catalog) -> bool {
    catalog[&u.kind.key()].attack < 0.0
        || u.signed()
        || (!u.silenced && u.any(&["2", "u21", "s14"]))
}
fn prepare(
    s: &State,
    u: &Unit,
    t: &Target,
    o: &Options,
    catalog: &Catalog,
) -> Result<Prepared, Failure> {
    let ally = t
        .unit
        .as_ref()
        .map(|v| v.side() == u.owner)
        .unwrap_or(t.owner == u.owner);
    let healing = !o.force_hostile
        && (catalog[&u.kind.key()].attack < 0.0
            || (!u.silenced && u.has("s14"))
            || if u.signed() {
                o.mode.as_deref() == Some("heal") || (o.mode.is_none() && ally)
            } else {
                healing_attack(u, catalog) && ally
            });
    ensure(
        o.mode
            .as_deref()
            .is_none_or(|v| ["heal", "damage"].contains(&v)),
        "请选择伤害或治疗。",
    )?;
    ensure(!healing || t.unit.is_some(), "治疗攻击只能选择棋子。")?;
    ensure(
        o.mode.as_deref() != Some("heal")
            || healing_attack(u, catalog)
            || catalog[&u.kind.key()].attack < 0.0,
        "该棋子不能选择治疗。",
    )?;
    let friendly = t.unit.as_ref().is_some_and(|v| {
        catalog[&u.kind.key()].attack < 0.0
            || u.signed()
            || (!u.silenced && u.any(&["2", "u21", "sage", "s5", "s14"]))
            || (!v.silenced && v.any(&["16", "s6"]))
    });
    ensure(
        !ally || o.force_hostile || (t.unit.is_some() && (friendly || healing)),
        "该棋子不能对所选友方进行这种攻击。",
    )?;
    ensure(
        t.id != u.id || healing,
        "不能通过普通攻击自杀；玉碎使用独立命令。",
    )?;
    ensure(
        (t.id == u.id && healing)
            || top_target(s, t, catalog)
            || (u.piercing()
                && t.unit
                    .as_ref()
                    .is_some_and(|v| catalog[&v.kind.key()].landmark.is_some())),
        "非穿透攻击优先命中地标上的友方棋子或叠放栈顶。",
    )?;
    let computed = stats(s, u, catalog);
    ensure(
        u.silenced || !u.has("firelord"),
        "炎魔之王不能普通攻击；沉默会移除此限制。",
    )?;
    if !o.reactive {
        if !o.no_pierce
            && let Some(kind) = attack_charge(u, catalog)
        {
            let r = reserve(u, &kind, catalog);
            ensure(
                number(&r["readyCharge"]) >= 1.0 && r["chargeType"] == "attack",
                "半速攻击需要在回合开始已有1层攻击蓄力。",
            )?;
        }
        ensure(
            !u.has("4")
                || u.silenced
                || number(&reserve(u, &Kind::Number(4), catalog)["readyCharge"]) >= 2.0,
            "定炮回合开始至少有2层蓄力才可开炮。",
        )?;
        ensure(
            !u.has("9")
                || u.silenced
                || !u.extra["attacked"]
                    .as_array()
                    .unwrap()
                    .contains(&json!(t.id)),
            "射手不能重复攻击本回合的同一目标。",
        )?;
    }
    let can_pierce = u.piercing() && !ally && !healing;
    let limit = if o.unlimited { 117.0 } else { computed.range };
    let path = if let Some(path) = &o.path {
        ensure(
            can_pierce && crate::geometry::valid_attack_route(u, path, limit),
            "穿透路径不合法：须从自身边缘逐格延伸，不得回绕、超距或穿过敌方基地。",
        )?;
        ensure(
            o.no_pierce
                || crate::geometry::piercing_targets(s, u, path, catalog)
                    .iter()
                    .any(|(v, _)| v.id == t.id),
            "路径必须命中选定敌方目标。",
        )?;
        Some(path.clone())
    } else if can_pierce && !u.weapon("u28") {
        let mut rays = vec![];
        for from in crate::geometry::cells(u, u.at()) {
            for to in t.footprint() {
                let length = crate::geometry::distance(from, to);
                if (from.x == to.x || from.y == to.y) && length > 0.0 && length <= limit {
                    rays.push((from, to, length));
                }
            }
        }
        rays.sort_by(|a, b| a.2.total_cmp(&b.2));
        let mut path = if t.footprint().iter().any(|p| crate::geometry::covers(u, *p)) {
            Some(vec![t.at])
        } else {
            None
        };
        if let Some((from, to, _)) = rays.first() {
            let dx = if to.x == from.x {
                0.0
            } else {
                (to.x - from.x).signum()
            };
            let dy = if to.y == from.y {
                0.0
            } else {
                (to.y - from.y).signum()
            };
            let mut route = vec![*from];
            for i in 1..=limit.floor() as usize {
                let p = crate::model::Point {
                    x: from.x + dx * i as f64,
                    y: from.y + dy * i as f64,
                };
                if !crate::geometry::inside(p) {
                    break;
                }
                route.push(p);
                if p == crate::geometry::base_point(3 - u.owner) {
                    break;
                }
            }
            path = Some(route);
        }
        path
    } else {
        attack_path(s, u, t, limit, o.direction.as_deref(), can_pierce)
    }
    .ok_or(Failure::Invalid("目标不在射程内，或所选攻击路径被阻挡。"))?;
    Ok(Prepared {
        ally,
        healing,
        stats: computed,
        path,
        can_pierce,
    })
}
fn attack_roll(
    s: &mut State,
    u: &Unit,
    mut amount: f64,
    base: bool,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<f64, Failure> {
    if u.silenced {
        return Ok(amount);
    }
    for k in u.kinds() {
        if k.is("1") {
            let r = s.random(ctx.preview)?;
            amount += if r < catalog.rule("/charger/heavyChance") {
                catalog.rule("/charger/heavyBonus")
            } else if r < catalog.rule("/charger/criticalChance") {
                catalog.rule("/charger/bonus")
            } else {
                0.0
            };
        }
        if k.is("u1") {
            let r = s.random(ctx.preview)?;
            amount = if r < catalog.rule("/superCritical/lethalChance") {
                catalog.rule("/superCritical/lethalDamage")
            } else if r < catalog.rule("/superCritical/lethalChance")
                + catalog.rule("/superCritical/doubleChance")
            {
                amount * 2.0
            } else {
                amount
            };
        }
        if k.is("u8") {
            let r = s.random(ctx.preview)?;
            if r < vampire(u, catalog).min(1.0) {
                amount *= 2.0;
            }
        }
        if k.is("u27") && base {
            amount = catalog.rule("/minerBaseDamage");
        }
    }
    Ok(amount)
}
fn vampire(u: &Unit, catalog: &Catalog) -> f64 {
    catalog.rule("/vampire/base") + catalog.rule("/vampire/perKill") * extra_number(u, "kills")
}
fn convert_target(s: &mut State, id: &str, owner: usize) {
    let turn = s.turns[&owner.to_string()];
    let ply = s.ply;
    if let Some(v) = s.unit_mut(id) {
        v.owner = owner;
        v.offset = 0.0;
        v.born = turn - if v.has("23") { 1.0 } else { 0.0 };
        v.effects.clear();
        crate::resolution::reset_unit(ply, v);
        v.operations = 1.0;
    }
}
fn knockback(
    s: &mut State,
    u: &Unit,
    t: &Target,
    path: &[crate::model::Point],
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    use crate::geometry::{can_place, cells, covers};
    use crate::model::Point;
    let Some(v) = s.unit(&t.id).cloned().filter(|_| path.len() >= 2) else {
        return Ok(());
    };
    if protected(s, t, &Source::new(u, "skill"), catalog, ctx)? {
        return Ok(());
    }
    let a = path[path.len() - 2];
    let b = path[path.len() - 1];
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let first = Point {
        x: v.x + dx,
        y: v.y + dy,
    };
    let second = Point {
        x: v.x + 2.0 * dx,
        y: v.y + 2.0 * dy,
    };
    let mut behind: Vec<Unit> = vec![];
    for p in cells(&v, first).into_iter().chain(cells(&v, second)) {
        for other in s.units.iter().filter(|x| x.id != v.id && covers(x, p)) {
            if !behind.iter().any(|x| x.id == other.id) {
                behind.push(other.clone());
            }
        }
    }
    if behind.len() > 1 || behind.iter().any(|v| v.size > 1.0) {
        return Ok(());
    }
    if let Some(follower) = behind.first() {
        let to = Point {
            x: follower.x + dx,
            y: follower.y + dy,
        };
        let mut view = s.clone();
        view.units.retain(|v| v.id != follower.id);
        let can_first = can_place(&view, &v, first, catalog);
        let mut view = s.clone();
        view.units.retain(|x| x.id != v.id);
        if can_first && can_place(&view, follower, to, catalog) {
            ctx.emit(s,json!({"type":"move","from":follower.actor_event(),"to":to,"unitId":follower.id,"owner":follower.owner,"text":"连带击退"}),None);
            let f = s.unit_mut(&follower.id).unwrap();
            f.x = to.x;
            f.y = to.y;
            ctx.emit(s,json!({"type":"move","from":v.actor_event(),"to":first,"unitId":v.id,"owner":v.owner,"text":"击退"}),None);
            let v = s.unit_mut(&v.id).unwrap();
            v.x = first.x;
            v.y = first.y;
        }
        return Ok(());
    }
    if cells(&v, second)
        .iter()
        .any(|p| !crate::geometry::inside(*p))
    {
        damage(s, t, 30.0, &Source::new(u, "skill"), catalog, ctx)?;
        return Ok(());
    }
    if can_place(s, &v, first, catalog) && can_place(s, &v, second, catalog) {
        ctx.emit(s,json!({"type":"move","from":v.actor_event(),"to":second,"unitId":v.id,"owner":v.owner,"text":"击退"}),None);
        let v = s.unit_mut(&v.id).unwrap();
        v.x = second.x;
        v.y = second.y;
    }
    Ok(())
}

/// 在命令副本内结算一次攻击与后续被动；反击复用此入口，按对记录阻止递归环。
/// 随机预检和未移植能力向外返回错误，由命令入口丢弃所有中间变化。
pub fn perform(
    s: &mut State,
    u: &Unit,
    t: &Target,
    o: &Options,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let p = prepare(s, u, t, o, catalog)?;
    let hit_start = ctx.attack_hits.len();
    let previous = ctx.enter(
        u.actor_event(),
        t.actor(),
        if p.healing { "mend" } else { "attack" },
    );
    let result = resolve(s, u, t, &p, o, catalog, ctx);
    ctx.facts = previous;
    result?;
    if !o.no_pierce
        && let Some(current) = s.unit_mut(&u.id)
    {
        if current.has("15") {
            consume(current, &Kind::Number(15), catalog);
        }
        if let Some(k) = attack_charge(current, catalog) {
            consume(current, &k, catalog);
        }
    }
    if !o.no_pierce && !o.reactive && !u.silenced {
        if u.has("s3") {
            let convert = s.random(ctx.preview)? < 1.0 / 3.0;
            let victim = s.unit(&t.id).cloned();
            if convert
                && !u.has("5")
                && victim.as_ref().is_some_and(|v| {
                    v.side() == 3 - u.owner && catalog[&v.kind.key()].tier != "shrine"
                })
                && ctx.attack_hits[hit_start..]
                    .iter()
                    .any(|(actor, id, actual)| actor == &u.id && id == &t.id && *actual > 0.0)
                && !protected(s, t, &Source::new(u, "skill"), catalog, ctx)?
            {
                convert_target(s, &t.id, u.owner);
                let v = s.unit(&t.id).unwrap();
                ctx.emit(s,json!({"type":"skill","to":v.actor_event(),"owner":u.owner,"action":"conversion","text":"CX · 策反"}),None);
            }
            if s.random(ctx.preview)? < 0.25 {
                s.summon_slots += 1.0;
                ctx.emit(
                    s,
                    json!({"type":"summon","owner":u.owner,"text":"CX · 本回合额外召唤+1"}),
                    None,
                );
            }
        }
        if u.has("s12") && s.random(ctx.preview)? < 3.0 / 5.0 && alive(s, &u.id) {
            let current = s.unit_mut(&u.id).unwrap();
            current.extra.insert(
                "extraOperations".into(),
                json!(extra_number(current, "extraOperations") + 1.0),
            );
            let actor = current.actor_event();
            ctx.emit(
                s,
                json!({"type":"skill","to":actor,"owner":u.owner,"text":"先攻 · 额外完整操作+1"}),
                None,
            );
        }
    }
    Ok(())
}
fn resolve(
    s: &mut State,
    u: &Unit,
    t: &Target,
    p: &Prepared,
    o: &Options,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    if p.can_pierce && !o.no_pierce && p.path.len() > 1 {
        for (victim, prefix) in crate::geometry::piercing_targets(s, u, &p.path, catalog) {
            if alive(s, &u.id) && (victim.unit.is_none() || alive(s, &victim.id)) {
                let current = s.unit(&u.id).unwrap().clone();
                let victim = active_target(s, &victim);
                perform(
                    s,
                    &current,
                    &victim,
                    &Options {
                        direction: None,
                        path: Some(prefix),
                        no_pierce: true,
                        amount: Some(p.stats.attack),
                        weapon_first: Some(
                            current.extra.get("weaponFirstUsed") != Some(&json!(true)),
                        ),
                        ..o.clone()
                    },
                    catalog,
                    ctx,
                )?;
            }
        }
        if u.weapon("u11")
            && let Some(u) = s.unit_mut(&u.id)
        {
            u.extra.insert("weaponFirstUsed".into(), json!(true));
        }
        return Ok(());
    }
    ctx.retaliations.insert(format!("{}>{}", u.id, t.id));
    ctx.emit(s,json!({"type":"attack","from":u.actor_event(),"to":t.actor(),"path":p.path,"unitId":u.id,"owner":u.owner,"text":if p.healing{"治疗"}else{"攻击"},"ultimate":catalog[&u.kind.key()].tier!="normal"}),None);
    let skill = Source::new(u, "skill");
    if p.healing && !o.force_hostile {
        if !protected(s, t, &skill, catalog, ctx)? {
            let amount = if catalog[&u.kind.key()].attack < 0.0 || u.has("s14") {
                20.0
            } else if u.signed() {
                catalog[&u.kind.key()].attack.abs()
            } else if u.has("2") {
                20.0
            } else {
                25.0
            };
            heal(s, t, amount, catalog, ctx)?;
            if t.unit.is_some() && alive(s, &t.id) && u.weapon("s16") {
                add_effect(
                    s,
                    &t.id,
                    "attack",
                    u.owner,
                    2.0,
                    Some(-15.0),
                    Some(&u.id),
                    true,
                );
            }
        }
        return Ok(());
    }
    if !p.ally
        && let Some(victim) = &t.unit
        && s.effect(u, "execute")
    {
        let effect = u
            .effects
            .iter()
            .find(|e| e["type"] == "execute" && s.active_effect(e, Some(u)))
            .unwrap()
            .clone();
        if let Some(u) = s.unit_mut(&u.id)
            && let Some(i) = u.effects.iter().position(|e| *e == effect)
        {
            u.effects.remove(i);
        }
        if !protected(s, t, &skill, catalog, ctx)? {
            let start = s.events.len();
            crate::damage::kill(s, victim, &skill, catalog, ctx)?;
            if let Some(e) = s.events[start..]
                .iter_mut()
                .find(|e| e["type"] == "death" && e["unitId"] == t.id)
            {
                e["action"] = json!("execution");
                e["stage"] = json!("trigger");
            }
        }
        return Ok(());
    }
    let target_effects = t
        .unit
        .as_ref()
        .map(|v| v.effects.clone())
        .unwrap_or_else(|| {
            s.extra["baseEffects"][t.owner.to_string()]
                .as_array()
                .unwrap()
                .clone()
        });
    let marks: Vec<_> = if !p.ally && !u.has("10") {
        target_effects
            .iter()
            .filter(|e| {
                e["type"] == "mark"
                    && number(&e["owner"]) == u.owner as f64
                    && s.active_effect(e, t.unit.as_ref())
            })
            .cloned()
            .collect()
    } else {
        vec![]
    };
    if !marks.is_empty() {
        set_effects(
            s,
            t,
            target_effects
                .into_iter()
                .filter(|e| !marks.contains(e))
                .collect(),
        );
    }
    let amount = attack_roll(
        s,
        u,
        o.amount.unwrap_or(p.stats.attack),
        t.unit.is_none(),
        catalog,
        ctx,
    )?;
    let wounded = t
        .unit
        .as_ref()
        .map(|v| (v.max_hp - v.hp).max(0.0))
        .unwrap_or(300.0 - number(&s.bases[t.owner.to_string()]));
    let reflected = if !u.silenced && u.has("s6") {
        u.extra
            .get("receivedDamage")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|v| number(&v["ply"]) >= s.ply - 1.0)
            .map(|v| number(&v["amount"]))
            .sum()
    } else {
        0.0
    };
    let lifesteal = if u.silenced {
        0.0
    } else if u.has("slayer") {
        1.0
    } else if u.has("u8") {
        vampire(u, catalog)
    } else {
        0.0
    };
    let mut packet = Source::new(u, "attack");
    packet.path = p.path.clone();
    packet.credit_friendly = p.ally && (u.signed() || u.has("s5"));
    let attack_loss = damage(s, t, amount, &packet, catalog, ctx)?;
    ctx.attack_hits
        .push((u.id.clone(), t.id.clone(), attack_loss));
    let mut loss = attack_loss;
    if u.weapon("s15") && wounded > 0.0 {
        loss += damage(s, t, wounded, &packet, catalog, ctx)?;
    }
    if reflected > 0.0 {
        loss += damage(s, t, reflected, &packet, catalog, ctx)?;
    }
    if t.unit.is_some()
        && alive(s, &t.id)
        && u.weapon("s16")
        && !protected(s, t, &skill, catalog, ctx)?
    {
        add_effect(
            s,
            &t.id,
            "attack",
            u.owner,
            2.0,
            Some(-15.0),
            Some(&u.id),
            true,
        );
    }
    if u.has("10") && !u.silenced && !p.ally {
        let e = json!({"type":"mark","owner":u.owner,"sourceId":u.id,"from":s.ply,"until":s.ply+2.0,"global":true});
        let full = t
            .unit
            .as_ref()
            // TS 保留离场对象的扣血结果；不能拿攻击前快照把已死目标误判为满血。
            .map(|v| {
                s.unit(&v.id)
                    .is_some_and(|current| current.hp >= current.max_hp)
            })
            .unwrap_or(number(&s.bases[t.owner.to_string()]) >= 300.0);
        if full {
            loss += damage(
                s,
                t,
                catalog.rule("/catapultMarkDamage"),
                &packet,
                catalog,
                ctx,
            )?;
        } else {
            if let Some(v) = s.unit_mut(&t.id) {
                v.effects.push(e);
            } else if t.unit.is_none() {
                s.extra.get_mut("baseEffects").unwrap()[t.owner.to_string()]
                    .as_array_mut()
                    .unwrap()
                    .push(e);
            }
            ctx.emit(s,json!({"type":"skill","to":t.actor(),"owner":u.owner,"action":"mark","stage":"apply","text":"标记"}),None);
        }
    }
    for _ in marks {
        let start = s.events.len();
        loss += damage(
            s,
            t,
            catalog.rule("/catapultMarkDamage"),
            &Source::new(u, "status"),
            catalog,
            ctx,
        )?;
        if let Some(e) = s.events[start..]
            .iter_mut()
            .find(|e| e["type"] == "damage" && e["unitId"] == t.id)
        {
            e["action"] = json!("mark");
            e["stage"] = json!("trigger");
        }
    }
    let drain = lifesteal
        + if u.weapon("u11")
            && o.weapon_first
                .unwrap_or(u.extra.get("weaponFirstUsed") != Some(&json!(true)))
        {
            1.0
        } else {
            0.0
        };
    if loss > 0.0 && drain > 0.0 {
        let start = s.events.len();
        heal(s, &Target::from(u), loss * drain, catalog, ctx)?;
        for e in &mut s.events[start..] {
            if e["type"] == "heal" {
                e["action"] = json!("siphon");
                e["actor"] = t.actor();
                e["from"] = json!(t.at);
            }
        }
    }
    if let Some(current) = s.unit_mut(&u.id) {
        if u.weapon("u11") && o.weapon_first.is_none() {
            current.extra.insert("weaponFirstUsed".into(), json!(true));
        }
        if !u.silenced {
            for kind in [Kind::Text("u2".into()), Kind::Number(4)] {
                if u.has(&kind.key()) {
                    consume(current, &kind, catalog);
                }
            }
        }
    }
    if t.unit.is_some() && alive(s, &t.id) && !p.ally {
        let victim = active_target(s, t).unit.unwrap();
        if s.effect(u, "convert") && victim.side() != 0 && !u.has("5") && attack_loss > 0.0 {
            let effect = u
                .effects
                .iter()
                .find(|e| e["type"] == "convert" && s.active_effect(e, Some(u)))
                .unwrap()
                .clone();
            if let Some(u) = s.unit_mut(&u.id)
                && let Some(i) = u.effects.iter().position(|e| *e == effect)
            {
                u.effects.remove(i);
            }
            if !protected(s, t, &skill, catalog, ctx)? {
                convert_target(s, &t.id, u.owner);
                let v = s.unit(&t.id).unwrap();
                ctx.emit(s,json!({"type":"skill","to":v.actor_event(),"owner":u.owner,"action":"conversion","stage":"trigger","text":"策反"}),Some(format!("{}加入{}",catalog[&v.kind.key()].name,crate::resolution::faction(u.owner))));
                return Ok(());
            }
        }
        if !u.silenced && u.has("u4") && !protected(s, t, &skill, catalog, ctx)? {
            s.unit_mut(&t.id).unwrap().silenced = true;
            add_effect(s, &t.id, "stun", u.owner, 2.0, None, Some(&u.id), false);
            ctx.emit(s,json!({"type":"skill","to":t.actor(),"owner":u.owner,"action":"silence","stage":"trigger","text":"沉默 · 眩晕"}),None);
        }
        if u.weapon("u5") && !victim.weapon("u28") && !protected(s, t, &skill, catalog, ctx)? {
            s.unit_mut(&t.id)
                .unwrap()
                .effects
                .retain(|e| e["type"] != "freeze");
            add_effect(
                s,
                &t.id,
                "freeze",
                u.owner,
                4.0,
                Some(5.0),
                Some(&u.id),
                false,
            );
            ctx.emit(s,json!({"type":"shield","to":t.actor(),"owner":u.owner,"action":"freeze","stage":"trigger","text":"冰冻 · 无法行动"}),None);
        }
        if (u.weapon("u28") || (!u.silenced && u.has("u6") && !u.weapon("u5")))
            && !protected(s, t, &skill, catalog, ctx)?
        {
            s.unit_mut(&t.id)
                .unwrap()
                .effects
                .retain(|e| e["type"] != "burn");
            add_effect(
                s,
                &t.id,
                "burn",
                u.owner,
                12.0,
                Some(5.0),
                Some(&u.id),
                false,
            );
            ctx.emit(s,json!({"type":"skill","to":t.actor(),"owner":u.owner,"action":"burn","stage":"trigger","text":"灼烧"}),None);
        }
        if !u.silenced && u.has("u20") {
            knockback(s, u, t, &p.path, catalog, ctx)?;
        }
        if u.has("formless") && !u.silenced && alive(s, &u.id) && !victim.has("5") {
            let source = s.unit(&u.id).unwrap();
            s.pending.push(json!({"kind":"hit-pull","owner":u.owner,"source":source,"targetId":t.id,"amount":0}));
        }
    } else if t.unit.as_ref().is_some_and(|v| v.side() != 0) && !p.ally && attack_loss > 0.0 {
        let effects: Vec<_> = s
            .unit(&u.id)
            .map(|u| {
                u.effects
                    .iter()
                    .filter(|e| !(e["type"] == "convert" && s.active_effect(e, Some(u))))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        if let Some(u) = s.unit_mut(&u.id) {
            u.effects = effects;
        }
    }
    Ok(())
}
pub fn retaliate(
    s: &mut State,
    u: &Unit,
    t: &Target,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let pair = format!("{}>{}", u.id, t.id);
    if ctx.retaliations.contains(&pair)
        || attack_path(s, u, t, stats(s, u, catalog).range, None, false).is_none()
    {
        return Ok(());
    }
    let options = Options {
        reactive: true,
        force_hostile: true,
        ..Default::default()
    };
    match prepare(s, u, t, &options, catalog) {
        Err(Failure::Invalid(_)) => return Ok(()),
        Err(e) => return Err(e),
        Ok(_) => {}
    }
    ctx.retaliations.insert(pair);
    perform(s, u, t, &options, catalog, ctx)
}
/// 只投影攻击者的本次操作；与 TS chooseMode + attackTarget 的预检边界相同。
fn command_attack(s: &State, c: &Command, catalog: &Catalog) -> Result<(Unit, Target), Failure> {
    let mut u = actor(s, c, catalog)?.clone();
    let computed = stats(s, &u, catalog);
    ensure(
        u.mode == "none" || u.mode == "attack",
        "本回合已选择另一操作模式，剩余攻击不能换成移动或技能。",
    )?;
    if u.mode == "none" {
        if computed.operations_left <= 0.0 && u.bonus_attacks > 0.0 {
            u.bonus_attacks -= 1.0;
            u.bonus_sequence = true;
        } else {
            ensure(computed.operations_left > 0.0, "本回合操作已用完。")?;
        }
        u.mode = "attack".into();
        u.shots = 0.0;
        u.moves = 0.0;
    }
    ensure(
        stats(s, &u, catalog).remaining > 0.0,
        "本次攻击操作次数已用完。",
    )?;
    if let Some(path) = &c.path {
        ensure(
            u.weapon("u28")
                && crate::geometry::valid_attack_route(&u, path, stats(s, &u, catalog).range),
            "所选穿透路径不合法。",
        )?;
    }
    let t = c
        .target_id
        .as_deref()
        .and_then(|id| find_target(s, id))
        .or_else(|| {
            if c.target_id.is_none() {
                c.path.as_ref().and_then(|path| {
                    crate::geometry::piercing_targets(s, &u, path, catalog)
                        .last()
                        .map(|(t, _)| t.clone())
                })
            } else {
                None
            }
        })
        .ok_or(Failure::Invalid("请选择有效的目标。"))?;
    Ok((u, t))
}
/// 只接受经 TS 重建的规范局面；阶段校验后复制，反应/攻击失败均不提交副本。
/// 成功后统一刷新保护、光环、虹吸与部署行，再检查终局，不驱动 UI 或选招。
pub fn apply(
    previous: &State,
    c: &Command,
    catalog: &Catalog,
    preview: bool,
) -> Result<State, Failure> {
    stage(previous, c, catalog)?;
    if preview && c.kind == "attack" {
        let (u, t) = command_attack(previous, c, catalog)?;
        prepare(
            previous,
            &u,
            &t,
            &Options {
                mode: c.mode.clone(),
                direction: c.direction.clone(),
                path: c.path.clone(),
                ..Default::default()
            },
            catalog,
        )?;
    }
    if preview
        && c.kind == "react"
        && let Some(r) = previous
            .pending
            .first()
            .filter(|r| r["kind"] == "hut-spawn")
    {
        crate::reactions::prepare_hut(
            previous,
            r,
            c,
            &crate::reactions::hut_points(previous, r, catalog),
        )?;
    }
    let mut s = previous.clone();
    s.events.clear();
    normalize_guards(&mut s);
    let mut ctx = Resolution {
        preview,
        ..Default::default()
    };
    if c.kind == "react" {
        crate::reactions::react(&mut s, c, catalog, &mut ctx)?;
    } else {
        let (u, t) = command_attack(&s, c, catalog)?;
        let mut hit_ids = vec![t.id.clone()];
        if u.piercing() && c.mode.as_deref() != Some("heal") {
            let path = c
                .path
                .clone()
                .or_else(|| {
                    attack_path(
                        &s,
                        &u,
                        &t,
                        stats(&s, &u, catalog).range,
                        c.direction.as_deref(),
                        true,
                    )
                })
                .unwrap_or_default();
            for (v, _) in crate::geometry::piercing_targets(&s, &u, &path, catalog) {
                if !hit_ids.contains(&v.id) {
                    hit_ids.push(v.id);
                }
            }
        }
        *s.unit_mut(&u.id).unwrap() = u.clone();
        perform(
            &mut s,
            &u,
            &t,
            &Options {
                mode: c.mode.clone(),
                direction: c.direction.clone(),
                path: c.path.clone(),
                ..Default::default()
            },
            catalog,
            &mut ctx,
        )?;
        if let Some(current) = s.unit_mut(&u.id) {
            current
                .extra
                .get_mut("attacked")
                .unwrap()
                .as_array_mut()
                .unwrap()
                .extend(hit_ids.into_iter().map(|id| json!(id)));
            current.shots += 1.0;
        }
        if let Some(current) = s.unit(&u.id).cloned()
            && current.shots >= stats(&s, &current, catalog).actions
        {
            finish(s.unit_mut(&u.id).unwrap());
        }
    }
    sync(&mut s, catalog);
    terminal(&mut s, &mut ctx);
    Ok(s)
}
