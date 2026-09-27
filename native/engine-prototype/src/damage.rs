use crate::geometry::{Target, attack_path, covers, find_target, frontal};
use crate::model::{Catalog, Failure, State, Unit, extra_number, number};
use crate::movement::sync_banners as sync;
use crate::resolution::{Resolution, active_target, add_unit, faction, guardian};
use crate::stats::stats;
use serde_json::{Value, json};

#[derive(Clone)]
pub struct Source {
    pub owner: usize,
    pub unit: Option<Unit>,
    pub kind: &'static str,
    pub path: Vec<crate::model::Point>,
    pub credit_friendly: bool,
    pub modified: bool,
}
impl Source {
    pub fn new(u: &Unit, kind: &'static str) -> Self {
        Self {
            owner: u.owner,
            unit: Some(u.clone()),
            kind,
            path: vec![],
            credit_friendly: false,
            modified: false,
        }
    }
}
pub fn alive(s: &State, id: &str) -> bool {
    s.units.iter().any(|u| u.id == id)
        || s.landmarks()
            .iter()
            .any(|u| u.id == id && !u.extra.contains_key("dormantSince"))
}
fn effects(s: &State, t: &Target) -> Vec<Value> {
    t.unit
        .as_ref()
        .map(|u| u.effects.clone())
        .unwrap_or_else(|| {
            s.extra["baseEffects"][t.owner.to_string()]
                .as_array()
                .unwrap()
                .clone()
        })
}
pub fn set_effects(s: &mut State, t: &Target, e: Vec<Value>) {
    if t.unit.is_some() {
        if let Some(u) = s.unit_mut(&t.id) {
            u.effects = e;
        }
    } else {
        s.extra.get_mut("baseEffects").unwrap()[t.owner.to_string()] = json!(e);
    }
}
pub fn protected(
    s: &mut State,
    t: &Target,
    source: &Source,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<bool, Failure> {
    let t = active_target(s, t);
    let Some(u) = &t.unit else {
        return Ok(false);
    };
    if source.owner == u.side() {
        return Ok(false);
    }
    if s.effect(u, "immune") {
        ctx.emit(s,json!({"type":"shield","to":t.actor(),"owner":t.owner,"action":"ward","stage":"blocked","text":"金身免疫"}),None);
        return Ok(true);
    }
    if !["spell", "skill"].contains(&source.kind) {
        return Ok(false);
    }
    let tower = s
        .units
        .iter()
        .find(|v| {
            v.has("u15")
                && v.owner == u.side()
                && !v.silenced
                && attack_path(s, v, &t, stats(s, v, catalog).range, None, false).is_some()
        })
        .cloned();
    if tower.is_some() {
        // 多包法术需要 token/protection 缓存；当前只对单次攻击附带技能开放后续扩展。
        return Err(Failure::Unsupported("protection-tower"));
    }
    Ok(false)
}
pub fn heal(
    s: &mut State,
    target: &Target,
    amount: f64,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let t = active_target(s, target);
    if t.unit.is_some() && !alive(s, &t.id) || s.aura(3 - t.owner, "s11") {
        return Ok(());
    }
    let gap = t
        .unit
        .as_ref()
        .map(|u| u.max_hp - u.hp)
        .unwrap_or(300.0 - number(&s.bases[t.owner.to_string()]));
    let gain = amount.min(gap).max(0.0);
    if let Some(u) = s.unit_mut(&t.id) {
        u.hp = ((u.hp + gain) * 1e6).round() / 1e6;
    } else if t.unit.is_none() {
        let p = t.owner.to_string();
        s.bases[&p] = json!(number(&s.bases[&p]) + gain);
    }
    if gain != 0.0 {
        ctx.emit(
            s,
            json!({"type":"heal","to":t.actor(),"amount":gain,"owner":t.owner}),
            Some(format!(
                "{}回复{}生命",
                t.unit
                    .as_ref()
                    .map(|u| catalog[&u.kind.key()].name.as_str())
                    .unwrap_or("基地"),
                gain
            )),
        );
    }
    let t = active_target(s, &t);
    let full = t
        .unit
        .as_ref()
        .map(|u| u.hp >= u.max_hp)
        .unwrap_or(number(&s.bases[t.owner.to_string()]) >= 300.0);
    if full {
        for e in effects(s, &t)
            .into_iter()
            .filter(|e| e["type"] == "mark")
            .collect::<Vec<_>>()
        {
            if !s.active_effect(&e, t.unit.as_ref()) {
                continue;
            }
            set_effects(
                s,
                &t,
                effects(s, &active_target(s, &t))
                    .into_iter()
                    .filter(|v| *v != e)
                    .collect(),
            );
            let source = Source {
                owner: e["owner"].as_u64().unwrap() as usize,
                unit: e["sourceId"]
                    .as_str()
                    .and_then(|id| s.units.iter().find(|u| u.id == id).cloned()),
                kind: "status",
                path: vec![],
                credit_friendly: false,
                modified: false,
            };
            damage(
                s,
                &t,
                catalog.rule("/catapultMarkDamage"),
                &source,
                catalog,
                ctx,
            )?;
        }
    }
    Ok(())
}

/// 死亡沿用 TS 顺序：快照、装备、移除、历史、因果事件、人头与反应队列。
/// 未移植的强夺/金晔入场不会提交半次死亡；外层复制保证失败原子性。
pub fn kill(
    s: &mut State,
    victim: &Unit,
    source: &Source,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    if !alive(s, &victim.id) {
        return Ok(());
    }
    let u = s
        .units
        .iter()
        .chain(s.landmarks().iter())
        .find(|u| u.id == victim.id)
        .unwrap()
        .clone();
    if catalog[&u.kind.key()].landmark.is_some() {
        let ply = s.ply;
        let l = s
            .landmarks
            .as_mut()
            .unwrap()
            .iter_mut()
            .find(|l| l.id == u.id)
            .unwrap();
        l.hp = 0.0;
        l.extra.insert("dormantSince".into(), json!(ply));
        l.extra.insert("rebuildTicks".into(), json!(0));
        l.mode = "none".into();
        ctx.emit(s,json!({"type":"death","to":u.actor_event(),"unitId":u.id,"owner":u.owner,"text":"地标休眠"}),Some(format!("{}被摧毁，等待重建",catalog[&u.kind.key()].name)));
        sync(s, catalog);
        return Ok(());
    }
    // TS 的击杀奖励与强夺只查普通棋子层，不能把独立地标层一并纳入。
    if let Some(killer) = source
        .unit
        .as_ref()
        .and_then(|u| s.units.iter().find(|v| v.id == u.id))
    {
        if killer.id != u.id && !killer.silenced && killer.has("s5") {
            return Err(Failure::Unsupported("ability-steal"));
        }
        let id = killer.id.clone();
        if killer.id != u.id && u.side() == 3 - source.owner && killer.weapon("s15") {
            s.unit_mut(&id)
                .unwrap()
                .extra
                .insert("bladeQualified".into(), json!(true));
        }
    }
    for k in &u.equipment {
        if k.is("s2") || (k.is("s15") && u.extra.get("bladeQualified") == Some(&json!(true))) {
            let id = u
                .extra
                .get("equipmentIds")
                .and_then(|e| e.get(k.key()))
                .cloned()
                .unwrap_or_else(|| {
                    let id = json!(format!("c{}", s.serial));
                    s.serial += 1;
                    id
                });
            let card = json!({"id":id,"kind":k,"drawnAt":s.turns[&u.owner.to_string()],"summonedPly":s.ply});
            s.extra.get_mut("hands").unwrap()[u.owner.to_string()]
                .as_array_mut()
                .unwrap()
                .push(card);
            ctx.emit(s,json!({"type":"skill","owner":u.owner,"text":format!("{}返回储存区",catalog[&k.key()].name)}),None);
        }
    }
    s.units.retain(|v| v.id != u.id);
    let mut dead = json!({"id":format!("dead{}",s.serial),"kind":u.kind,"owner":u.owner,"ply":s.ply,"revived":false});
    s.serial += 1;
    if let Some(group) = u.extra.get("group") {
        dead["group"] = group.clone();
    }
    let deaths = s.extra.get_mut("deaths").unwrap().as_array_mut().unwrap();
    deaths.push(dead);
    if deaths.len() > 240 {
        deaths.remove(0);
    }
    ctx.emit(
        s,
        json!({"type":"death","to":u.actor_event(),"unitId":u.id,"owner":u.owner}),
        Some(format!(
            "{}的{}离场",
            faction(u.owner),
            catalog[&u.kind.key()].name
        )),
    );
    s.siphons
        .retain(|l| l["sourceId"] != u.id && l["fromId"] != u.id && l["toId"] != u.id);
    s.extra
        .get_mut("iceMarks")
        .unwrap()
        .as_array_mut()
        .unwrap()
        .retain(|m| m["sourceId"] != u.id);
    let group_final = u.extra.get("group").is_none_or(|group| {
        !s.units.iter().any(|v| v.extra.get("group") == Some(group))
            && ["1", "2"].iter().all(|p| {
                !s.extra["hands"][p]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|c| c.get("group") == Some(group))
            })
    });
    let cities: Vec<_> = s
        .units
        .iter()
        .filter(|city| {
            city.has("citadel")
                && city.owner == u.owner
                && u.side() == u.owner
                && !city.silenced
                && attack_path(
                    s,
                    city,
                    &Target::from(&u),
                    stats(s, city, catalog).range,
                    None,
                    false,
                )
                .is_some()
        })
        .cloned()
        .collect();
    for city in cities {
        s.pending
            .push(json!({"kind":"hut-spawn","owner":city.owner,"source":city,"amount":0}));
    }
    sync(s, catalog);
    if !group_final {
        return Ok(());
    }
    let enabled = !u.silenced;
    let deny = enabled && u.has("20") && s.random(ctx.preview)? < 0.5;
    let enemy = u.side() != 0 && source.owner != u.owner;
    if u.side() != 0 && (enemy || (source.credit_friendly && source.owner == u.owner)) && !deny {
        let p = source.owner.to_string();
        let heads = s.extra.get_mut("heads").unwrap();
        heads[&p] = json!(number(&heads[&p]) + 1.0);
        ctx.emit(
            s,
            json!({"type":"skill","owner":source.owner,"text":"+1 人头"}),
            Some(format!("{}获得1人头", faction(source.owner))),
        );
    }
    if deny {
        ctx.emit(
            s,
            json!({"type":"skill","to":u.actor_event(),"owner":u.owner,"text":"人头遁逃"}),
            Some("超级跑得快：本次死亡不提供人头".into()),
        );
    }
    if let Some(killer) = source
        .unit
        .as_ref()
        .and_then(|v| s.units.iter().find(|u| u.id == v.id))
        .cloned()
        .filter(|v| enemy && !v.silenced)
    {
        let k = s.unit_mut(&killer.id).unwrap();
        let kills = extra_number(k, "kills") + 1.0;
        k.extra.insert("kills".into(), json!(kills));
        if k.has("26") {
            match (kills as i64 - 1) % 3 {
                0 => {
                    k.max_hp += 10.0;
                }
                1 => {
                    k.extra.insert(
                        "attackBonus".into(),
                        json!(extra_number(k, "attackBonus") + 5.0),
                    );
                }
                _ => {
                    k.extra.insert(
                        "rangeBonus".into(),
                        json!(extra_number(k, "rangeBonus") + 1.0),
                    );
                }
            }
        }
        if k.has("u8") && kills == 5.0 {
            k.extra.insert(
                "rangeBonus".into(),
                json!(extra_number(k, "rangeBonus") + 1.0),
            );
        }
        if k.has("u23") {
            let now = s.ply + killer.offset;
            let k = s.unit_mut(&killer.id).unwrap();
            k.extra.insert("hookReadyAt".into(), json!(now + 2.0));
            k.extra.insert("hookExpiresAt".into(), json!(now + 4.0));
        }
        if killer.has("26") && (kills as i64 - 1) % 3 == 0 {
            heal(s, &Target::from(&killer), 10.0, catalog, ctx)?;
        }
    }
    if enabled {
        if u.has("11") {
            let p = u.owner.to_string();
            let bonus = s.extra.get_mut("bonus").unwrap();
            bonus[&p] = json!(number(&bonus[&p]) + 1.0);
        }
        if u.has("12") {
            add_unit(s, "grave", u.owner, u.at(), catalog, ctx)?;
        }
        if u.has("2") {
            s.pending
                .push(json!({"kind":"death-shot","owner":u.owner,"source":u,"amount":20}));
        }
        if u.has("sage") || u.has("u21") {
            let friends: Vec<_> = s
                .units
                .iter()
                .filter(|v| v.side() == u.owner)
                .cloned()
                .collect();
            if u.has("sage") {
                for v in &friends {
                    heal(s, &Target::from(v), v.max_hp - v.hp, catalog, ctx)?;
                }
            }
            if u.has("u21") {
                for v in &friends {
                    if attack_path(
                        s,
                        &u,
                        &Target::from(v),
                        stats(s, &u, catalog).range,
                        None,
                        false,
                    )
                    .is_some()
                    {
                        heal(s, &Target::from(v), 25.0, catalog, ctx)?;
                    }
                }
            }
        }
        if u.has("20")
            && !deny
            && source.kind != "reflect"
            && let Some(t) = source
                .unit
                .as_ref()
                .and_then(|origin| find_target(s, &origin.id).filter(|t| t.id != u.id))
        {
            damage(s, &t, 20.0, &Source::new(&u, "reflect"), catalog, ctx)?;
        }
    }
    let huts: Vec<_> = s
        .units
        .iter()
        .filter(|h| {
            h.has("u22")
                && !u.kind.is("20")
                && u.side() == h.owner
                && !h.silenced
                && attack_path(
                    s,
                    h,
                    &Target::from(&u),
                    stats(s, h, catalog).range,
                    None,
                    false,
                )
                .is_some()
        })
        .cloned()
        .collect();
    for hut in huts {
        if !s.pending.iter().any(|r| {
            r["kind"] == "hut-spawn"
                && r["source"]["id"] == hut.id
                && number(&r["source"]["lastCharge"]) == s.serial as f64
        }) {
            s.pending
                .push(json!({"kind":"hut-spawn","owner":hut.owner,"source":hut,"amount":0}));
        }
    }
    Ok(())
}

/// 按 TS 顺序处理单份伤害、免伤、死亡和反伤；多层标记必须逐份调用。
/// 只修改外层命令副本，遇到随机预检/未移植钩子向上传播，禁止提交半次结算。
pub fn damage(
    s: &mut State,
    target: &Target,
    amount: f64,
    source: &Source,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<f64, Failure> {
    let mut bonus = if s.aura(source.owner, "s11") {
        5.0
    } else {
        0.0
    };
    if source.kind == "spell"
        || (["skill", "status"].contains(&source.kind)
            && source
                .unit
                .as_ref()
                .is_some_and(|u| u.kinds().iter().any(|k| catalog[&k.key()].mage)))
    {
        bonus += 15.0
            * s.pieces()
                .filter(|u| u.owner == source.owner && u.has("s14") && !u.silenced)
                .count() as f64;
    }
    let mut amount = (amount + if source.modified { 0.0 } else { bonus }).max(0.0);
    if amount <= 0.0 {
        return Ok(0.0);
    }
    let t = active_target(s, target);
    if t.unit.is_none() {
        let p = t.owner.to_string();
        let loss = number(&s.bases[&p]).min(amount);
        s.bases[&p] = json!(number(&s.bases[&p]) - loss);
        if loss != 0.0 {
            ctx.emit(
                s,
                json!({"type":"damage","to":t.actor(),"unitId":t.id,"owner":t.owner,"amount":loss}),
                Some(format!("{}基地受到{}伤害", faction(t.owner), loss)),
            );
        }
        return Ok(loss);
    }
    if !alive(s, &t.id) {
        return Ok(0.0);
    }
    let u = t.unit.as_ref().unwrap();
    if catalog[&u.kind.key()].landmark.is_some()
        && source.owner != u.owner
        && ["attack", "collision"].contains(&source.kind)
        && !source.unit.as_ref().is_some_and(Unit::piercing)
        && let Some(v) = s
            .units
            .iter()
            .find(|v| v.side() == u.owner && covers(v, u.at()))
            .cloned()
    {
        let mut modified = source.clone();
        modified.modified = true;
        return damage(s, &Target::from(&v), amount, &modified, catalog, ctx);
    }
    if s.effect(u, "immune") {
        ctx.emit(s,json!({"type":"shield","to":u.actor_event(),"owner":u.owner,"action":"ward","stage":"blocked","text":"金身"}),None);
        return Ok(0.0);
    }
    if protected(s, &t, source, catalog, ctx)? {
        return Ok(0.0);
    }
    if !u.silenced && u.has("17p") && s.random(ctx.preview)? < catalog.rule("/littleGoldImmunity") {
        ctx.emit(s,json!({"type":"shield","to":u.actor_event(),"owner":u.owner,"action":"ward","stage":"blocked","text":"小金耶 · 免疫"}),None);
        return Ok(0.0);
    }
    if !u.silenced
        && u.has("u18")
        && source.kind == "attack"
        && amount <= catalog.rule("/kingAttackImmunity")
    {
        ctx.emit(s,json!({"type":"shield","to":u.actor_event(),"owner":u.owner,"action":"ward","stage":"blocked","text":"王之蔑视"}),None);
        return Ok(0.0);
    }
    if !u.silenced && u.has("24") && frontal(&source.path, u.owner) {
        amount = amount.min(catalog.rule("/frontDamageCap"));
    }
    let guard = if amount >= u.hp {
        guardian(s, u, catalog)
    } else {
        None
    };
    let before = u.hp;
    let v = s.unit_mut(&u.id).unwrap();
    if let Some(ref guard) = guard {
        v.hp = 1.0;
        v.extra
            .entry("guardSourceIds")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .unwrap()
            .push(json!(guard));
        v.extra.insert("guardUsed".into(), json!(true));
    } else {
        v.hp = ((v.hp - amount) * 1e6).round().max(0.0) / 1e6;
    }
    let loss = before - v.hp;
    if let Some(guard) = guard {
        let g = s.unit(&guard).unwrap();
        ctx.emit(s,json!({"type":"shield","to":u.actor_event(),"owner":u.owner,"action":"ward","stage":"blocked","text":"名刀","actor":g.actor_event(),"ability":3}),None);
    }
    if loss != 0.0 {
        ctx.emit(s,json!({"type":"damage","to":u.actor_event(),"unitId":u.id,"amount":loss,"owner":u.owner}),Some(format!("{}受到{}伤害",catalog[&u.kind.key()].name,loss)));
    }
    let ply = s.ply;
    let v = s
        .units
        .iter_mut()
        .chain(s.landmarks.iter_mut().flatten())
        .find(|v| v.id == u.id)
        .unwrap();
    if loss > 0.0 {
        let received = v
            .extra
            .entry("receivedDamage")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .unwrap();
        received.retain(|r| number(&r["ply"]) >= ply - 1.0);
        if let Some(r) = received.iter_mut().find(|r| number(&r["ply"]) == ply) {
            r["amount"] = json!(number(&r["amount"]) + loss);
        } else {
            received.push(json!({"ply":ply,"amount":loss}));
        }
    }
    if v.hp <= v.max_hp {
        v.extra.remove("overMaxFromBanner");
    }
    if loss > 0.0 && !v.silenced && v.has("u10") {
        v.extra.insert(
            "attackBonus".into(),
            json!(extra_number(v, "attackBonus") + 15.0),
        );
    }
    let snap = v.clone();
    let origin = source.unit.as_ref().and_then(|u| find_target(s, &u.id));
    if snap.hp <= 0.0 {
        kill(s, &snap, source, catalog, ctx)?;
    }
    if loss > 0.0
        && snap.has("slayer")
        && !snap.silenced
        && source.kind != "reflect"
        && let Some(origin) = origin.as_ref().filter(|o| o.id != snap.id)
    {
        damage(
            s,
            origin,
            loss * catalog.rule("/slayerReflectRate"),
            &Source::new(&snap, "reflect"),
            catalog,
            ctx,
        )?;
    }
    if loss > 0.0
        && !snap.silenced
        && snap.has("16")
        && source.owner == snap.owner
        && source.unit.as_ref().is_some_and(|v| v.id != snap.id)
    {
        s.pending
            .push(json!({"kind":"reflect","owner":snap.owner,"source":snap,"amount":loss}));
    }
    if loss > 0.0
        && alive(s, &snap.id)
        && !snap.silenced
        && snap.has("u18")
        && !s.effect(&snap, "freeze")
        && !s.effect(&snap, "stun")
        && let Some(origin) = origin.filter(|o| o.id != snap.id)
    {
        crate::combat::retaliate(s, &snap, &origin, catalog, ctx)?;
    }
    Ok(loss)
}
