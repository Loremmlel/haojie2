//! 神龛的实际回合时钟、暗选和局部快照；不保存或重置规则随机状态。
use crate::geometry::{Target, attack_path, covers};
use crate::model::{Catalog, Command, Failure, Point, State, Unit, ensure, number};
use crate::preparation::kind;
use crate::resolution::{Resolution, reset_unit};
use crate::stats::stats;
use serde_json::{Value, json};

pub fn point(c: &Command) -> Result<Point, Failure> {
    ensure(
        c.x.is_some_and(|v| v.is_finite() && v.fract() == 0.0)
            && c.y.is_some_and(|v| v.is_finite() && v.fract() == 0.0),
        "请选择棋盘格。",
    )?;
    Ok(Point {
        x: c.x.unwrap(),
        y: c.y.unwrap(),
    })
}
pub fn capture_clock(s: &mut State) {
    if !s.aura(1, "s10")
        && !s.aura(2, "s10")
        && !["1", "2"].iter().any(|p| {
            s.extra["hands"][p]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["kind"] == "s10")
        })
    {
        return;
    }
    let current = json!({"ply":s.ply,"turns":s.turns,"units":s.units});
    let f = &mut s
        .extra
        .entry("clockFrames")
        .or_insert_with(|| json!({"1":{},"2":{}}))[s.active.to_string()];
    if let Some(old) = f.get("current").cloned() {
        f["previous"] = old;
    } else {
        f.as_object_mut().unwrap().remove("previous");
    }
    f["current"] = current;
}
pub fn rebuild(s: &mut State, catalog: &Catalog, ctx: &mut Resolution) {
    for old in s.landmarks().to_vec() {
        if old.owner != s.active {
            continue;
        }
        let mut l = old.clone();
        if old
            .extra
            .get("dormantSince")
            .is_some_and(|v| s.ply > number(v))
        {
            let limit = number(&catalog[&l.kind.key()].landmark.as_ref().unwrap()["rebuild"]);
            let ticks = (l.extra.get("rebuildTicks").map(number).unwrap_or(0.0) + 1.0).min(limit);
            l.extra.insert("rebuildTicks".into(), json!(ticks));
            if ticks >= limit
                && s.units
                    .iter()
                    .filter(|u| covers(u, l.at()))
                    .all(|u| u.side() == l.owner)
            {
                l.extra.remove("dormantSince");
                l.extra.remove("rebuildTicks");
                l.hp = l.max_hp;
                l.silenced = false;
                l.effects.clear();
                ctx.emit(s,json!({"type":"spawn","to":l.actor_event(),"unitId":l.id,"owner":l.owner,"text":"地标重建"}),Some(format!("{}已重建",catalog[&l.kind.key()].name)));
            }
        }
        if l.live() {
            reset_unit(s.ply, &mut l);
        }
        let slot = s
            .landmarks
            .as_mut()
            .unwrap()
            .iter_mut()
            .find(|u| u.id == old.id)
            .unwrap();
        *slot = l;
    }
    crate::movement::sync_banners(s, catalog);
}
pub fn end(s: &mut State, catalog: &Catalog, ctx: &mut Resolution) -> Result<(), Failure> {
    for source in s.units.clone() {
        if source.owner != s.active
            || !source.has("s4")
            || source.silenced
            || !crate::damage::alive(s, &source.id)
        {
            continue;
        }
        for friend in s.pieces().cloned().collect::<Vec<_>>() {
            if friend.side() == source.owner
                && attack_path(
                    s,
                    &source,
                    &Target::from(&friend),
                    stats(s, &source, catalog).range,
                    None,
                    false,
                )
                .is_some()
            {
                crate::damage::heal(
                    s,
                    &Target::from(&friend),
                    friend.max_hp - friend.hp,
                    catalog,
                    ctx,
                )?;
            }
        }
    }
    Ok(())
}
pub fn choose(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let d = s.extra.get("shrineDraft");
    ensure(
        s.phase == "shrine-draft" && d.is_some_and(|d| d["revealed"] != true),
        "当前不是神龛选择阶段。",
    )?;
    let p = c.player.unwrap_or(0);
    ensure(p == 1 || p == 2, "请选择作出决定的一方。")?;
    let key = p.to_string();
    let d = d.unwrap();
    ensure(d["committed"][&key] != true, "这一方已经锁定，不可改选。")?;
    let selected = c.shrine_kind.as_ref().map(|k| json!(k));
    ensure(
        selected
            .as_ref()
            .is_some_and(|k| d["offers"][&key].as_array().unwrap().contains(k)),
        "只能选择本方三个候选中的一个。",
    )?;
    let selected = selected.unwrap();
    ensure(
        selected != "s9"
            || c.parity
                .as_deref()
                .is_some_and(|p| ["odd", "even"].contains(&p)),
        "玉碎需要同时选择奇数或偶数。",
    )?;
    if ctx.preview {
        return Ok(());
    }
    let d = s.extra.get_mut("shrineDraft").unwrap();
    let mut choice = json!({"kind":selected});
    if selected == "s9" {
        choice["parity"] = json!(c.parity);
    }
    d["choices"][&key] = choice;
    d["committed"][&key] = json!(true);
    ctx.emit(
        s,
        json!({"type":"turn","owner":p,"text":"神龛已锁定"}),
        Some("一方已锁定神龛，等待另一方".into()),
    );
    if s.extra["shrineDraft"]["committed"][(3 - p).to_string()] != true {
        s.active = 3 - p;
        return Ok(());
    }
    let d = s.extra.get_mut("shrineDraft").unwrap();
    ensure(
        !d["choices"]["1"].is_null() && !d["choices"]["2"].is_null(),
        "等待另一方的保密选择；观察视图不能代替权威对局。",
    )?;
    d["revealed"] = json!(true);
    s.active = 1;
    s.phase = "shrine-setup".into();
    for owner in 1..=2 {
        let choice = s.extra["shrineDraft"]["choices"][owner.to_string()].clone();
        let mut card =
            json!({"id":format!("c{}",s.serial),"kind":choice["kind"],"drawnAt":0,"summonedPly":0});
        s.serial += 1;
        if let Some(parity) = choice.get("parity") {
            card["parity"] = parity.clone();
        }
        s.extra.get_mut("hands").unwrap()[owner.to_string()]
            .as_array_mut()
            .unwrap()
            .push(card);
        let name = &catalog[&kind(&choice["kind"]).key()].name;
        ctx.emit(
            s,
            json!({"type":"summon","owner":owner,"text":name,"ultimate":true}),
            Some(format!(
                "{owner}方揭示：{name}{}",
                match choice["parity"].as_str() {
                    Some("odd") => " · 奇数",
                    Some("even") => " · 偶数",
                    _ => "",
                }
            )),
        );
    }
    Ok(())
}
fn restored(s: &State, u: &Unit, catalog: &Catalog) -> Result<Unit, Failure> {
    let frame = s
        .extra
        .get("clockFrames")
        .and_then(|v| v.get(s.active.to_string()))
        .and_then(|v| v.get("previous"));
    let old = frame
        .and_then(|f| f["units"].as_array())
        .and_then(|a| a.iter().find(|v| v["id"] == u.id));
    if let (Some(frame), Some(old)) = (frame, old) {
        let mut v: Unit = serde_json::from_value(old.clone()).unwrap();
        let shift = s.ply - number(&frame["ply"]);
        v.born += s.turns[&v.owner.to_string()] - number(&frame["turns"][v.owner.to_string()]);
        for e in &mut v.effects {
            e["from"] = json!(number(&e["from"]) + shift);
            e["until"] = json!((number(&e["until"]) + shift).min(9007199254740991.0));
        }
        for key in ["hookReadyAt", "hookExpiresAt", "expiresAt", "rerollUsedPly"] {
            if let Some(n) = v.extra.get_mut(key) {
                *n = json!(number(n) + shift);
            }
        }
        for key in ["freeUsed", "lastCharge"] {
            if let Some(n) = v.extra.get_mut(key).filter(|v| number(v) >= 0.0) {
                *n = json!(number(n) + shift);
            }
        }
        if let Some(reserves) = v
            .extra
            .get_mut("abilityCharges")
            .and_then(Value::as_object_mut)
        {
            for r in reserves.values_mut().filter(|r| r.is_object()) {
                if number(&r["lastCharge"]) >= 0.0 {
                    r["lastCharge"] = json!(number(&r["lastCharge"]) + shift);
                }
            }
        }
        if let Some(usage) = v
            .extra
            .get_mut("abilityUsage")
            .and_then(Value::as_object_mut)
        {
            for r in usage.values_mut().filter(|r| r.is_object()) {
                if number(&r["free"]) >= 0.0 {
                    r["free"] = json!(number(&r["free"]) + shift);
                }
            }
        }
        if let Some(records) = v
            .extra
            .get_mut("receivedDamage")
            .and_then(Value::as_array_mut)
        {
            for r in records {
                r["ply"] = json!(number(&r["ply"]) + shift);
            }
        }
        return Ok(v);
    }
    ensure(
        u.owner != s.active
            && frame
                .is_none_or(|f| crate::model::extra_number(u, "deployedAt") > number(&f["ply"])),
        "目标在上一个己方回合没有快照，且不是敌方新召唤棋子。",
    )?;
    let mut v = crate::resolution::template(s, &u.kind.key(), u.owner, u.at(), &u.id, catalog);
    v.born = u.born;
    for key in ["deployedAt", "chargedOnDeploy", "group"] {
        if let Some(value) = u.extra.get(key) {
            v.extra.insert(key.into(), value.clone());
        }
    }
    if u.kind.is("1") && u.extra.get("chargedOnDeploy") == Some(&json!(true)) {
        v.max_hp -= 10.0;
        v.hp -= 10.0;
    }
    Ok(v)
}
/// 与 TS prepareClockRestore 相同的只读准备；预检和正式结算共用，不支付光环次数。
pub fn prepare_clock(s: &State, c: &Command, catalog: &Catalog) -> Result<(Unit, Unit), Failure> {
    let a = s
        .extra
        .get("auras")
        .and_then(|a| a.get(s.active.to_string()))
        .and_then(Value::as_array)
        .and_then(|a| a.iter().find(|v| v["kind"] == "s10"));
    ensure(
        s.phase == "play"
            && a.is_some_and(|a| a.get("usedPly").and_then(Value::as_f64) != Some(s.ply)),
        "时钟每个实际己方回合只能使用一次。",
    )?;
    let u = crate::abilities::unit(s, c.target_id.as_deref())?;
    ensure(
        catalog[&u.kind.key()].tier != "shrine" && catalog[&u.kind.key()].landmark.is_none(),
        "时钟不能对神龛或地标生效。",
    )?;
    let restored = restored(s, &u, catalog)?;
    ensure(
        crate::geometry::can_place(s, &restored, restored.at(), catalog),
        "时钟原位置被占或不再合法；未消耗次数。",
    )?;
    for k in &restored.equipment {
        if catalog[&k.key()].tier != "shrine" {
            continue;
        }
        let id = restored
            .extra
            .get("equipmentIds")
            .and_then(|v| v.get(k.key()));
        if let Some(id) = id {
            ensure(
                !["1", "2"].iter().any(|p| {
                    s.extra["hands"][p]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|c| c["id"] == *id)
                }) && !s.units.iter().any(|v| {
                    v.id != u.id
                        && v.extra
                            .get("equipmentIds")
                            .and_then(Value::as_object)
                            .is_some_and(|v| v.values().any(|x| x == id))
                }),
                "原神龛武器已经转移，不能通过时钟复制。",
            )?;
        }
    }
    Ok((u, restored))
}
pub fn clock(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let (u, restored) = prepare_clock(s, c, catalog)?;
    s.extra.get_mut("auras").unwrap()[s.active.to_string()]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|v| v["kind"] == "s10")
        .unwrap()["usedPly"] = json!(s.ply);
    if !crate::damage::protected(
        s,
        &Target::from(&u),
        &crate::damage::Source::effect(s.active, "skill"),
        catalog,
        ctx,
    )? {
        *s.units.iter_mut().find(|v| v.id == u.id).unwrap() = restored.clone();
        crate::movement::sync_banners(s, catalog);
        ctx.emit(s,json!({"type":"skill","from":u.actor_event(),"to":restored.actor_event(),"unitId":u.id,"owner":s.active,"action":"clock","text":"时钟 · 状态复原"}),None);
    }
    Ok(())
}
pub fn shatter(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let u = crate::movement::actor(s, c, catalog)?.clone();
    let key = u.kind.key();
    let d = &catalog[&key];
    let ordinal = key
        .trim_start_matches(['u', 's'])
        .trim_end_matches('p')
        .parse::<u32>()
        .ok();
    let aura = s
        .extra
        .get("auras")
        .and_then(|a| a.get(u.owner.to_string()))
        .and_then(Value::as_array)
        .and_then(|a| a.iter().find(|v| v["kind"] == "s9"));
    ensure(
        d.spell.is_none()
            && d.weapon.is_none()
            && !d.aura
            && d.landmark.is_none()
            && !u.any(&["grave", "wall"])
            && ordinal.is_some_and(|n| {
                aura.is_some_and(|a| a["parity"] == if n % 2 == 1 { "odd" } else { "even" })
            }),
        "玉碎只对所选奇偶编号的友方随从开放。",
    )?;
    crate::movement::choose_skill(s, &u.id, catalog)?;
    let t = c
        .target_id
        .as_deref()
        .and_then(|id| crate::geometry::find_target(s, id));
    ensure(
        t.as_ref().is_some_and(|t| {
            t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) == 3 - u.owner
                && crate::geometry::top_target(s, t, catalog)
        }),
        "玉碎需要一个敌方目标。",
    )?;
    let t = t.unwrap();
    ensure(
        attack_path(s, &u, &t, stats(s, &u, catalog).range, None, false).is_some(),
        "玉碎目标不在攻击范围内。",
    )?;
    let amount = ((stats(s, &u, catalog).attack + u.hp) / 10.0)
        .ceil()
        .max(0.0)
        * 5.0;
    let u = s.unit(&u.id).unwrap().clone();
    crate::damage::kill(
        s,
        &u,
        &crate::damage::Source::new(&u, "sacrifice"),
        catalog,
        ctx,
    )?;
    crate::damage::damage(
        s,
        &t,
        amount,
        &crate::damage::Source::new(&u, "skill"),
        catalog,
        ctx,
    )?;
    ctx.emit(
        s,
        json!({"type":"skill","from":u.actor_event(),"to":t.actor(),"owner":u.owner,"text":"玉碎"}),
        None,
    );
    Ok(())
}
pub fn steal(
    s: &mut State,
    killer_id: &str,
    victim: &mut Unit,
    catalog: &Catalog,
    ctx: &mut Resolution,
) {
    let Some(killer) = s
        .unit_mut(killer_id)
        .filter(|u| !u.silenced && u.has("s5") && u.id != victim.id)
    else {
        return;
    };
    let mut traits = killer
        .extra
        .get("traits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for k in victim.kinds() {
        let value = json!(k);
        if k != killer.kind && !traits.contains(&value) {
            traits.push(value);
        }
    }
    killer.extra.insert("traits".into(), json!(traits));
    if let Some(k) = victim
        .equipment
        .first()
        .cloned()
        .filter(|_| !killer.any(&["10", "s7"]))
    {
        let without = killer.max_hp
            - killer
                .equipment
                .iter()
                .map(crate::preparation::weapon_health)
                .sum::<f64>();
        let mage = killer.kinds().iter().any(|k| catalog[&k.key()].mage);
        if without + crate::preparation::weapon_health(&k) > 0.0
            && (!(k.is("u5") || k.is("s16")) || mage)
            && (!k.is("u28") || !mage)
        {
            killer.max_hp = without;
            killer.hp = killer.hp.min(without);
            killer.equipment = vec![k.clone()];
            let id = victim
                .extra
                .get("equipmentIds")
                .and_then(|v| v.get(k.key()))
                .cloned();
            killer.extra.insert(
                "equipmentIds".into(),
                id.map(|id| json!({k.key():id}))
                    .unwrap_or_else(|| json!({})),
            );
            killer.extra.insert("bladeQualified".into(), json!(false));
            killer.max_hp += crate::preparation::weapon_health(&k);
            if k.is("u28") || k.is("s2") {
                killer.hp += crate::preparation::weapon_health(&k);
            }
            killer.hp = killer.hp.min(killer.max_hp);
            if k.is("u28") {
                killer.effects.retain(|e| e["type"] != "freeze");
            }
            killer.extra.remove("overMaxFromBanner");
            victim.equipment.clear();
            victim.extra.insert("equipmentIds".into(), json!({}));
        }
    }
    let event = json!({"type":"skill","from":victim.actor_event(),"to":killer.actor_event(),"owner":killer.owner,"text":"ZF·强夺 · 获得技能与武器"});
    ctx.emit(s, event, None);
}
