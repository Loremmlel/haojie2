//! 召唤、部署与准备操作的等价移植。只消费初始化注入的图鉴/编号池，不维护第二份数值表。
//! 所有支付、随机数和卡牌变化发生在命令副本上；失败或预检遇随机边界时整条命令丢弃。
use crate::geometry::can_deploy;
use crate::model::{Catalog, Command, Failure, Kind, Point, State, ensure, extra_number, number};
use crate::movement::{actor, finish, movement_stats, reserve, stage, sync};
use crate::resolution::{Resolution, deploy_unit, normalize_guards, template, terminal};
use serde_json::{Value, json};

pub fn hand(s: &State) -> &[Value] {
    s.extra["hands"][s.active.to_string()].as_array().unwrap()
}
pub fn hand_mut(s: &mut State) -> &mut Vec<Value> {
    s.extra.get_mut("hands").unwrap()[s.active.to_string()]
        .as_array_mut()
        .unwrap()
}
pub fn card(s: &State, c: &Command) -> Option<Value> {
    hand(s)
        .iter()
        .find(|v| v["id"].as_str() == c.card_id.as_deref())
        .cloned()
}
pub fn kind(v: &Value) -> Kind {
    serde_json::from_value(v.clone()).expect("规范局面的卡牌编号")
}
pub fn remove_card(s: &mut State, card: &Value) {
    hand_mut(s).retain(|v| v["id"] != card["id"]);
}
pub fn pay(s: &mut State, cost: f64, message: &'static str) -> Result<(), Failure> {
    let heads = &mut s.extra.get_mut("heads").unwrap()[s.active.to_string()];
    ensure(number(heads) >= cost, message)?;
    *heads = json!(number(heads) - cost);
    Ok(())
}
fn chosen(s: &mut State, k: &Kind, ultimate: bool, catalog: &Catalog) -> Result<(), Failure> {
    let ply = s.ply;
    let aura = s
        .extra
        .get_mut("auras")
        .and_then(|a| a.get_mut(s.active.to_string()))
        .and_then(Value::as_array_mut)
        .and_then(|a| a.iter_mut().find(|v| v["kind"] == "laoqian"));
    let aura = aura
        .filter(|a| a.get("usedPly").and_then(Value::as_f64) != Some(ply))
        .ok_or(Failure::Invalid(
            "本回合牢千K自选召唤已使用，或尚未获得光环。",
        ))?;
    ensure(
        catalog.pools[if ultimate { "ultimate" } else { "normal" }].contains(k),
        "自选结果必须属于本次召唤的来源池。",
    )?;
    aura["usedPly"] = json!(ply);
    Ok(())
}
/// 每次完整抽取含派生变体和克隆整批；预检在第一次真正需要随机数时停止。
pub fn draw(
    s: &mut State,
    ultimate: bool,
    selected: Option<&Kind>,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<Vec<Value>, Failure> {
    if let Some(k) = selected {
        chosen(s, k, ultimate, catalog)?;
    }
    let pool = &catalog.pools[if ultimate { "ultimate" } else { "normal" }];
    let mut k = if let Some(k) = selected {
        k.clone()
    } else {
        pool[(s.random(ctx.preview)? * pool.len() as f64).floor() as usize].clone()
    };
    if selected.is_none() {
        if k.is("3") && s.random(ctx.preview)? >= 1.0 / 3.0 {
            k = Kind::Text("3p".into());
        }
        if k.is("17") && s.random(ctx.preview)? >= catalog.rule("/goldSpellChance") {
            k = Kind::Text("17p".into());
        }
        if k.is("u12") && s.random(ctx.preview)? >= 0.1 {
            k = Kind::Text("u12p".into());
        }
    }
    let group = if k.is("u25") {
        let id = format!("group{}", s.serial);
        s.serial += 1;
        Some(id)
    } else {
        None
    };
    let d = &catalog[&k.key()];
    let mut cards = vec![];
    for _ in 0..if group.is_some() { 8 } else { 1 } {
        let mut card = json!({"id":format!("c{}",s.serial),"kind":k,"drawnAt":s.turns[&s.active.to_string()],"summonedPly":s.ply,"summonPool":if ultimate {"ultimate"} else {"normal"}});
        s.serial += 1;
        if let Some(limit) = d.spell.or(d.weapon).filter(|v| *v >= 0.0) {
            card["expiresAt"] = json!(s.turns[&s.active.to_string()] + limit);
        }
        if let Some(group) = &group {
            card["group"] = json!(group);
        }
        hand_mut(s).push(card.clone());
        cards.push(card);
    }
    ctx.emit(
        s,
        json!({"type":"summon","owner":s.active,"text":d.name,"ultimate":ultimate}),
        Some(format!(
            "{}召唤：{}{}",
            if ultimate { "终极" } else { "普通" },
            d.name,
            if group.is_some() { " ×8" } else { "" }
        )),
    );
    Ok(cards)
}
fn summon(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    ensure(
        !s.extra.contains_key("summonOffer"),
        "先从候选召唤中选出两个结果。",
    )?;
    ensure(s.summon_slots > 0.0, "本回合召唤次数已用完。")?;
    let shrine = s.extra.get("mode") == Some(&json!("shrine"));
    if shrine {
        ensure(
            c.ultimate != Some(false) || c.mode.as_deref() != Some("normal"),
            "常驻召唤使用终极池；普通池需消耗2人头额外召唤。",
        )?;
    }
    let ultimate = shrine || c.ultimate == Some(true);
    if !shrine && ultimate {
        pay(s, 2.0, "终极召唤需要2人头。")?;
    }
    if shrine
        && s.extra.get("regularSummons").and_then(Value::as_f64) == Some(2.0)
        && s.aura(s.active, "s13")
    {
        let count = if s.random(ctx.preview)? < 1.0 / 3.0 {
            4
        } else {
            3
        };
        let mut groups = vec![];
        for i in 0..count {
            groups.push(draw(
                s,
                true,
                if i == 0 { c.chosen_kind.as_ref() } else { None },
                catalog,
                ctx,
            )?);
        }
        hand_mut(s).retain(|v| !groups.iter().flatten().any(|card| card["id"] == v["id"]));
        s.extra.insert(
            "summonOffer".into(),
            json!({"owner":s.active,"groups":groups,"count":2}),
        );
        s.summon_slots -= 2.0;
        s.extra.insert("regularSummons".into(), json!(0));
    } else {
        draw(s, ultimate, c.chosen_kind.as_ref(), catalog, ctx)?;
        s.summon_slots -= 1.0;
        if shrine && s.extra.get("regularSummons").map(number).unwrap_or(0.0) > 0.0 {
            s.extra.insert(
                "regularSummons".into(),
                json!(number(&s.extra["regularSummons"]) - 1.0),
            );
        }
    }
    Ok(())
}
fn choose_summons(s: &mut State, c: &Command, ctx: &mut Resolution) -> Result<(), Failure> {
    let offer = s
        .extra
        .get("summonOffer")
        .filter(|v| v["owner"] == s.active)
        .ok_or(Failure::Invalid("当前没有待选的召唤结果。"))?;
    let groups = offer["groups"].as_array().unwrap();
    let indices = c.offer_indices.as_deref().unwrap_or(&[]);
    ensure(
        indices.len() == 2
            && indices[0] != indices[1]
            && indices
                .iter()
                .all(|i| i.fract() == 0.0 && *i >= 0.0 && *i < groups.len() as f64),
        "请选择两个不同的完整召唤结果。",
    )?;
    let cards: Vec<_> = indices
        .iter()
        .flat_map(|i| groups[*i as usize].as_array().unwrap().iter().cloned())
        .collect();
    hand_mut(s).extend(cards);
    s.extra.remove("summonOffer");
    ctx.emit(
        s,
        json!({"type":"summon","owner":s.active,"text":"老千K · 选定两个结果"}),
        None,
    );
    Ok(())
}
fn deploy(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let card = card(s, c)
        .filter(|v| {
            let d = &catalog[&kind(&v["kind"]).key()];
            d.spell.is_none() && d.weapon.is_none() && !d.aura
        })
        .ok_or(Failure::Invalid("请选择待部署随从。"))?;
    let (Some(x), Some(y)) = (c.x, c.y) else {
        return Err(Failure::Invalid("请选择棋盘格。"));
    };
    ensure(x.fract() == 0.0 && y.fract() == 0.0, "请选择棋盘格。")?;
    let at = Point { x, y };
    let k = kind(&card["kind"]).key();
    let ghost = template(s, &k, s.active, at, "preview", catalog);
    ensure(
        can_deploy(s, &ghost, at, catalog),
        "非法部署：检查行权限、占位、基地与独行侠禁区。",
    )?;
    deploy_unit(
        s,
        &k,
        s.active,
        at,
        card["group"].as_str(),
        c.charge == Some(true),
        catalog,
        ctx,
    );
    remove_card(s, &card);
    Ok(())
}
fn charge(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let u = s
        .unit(c.unit_id.as_deref().unwrap_or(""))
        .ok_or(Failure::Invalid("请选择仍在场上的随从。"))?;
    let k = c.ability.as_ref().unwrap_or(&u.kind).clone();
    ensure(u.has(&k.key()), "该棋子没有选定的蓄力能力。")?;
    let u = actor(s, c, catalog)?;
    ensure(
        u.mode == "none" || u.mode == "charge",
        "本回合已选择另一操作模式，剩余攻击不能换成移动或技能。",
    )?;
    if u.mode == "none" {
        ensure(movement_stats(s, u, catalog).1 > 0.0, "本回合操作已用完。")?;
    }
    let d = &catalog[&k.key()];
    let mode = c.mode.as_deref().unwrap_or("");
    let once = if u.kind.is("u6") {
        u.extra.get("onceUsed") == Some(&json!(true))
    } else {
        u.extra
            .get("abilityUsage")
            .is_some_and(|v| v["u6"]["once"] == true)
    };
    let max = if (mode == "move" && d.movement.fract() != 0.0)
        || (mode == "attack" && d.actions == 0.5)
    {
        1.0
    } else if mode == "attack" && k.is("4") && !u.silenced {
        5.0
    } else if mode == "attack" && k.is("15") && !u.silenced {
        catalog.rule("/accumulator/max")
    } else if mode == "attack" && k.is("u2") && !u.silenced {
        4.0
    } else if mode == "skill" && k.is("21") && !u.silenced {
        2.0
    } else if mode == "skill" && k.is("u6") && !u.silenced && !once {
        1.0
    } else {
        0.0
    };
    ensure(max > 0.0, "这枚随从没有此类蓄力。")?;
    let mut r = reserve(u, &k, catalog);
    let value = number(&r["charge"]);
    ensure(value < max, "该类蓄力已满。")?;
    ensure(
        value == 0.0 || r["chargeType"] == mode,
        "当前蓄力属于另一模式。",
    )?;
    r["chargeType"] = json!(mode);
    r["charge"] = json!(value + 1.0);
    r["lastCharge"] = json!(s.ply + u.offset);
    ctx.facts = Some(json!({"action":"charge","actor":u.actor_event()}));
    let u = s.unit_mut(c.unit_id.as_deref().unwrap()).unwrap();
    if k == u.kind {
        u.extra.extend(r.as_object().unwrap().clone());
    } else {
        // 与 withAbilityCharge 一样只回写四个资源字段，原生蓄力保持独立。
        u.extra.entry("abilityCharges").or_insert_with(|| json!({}))[k.key()] = json!({"charge":r["charge"],"readyCharge":r["readyCharge"],"chargeType":r["chargeType"],"lastCharge":r["lastCharge"]});
    }
    finish(u);
    let event = json!({"type":"skill","to":u.actor_event(),"owner":u.owner,"text":format!("蓄力 {}/{}",value+1.0,max)});
    ctx.emit(s, event, None);
    ctx.facts = None;
    Ok(())
}
pub fn weapon_health(k: &Kind) -> f64 {
    match k.key().as_str() {
        "u11" => 25.0,
        "u28" => 5.0,
        "s2" => 20.0,
        "s15" => -5.0,
        _ => 0.0,
    }
}
fn equip(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let card = card(s, c)
        .filter(|v| catalog[&kind(&v["kind"]).key()].weapon.is_some())
        .ok_or(Failure::Invalid("请选择武器牌。"))?;
    let k = kind(&card["kind"]);
    let u = s
        .unit(c.target_id.as_deref().unwrap_or(""))
        .ok_or(Failure::Invalid("请选择仍在场上的随从。"))?;
    ensure(
        u.side() == s.active && catalog[&u.kind.key()].landmark.is_none(),
        "只能给非中立的友方随从装备武器。",
    )?;
    ensure(!u.any(&["10", "s7"]), "投石机与YYF不能装备武器。")?;
    let mage = u.kinds().iter().any(|k| catalog[&k.key()].mage);
    ensure(!(k.is("u5") || k.is("s16")) || mage, "这件法杖仅限法师。")?;
    ensure(!k.is("u28") || !mage, "炎魔之心仅限非法师。")?;
    ensure(
        !k.is("u28")
            || u.extra["chargedOnDeploy"] != true
            || extra_number(u, "deployedAt") != s.ply,
        "冲锋随从部署当回合不能装备炎魔之心。",
    )?;
    let without = u.max_hp - u.equipment.iter().map(weapon_health).sum::<f64>();
    ensure(
        without + weapon_health(&k) > 0.0,
        "更换这件武器会使生命上限归零，不能装备。",
    )?;
    let u = s.unit_mut(c.target_id.as_deref().unwrap()).unwrap();
    u.max_hp = without;
    u.hp = u.hp.min(without);
    u.equipment = vec![k.clone()];
    u.extra
        .insert("equipmentIds".into(), json!({k.key():card["id"]}));
    u.extra.insert("bladeQualified".into(), json!(false));
    u.max_hp += weapon_health(&k);
    if k.is("u28") || k.is("s2") {
        u.hp += weapon_health(&k);
    }
    u.hp = u.hp.min(u.max_hp);
    if k.is("u28") {
        u.effects.retain(|e| e["type"] != "freeze");
    }
    u.extra.remove("overMaxFromBanner");
    let event = json!({"type":"shield","to":u.actor_event(),"owner":u.owner,"text":format!("装备 · {}",catalog[&k.key()].name),"ultimate":true});
    remove_card(s, &card);
    ctx.facts = Some(json!({"action":"equip","ability":k}));
    ctx.emit(s, event, None);
    ctx.facts = None;
    Ok(())
}
fn aura(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    ensure(
        s.phase == "play" || s.phase == "shrine-setup",
        "请在行动或开局入场阶段启用光环。",
    )?;
    let card = card(s, c)
        .filter(|v| catalog[&kind(&v["kind"]).key()].aura)
        .ok_or(Failure::Invalid("请选择光环牌。"))?;
    let k = kind(&card["kind"]);
    ensure(!s.aura(s.active, &k.key()), "这个永久光环已经启用。")?;
    let mut entry = json!({"kind":k});
    if let Some(parity) = card.get("parity") {
        entry["parity"] = parity.clone();
    }
    ensure(
        !k.is("s9") || entry.get("parity").is_some(),
        "玉碎的开局奇偶选择缺失。",
    )?;
    s.extra
        .entry("auras")
        .or_insert_with(|| json!({"1":[],"2":[]}))[s.active.to_string()]
    .as_array_mut()
    .unwrap()
    .push(entry);
    remove_card(s, &card);
    ctx.emit(s,json!({"type":"skill","owner":s.active,"text":format!("永久光环 · {}",catalog[&k.key()].name)}),None);
    Ok(())
}
fn reroll(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let card = card(s, c)
        .filter(|v| v["summonedPly"] == s.ply)
        .ok_or(Failure::Invalid("改判仅限本回合刚召唤的牌。"))?;
    let group = card["group"].as_str();
    if group.is_some() {
        ensure(
            hand(s)
                .iter()
                .filter(|v| v["group"] == card["group"])
                .count()
                == 8,
            "已有克隆部署，不可对半批军团改判。",
        )?;
    }
    ensure(
        s.summon_slots == 0.0,
        "请先用完本回合的召唤次数，再选择改判。",
    )?;
    let k = kind(&card["kind"]);
    let pool = card["summonPool"].as_str().or_else(|| {
        if catalog.pools["normal"].contains(&k) || k.is("3p") || k.is("17p") {
            Some("normal")
        } else if catalog.pools["ultimate"].contains(&k) || k.is("u12p") {
            Some("ultimate")
        } else {
            None
        }
    });
    let source = if let Some(id) = &c.unit_id {
        s.units.iter().any(|u| {
            u.id == *id
                && u.has("u13")
                && u.owner == s.active
                && !u.silenced
                && u.extra
                    .get("rerollUsedPly")
                    .map(|v| number(v) != s.ply)
                    .unwrap_or(extra_number(u, "freeUsed") != s.ply + u.offset)
        })
    } else {
        k.is("u13") && s.turns[&s.active.to_string()] <= 5.0 && card["rerolled"] != true
    };
    ensure(
        pool.is_some()
            && source
            && !["synthesis", "shrine-draft", "shrine-setup"].contains(&s.phase.as_str()),
        "需要一名本回合尚未改判的友方改判小法师，或前五回合刚抽到的自身。",
    )?;
    if let Some(id) = &c.unit_id {
        let ply = s.ply;
        s.unit_mut(id)
            .unwrap()
            .extra
            .insert("rerollUsedPly".into(), json!(ply));
    }
    hand_mut(s).retain(|v| {
        if group.is_some() {
            v["group"] != card["group"]
        } else {
            v["id"] != card["id"]
        }
    });
    let cards = draw(
        s,
        pool == Some("ultimate"),
        c.chosen_kind.as_ref(),
        catalog,
        ctx,
    )?;
    for v in hand_mut(s) {
        if cards.iter().any(|c| c["id"] == v["id"]) {
            v["rerolled"] = json!(true);
        }
    }
    ctx.emit(
        s,
        json!({"type":"skill","owner":s.active,"text":"改判"}),
        Some(format!("{}重新召唤", catalog[&k.key()].name)),
    );
    Ok(())
}
pub fn apply(
    previous: &State,
    c: &Command,
    catalog: &Catalog,
    preview: bool,
) -> Result<State, Failure> {
    stage(previous, c, catalog)?;
    let mut s = previous.clone();
    normalize_guards(&mut s);
    s.events.clear();
    let mut ctx = Resolution {
        preview,
        ..Default::default()
    };
    match c.kind.as_str() {
        "skill" => crate::abilities::skill(&mut s, c, catalog, &mut ctx)?,
        "cast" => crate::spells::cast(&mut s, c, catalog, &mut ctx)?,
        "clock" => crate::shrines::clock(&mut s, c, catalog, &mut ctx)?,
        "shatter" => crate::shrines::shatter(&mut s, c, catalog, &mut ctx)?,
        "end" => crate::lifecycle::end(&mut s, catalog, &mut ctx)?,
        "synthesize" => crate::synthesis::synthesize(&mut s, c, catalog, &mut ctx)?,
        "craft" => {
            let mut command = c.clone();
            command.recipe_id = Some("firelord".into());
            command.material_ids = c.card_ids.clone();
            crate::synthesis::synthesize(&mut s, &command, catalog, &mut ctx)?;
        }
        "choose-shrine" => crate::shrines::choose(&mut s, c, catalog, &mut ctx)?,
        "finish-shrine-setup" => {
            ensure(s.phase == "shrine-setup", "当前不是神龛入场阶段。")?;
            s.extra
                .entry("shrineSetupDone")
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .unwrap()
                .push(json!(s.active));
            if s.active == 1 {
                s.active = 2;
            } else {
                s.active = 1;
                s.ply = 1.0;
                crate::lifecycle::begin(&mut s, catalog, &mut ctx)?;
            }
        }
        "summon" => summon(&mut s, c, catalog, &mut ctx)?,
        "choose-summons" => choose_summons(&mut s, c, &mut ctx)?,
        "extra-summon" => {
            ensure(
                s.extra.get("mode") == Some(&json!("shrine"))
                    && s.phase == "summon"
                    && !s.extra.contains_key("summonOffer"),
                "人头额外召唤仅限神龛模式回合开始。",
            )?;
            let ultimate = c.ultimate != Some(false);
            pay(
                &mut s,
                if ultimate { 3.0 } else { 2.0 },
                if ultimate {
                    "本次终极召唤需要3人头。"
                } else {
                    "本次普通召唤需要2人头。"
                },
            )?;
            draw(&mut s, ultimate, c.chosen_kind.as_ref(), catalog, &mut ctx)?;
        }
        "deploy" => deploy(&mut s, c, catalog, &mut ctx)?,
        "charge" => charge(&mut s, c, catalog, &mut ctx)?,
        "equip" => equip(&mut s, c, catalog, &mut ctx)?,
        "activate-aura" => aura(&mut s, c, catalog, &mut ctx)?,
        "reroll" => reroll(&mut s, c, catalog, &mut ctx)?,
        "begin" => {
            ensure(
                s.phase == "summon" && s.summon_slots == 0.0,
                "请先完成所有召唤。",
            )?;
            s.phase = "play".into();
            ctx.emit(
                &mut s,
                json!({"type":"turn","owner":previous.active,"text":"行动阶段"}),
                None,
            );
        }
        "skip-synthesis" => {
            ensure(s.phase == "synthesis", "当前不是合成窗口。")?;
            s.phase = "summon".into();
            ctx.emit(
                &mut s,
                json!({"type":"turn","owner":previous.active,"text":"进入召唤阶段"}),
                None,
            );
        }
        _ => return Err(Failure::Unsupported("command-kind")),
    }
    sync(&mut s, catalog);
    terminal(&mut s, &mut ctx);
    Ok(s)
}
