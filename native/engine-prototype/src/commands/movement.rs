use crate::geometry::{can_place, cells, deployment_rows, distance, empty_for, movement_path};
use crate::model::{
    COMMANDS, Catalog, Command, Failure, Kind, Point, State, Unit, ensure, extra_number, number,
};
use serde_json::{Value, json};

struct Prepared {
    index: usize,
    to: Point,
    starting: bool,
    charge: Option<Kind>,
    moves: f64,
    path: Vec<Point>,
}

pub fn charge_kind(u: &Unit, catalog: &Catalog) -> Option<Kind> {
    let native = std::iter::once(u.kind.clone());
    let inherited = u
        .extra
        .get("traits")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|k| serde_json::from_value::<Kind>(k.clone()).expect("入口已校验能力编号"));
    native.chain(inherited).find(|k| {
        let m = catalog[&k.key()].movement;
        m > 0.0 && m.fract() != 0.0
    })
}
pub fn reserve(u: &Unit, kind: &Kind, catalog: &Catalog) -> Value {
    if *kind == u.kind {
        json!({"charge":u.extra["charge"],"readyCharge":u.extra["readyCharge"],"chargeType":u.extra["chargeType"],"lastCharge":u.extra["lastCharge"]})
    } else {
        u.extra.get("abilityCharges").and_then(|r| r.get(kind.key())).cloned().unwrap_or_else(||
            json!({"charge":0,"readyCharge":0,"chargeType":if catalog[&kind.key()].movement.fract()!=0.0 {"move"} else if kind.is("21") {"skill"} else {"attack"},"lastCharge":-1}))
    }
}

/// getStats 的移动依赖投影：只计算睡眠、冻结、眩晕、操作预算和移动力。
/// 公式和时钟来自 TS，不计算移动入口不读取的攻击、治疗和攻击光环字段。
pub fn movement_stats(s: &State, u: &Unit, catalog: &Catalog) -> (bool, f64, f64) {
    let age = s.turns[&u.owner.to_string()] + u.offset / 2.0 - u.born;
    let sleeping = if catalog[&u.kind.key()].landmark.is_some() {
        u.hp <= 0.0
    } else {
        age <= 0.0 || (!u.silenced && u.has("23") && age < 2.0)
    };
    let locked = sleeping || s.effect(u, "freeze") || s.effect(u, "stun");
    let limit = (if !u.silenced && u.has("u27") && age == 1.0 {
        2.0
    } else {
        1.0
    }) + extra_number(u, "extraOperations");
    let left = if locked {
        0.0
    } else {
        (limit - u.operations).max(0.0)
    };
    let mut movement = catalog[&u.kind.key()].movement;
    if !u.silenced
        && u.has("12")
        && s.units.iter().any(|v| {
            v.side() != u.owner
                && v.id != u.id
                && cells(u, u.at())
                    .iter()
                    .any(|&p| cells(v, v.at()).iter().any(|&q| distance(p, q) == 1.0))
        })
    {
        movement -= 1.0;
    }
    (locked, left, movement)
}
pub fn stage(s: &State, c: &Command, catalog: &Catalog) -> Result<(), Failure> {
    if !COMMANDS.contains(&c.kind.as_str()) {
        return Err(Failure::Unsupported("command-kind"));
    }
    ensure(
        s.version == 2,
        "此命令只接受浩劫2.0局面；旧版对局不会被静默迁移。",
    )?;
    ensure(
        !s.extra
            .get("winner")
            .is_some_and(|w| !w.is_null() && w != false && w != 0),
        "对局已经结束，可悔棋或开新局。",
    )?;
    ensure(
        s.pending.is_empty() || c.kind == "react",
        "请先处理当前待结算效果。",
    )?;
    ensure(
        s.summon_slots != -1.0 || c.kind == "react",
        "当前正在结算回合结束效果。",
    )?;
    let transit = s
        .units
        .iter()
        .find(|u| (u.has("u12") || u.has("u12p")) && u.mode == "move" && !empty_for(s, u, catalog));
    ensure(
        c.kind == "react"
            || transit.is_none_or(|u| c.kind == "move" && c.unit_id.as_deref() == Some(&u.id)),
        "冲撞移动正在经过其他占位，必须先完成弹出或回到空地。",
    )?;
    let giant = c.kind == "skill"
        && c.ability
            .as_ref()
            .or_else(|| s.unit(c.unit_id.as_deref().unwrap_or("")).map(|u| &u.kind))
            .is_some_and(|k| k.is("u7"));
    ensure(
        s.phase != "shrine-draft" || c.kind == "choose-shrine",
        "请先秘密选择神龛。",
    )?;
    ensure(
        s.phase != "shrine-setup"
            || ["deploy", "equip", "activate-aura", "finish-shrine-setup"]
                .contains(&c.kind.as_str()),
        "第0回合仅能部署、装备、启用或储存神龛。",
    )?;
    ensure(
        !s.extra.get("summonOffer").is_some_and(|v| !v.is_null()) || c.kind == "choose-summons",
        "请先从候选召唤中选出两个结果。",
    )?;
    ensure(
        s.phase != "synthesis"
            || giant
            || ["react", "skip-synthesis", "synthesize", "craft"].contains(&c.kind.as_str()),
        "请先选择合成，或跳过合成进入召唤。",
    )?;
    ensure(
        giant
            || s.phase == "play"
            || s.phase == "shrine-setup"
            || [
                "activate-aura",
                "extra-summon",
                "choose-summons",
                "summon",
                "begin",
                "reroll",
                "react",
                "skip-synthesis",
                "choose-shrine",
                "finish-shrine-setup",
                "synthesize",
                "craft",
            ]
            .contains(&c.kind.as_str()),
        "先完成回合开始的召唤选择，再进入行动阶段。",
    )?;
    Ok(())
}
pub fn actor<'a>(s: &'a State, c: &Command, catalog: &Catalog) -> Result<&'a Unit, Failure> {
    let u = s
        .pieces()
        .find(|u| c.unit_id.as_deref() == Some(&u.id))
        .ok_or(Failure::Invalid("请选择仍在场上的随从。"))?;
    ensure(s.phase == "play", "请先完成回合开始阶段。")?;
    ensure(u.side() == s.active, "不是该随从的回合。")?;
    ensure(
        !movement_stats(s, u, catalog).0,
        "该随从正在疲劳、休整、冰冻或眩晕中。",
    )?;
    Ok(u)
}
fn prepare(s: &State, c: &Command, catalog: &Catalog) -> Result<Prepared, Failure> {
    let u = actor(s, c, catalog)?;
    if c.kind == "finish-mode" {
        ensure(
            u.mode == "attack" || (u.mode == "move" && (u.runner() || u.size > 1.0)),
            "没有可提前结束的连续操作。",
        )?;
        ensure(
            !u.runner() || empty_for(s, u, catalog),
            "必须先移到空地，不能结束在另一个棋子、地标或基地内。",
        )?;
        return Ok(Prepared {
            index: s.units.iter().position(|v| v.id == u.id).unwrap_or(0),
            to: u.at(),
            starting: false,
            charge: None,
            moves: 0.0,
            path: vec![],
        });
    }
    let (Some(x), Some(y)) = (c.x, c.y) else {
        return Err(Failure::Invalid("请选择棋盘格。"));
    };
    ensure(x.fract() == 0.0 && y.fract() == 0.0, "请选择棋盘格。")?;
    let to = Point { x, y };
    ensure(catalog[&u.kind.key()].landmark.is_none(), "地标不能移动。")?;
    ensure(
        u.mode == "none" || u.mode == "move",
        "本回合已选择另一操作模式，剩余攻击不能换成移动或技能。",
    )?;
    let starting = u.mode == "none";
    let (_, left, mut limit) = movement_stats(s, u, catalog);
    if starting {
        ensure(left > 0.0, "本回合操作已用完。")?;
    }
    if u.runner() {
        return Err(Failure::Unsupported("collision-move"));
    }
    let charge = charge_kind(u, catalog);
    let ready = charge.as_ref().is_some_and(|k| {
        let r = reserve(u, k, catalog);
        number(&r["readyCharge"]) >= 1.0 && r["chargeType"] == "move"
    });
    let mut moves = if starting { 0.0 } else { u.moves };
    let path = if u.size > 1.0 {
        if starting {
            if charge.is_some() {
                ensure(ready, "分数移动须先蓄力。")?;
                if limit.fract() != 0.0 {
                    limit *= 2.0;
                }
            }
            moves = limit.floor();
        }
        ensure(
            moves > 0.0 && distance(u.at(), to) == 1.0 && can_place(s, u, to, catalog),
            "2×2每次整体向一个方向移动一小格，四格均须合法且不能重叠。",
        )?;
        vec![u.at(), to]
    } else {
        if charge.is_some() {
            ensure(ready, "分数移动需要在回合开始已有1层移动蓄力。")?;
            if limit.fract() != 0.0 {
                limit *= 2.0;
            }
        }
        movement_path(s, u, to, limit, u.has("13") && !u.silenced, catalog)
            .ok_or(Failure::Invalid("移动距离、路径、占位或独行侠禁区不合法。"))?
    };
    Ok(Prepared {
        index: s.units.iter().position(|v| v.id == u.id).unwrap(),
        to,
        starting,
        charge,
        moves,
        path,
    })
}
pub fn finish(u: &mut Unit) {
    if !u.bonus_sequence {
        u.operations += 1.0;
    }
    u.bonus_sequence = false;
    u.mode = "none".into();
    u.shots = 0.0;
    u.moves = 0.0;
}
pub fn choose_skill(s: &mut State, id: &str, catalog: &Catalog) -> Result<(), Failure> {
    let u = s
        .unit(id)
        .ok_or(Failure::Invalid("请选择仍在场上的随从。"))?;
    ensure(
        u.mode == "none" || u.mode == "skill",
        "本回合已选择另一操作模式，剩余攻击不能换成移动或技能。",
    )?;
    if u.mode == "none" {
        ensure(
            crate::stats::stats(s, u, catalog).operations_left > 0.0,
            "本回合操作已用完。",
        )?;
        let u = s.unit_mut(id).unwrap();
        u.mode = "skill".into();
        u.shots = 0.0;
        u.moves = 0.0;
    }
    Ok(())
}
pub fn consume(u: &mut Unit, kind: &Kind, catalog: &Catalog) {
    if *kind == u.kind {
        u.extra.insert("charge".into(), json!(0));
        u.extra.insert("readyCharge".into(), json!(0));
    } else {
        let mut r = reserve(u, kind, catalog);
        r["charge"] = json!(0);
        r["readyCharge"] = json!(0);
        // TS withAbilityCharge 只回写四个独立资源字段。
        let r = json!({"charge":r["charge"],"readyCharge":r["readyCharge"],"chargeType":r["chargeType"],"lastCharge":r["lastCharge"]});
        u.extra.entry("abilityCharges").or_insert_with(|| json!({}))[kind.key()] = r;
    }
}
pub fn sync(s: &mut State, catalog: &Catalog) {
    crate::resolution::normalize_guards(s);
    sync_banners(s, catalog);
    crate::stats::prune_siphons(s, catalog);
    s.deploy_rows = json!({"1":deployment_rows(s,1),"2":deployment_rows(s,2)});
}
pub fn sync_banners(s: &mut State, catalog: &Catalog) {
    let mut banners = [0.0; 3];
    for l in s.landmarks() {
        if l.live() && !l.silenced && l.has("s8") {
            banners[l.owner] += 10.0;
        }
    }
    for u in s.units.iter_mut().chain(s.landmarks.iter_mut().flatten()) {
        let eligible = catalog[&u.kind.key()].landmark.is_none() || u.live();
        let next = if eligible && u.side() == u.owner {
            banners[u.owner]
        } else {
            0.0
        };
        let old = extra_number(u, "bannerHp");
        if next != old {
            u.max_hp += next - old;
            if next > old && u.hp > 0.0 {
                u.hp += next - old;
            }
            u.extra.insert("bannerHp".into(), json!(next));
            if u.hp > u.max_hp {
                u.extra.insert("overMaxFromBanner".into(), json!(true));
            }
        }
        if u.hp <= u.max_hp {
            u.extra.remove("overMaxFromBanner");
        }
    }
}

/// 完整的切片命令：先只读准备，再复制并结算；任何失败不改输入，也不执行 TS 回退。
pub fn apply(previous: &State, c: &Command, catalog: &Catalog) -> Result<State, Failure> {
    stage(previous, c, catalog)?;
    let p = prepare(previous, c, catalog)?;
    let mut s = previous.clone();
    s.events.clear();
    if c.kind == "finish-mode" {
        let u = s.unit_mut(c.unit_id.as_deref().unwrap()).unwrap();
        let was_move = u.mode == "move";
        finish(u);
        if was_move && u.weapon("u16") {
            u.bonus_attacks += 1.0;
        }
    } else {
        let u = &mut s.units[p.index];
        let from = u.at();
        let actor = u.actor_event();
        if p.starting {
            u.mode = "move".into();
            u.shots = 0.0;
            u.moves = 0.0;
        }
        let id = format!("e{}", s.serial);
        s.serial += 1;
        s.events.push(json!({"type":"move","id":id,"causeId":id,"actor":actor,"subject":actor,"from":from,"to":p.to,"path":p.path,"unitId":u.id,"owner":u.owner}));
        u.x = p.to.x;
        u.y = p.to.y;
        if let Some(kind) = &p.charge
            && (u.size == 1.0 || p.starting)
        {
            consume(u, kind, catalog);
        }
        if u.size > 1.0 {
            u.moves = p.moves - 1.0;
        }
        if u.size == 1.0 || u.moves == 0.0 {
            finish(u);
            if u.weapon("u16") {
                u.bonus_attacks += 1.0;
            }
        }
    }
    sync(&mut s, catalog);
    crate::resolution::terminal(&mut s, &mut crate::resolution::Resolution::default());
    Ok(s)
}
