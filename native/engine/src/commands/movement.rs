use crate::geometry::{can_place, cells, deployment_rows, distance, empty_for};
use crate::model::{
    COMMANDS, Catalog, Command, Failure, Kind, Point, State, Unit, ensure, extra_number,
};
use serde_json::json;

pub struct Prepared {
    pub to: Point,
    pub source: MoveSource,
    pub path: Option<Vec<Point>>,
}
pub struct MoveSource {
    pub unit: Unit,
    pub starting: bool,
    pub charge: Option<Kind>,
    pub moves: f64,
    limit: f64,
    field: std::cell::OnceCell<crate::geometry::MovementField>,
}

pub fn charge_kind(u: &Unit, catalog: &Catalog) -> Option<Kind> {
    let native = std::iter::once(u.kind.clone());
    let inherited = u.traits.iter().flatten().cloned();
    native.chain(inherited).find(|k| {
        let m = catalog.by_kind(k).movement;
        m > 0.0 && m.fract() != 0.0
    })
}
pub fn reserve(u: &Unit, kind: &Kind, catalog: &Catalog) -> crate::model::Charge {
    if *kind == u.kind {
        u.reserve
    } else {
        u.ability_charges
            .as_ref()
            .and_then(|r| r.get(&kind.key()))
            .copied()
            .unwrap_or_else(|| crate::model::Charge {
                charge: 0.0,
                ready_charge: 0.0,
                charge_type: if catalog.by_kind(kind).movement.fract() != 0.0 {
                    crate::model::ChargeMode::Move
                } else if kind.is("21") {
                    crate::model::ChargeMode::Skill
                } else {
                    crate::model::ChargeMode::Attack
                },
                last_charge: -1.0,
            })
    }
}

/// getStats 的移动依赖投影：只计算睡眠、冻结、眩晕、操作预算和移动力。
/// 公式和时钟来自 TS，不计算移动入口不读取的攻击、治疗和攻击光环字段。
pub fn movement_stats(s: &State, u: &Unit, catalog: &Catalog) -> (bool, f64, f64) {
    let age = s.turns[&u.owner.to_string()] + u.offset / 2.0 - u.born;
    let sleeping = if catalog.by_kind(&u.kind).landmark.is_some() {
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
    let mut movement = catalog.by_kind(&u.kind).movement;
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
    stage_with_transit(s, c, || transit(s, catalog).map(str::to_owned))
}
pub fn transit<'a>(s: &'a State, catalog: &Catalog) -> Option<&'a str> {
    s.units
        .iter()
        .find(|u| (u.has("u12") || u.has("u12p")) && u.mode == "move" && !empty_for(s, u, catalog))
        .map(|u| u.id.as_str())
}
pub fn stage_with_transit(
    s: &State,
    c: &Command,
    transit: impl FnOnce() -> Option<String>,
) -> Result<(), Failure> {
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
    let transit = transit();
    ensure(
        c.kind == "react"
            || transit
                .as_deref()
                .is_none_or(|id| c.kind == "move" && c.unit_id.as_deref() == Some(id)),
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
pub fn prepare_finish(s: &State, c: &Command, catalog: &Catalog) -> Result<(), Failure> {
    let u = actor(s, c, catalog)?;
    ensure(
        u.mode == "attack" || (u.mode == "move" && (u.runner() || u.size > 1.0)),
        "没有可提前结束的连续操作。",
    )?;
    ensure(
        !u.runner() || empty_for(s, u, catalog),
        "必须先移到空地，不能结束在另一个棋子、地标或基地内。",
    )
}
/// 模式和资源只在当前观察内准备；规则执行复用相同函数，目标和路径仍逐个精确验证。
pub fn move_source(s: &State, u: Unit, catalog: &Catalog) -> Result<MoveSource, Failure> {
    ensure(
        catalog.by_kind(&u.kind).landmark.is_none(),
        "地标不能移动。",
    )?;
    ensure(
        u.mode == "none" || u.mode == "move",
        "本回合已选择另一操作模式，剩余攻击不能换成移动或技能。",
    )?;
    let starting = u.mode == "none";
    let (_, left, mut limit) = movement_stats(s, &u, catalog);
    if starting {
        ensure(left > 0.0, "本回合操作已用完。")?;
    }
    let charge = charge_kind(&u, catalog);
    let ready = charge.as_ref().is_some_and(|k| {
        let r = reserve(&u, k, catalog);
        r.ready_charge >= 1.0 && r.charge_type.as_str() == "move"
    });
    let mut moves = if starting { 0.0 } else { u.moves };
    if u.runner() {
        if starting {
            ensure(ready, "需要在回合开始已有1层移动蓄力。")?;
            moves = 5.0;
        }
    } else if u.size > 1.0 {
        if starting {
            if charge.is_some() {
                ensure(ready, "分数移动须先蓄力。")?;
                if limit.fract() != 0.0 {
                    limit *= 2.0;
                }
            }
            moves = limit.floor();
        }
    } else if charge.is_some() {
        ensure(ready, "分数移动需要在回合开始已有1层移动蓄力。")?;
        if limit.fract() != 0.0 {
            limit *= 2.0;
        }
    }
    Ok(MoveSource {
        unit: u,
        starting,
        charge,
        moves,
        limit,
        field: Default::default(),
    })
}
pub fn move_destination(
    s: &State,
    source: &MoveSource,
    to: Point,
    catalog: &Catalog,
    query: Option<&crate::geometry::PlacementQuery>,
) -> Result<Option<Vec<Point>>, Failure> {
    let u = &source.unit;
    let place = |p| {
        query.map_or_else(
            || can_place(s, u, p, catalog),
            |q| q.can_place(s, u, p, catalog),
        )
    };
    if u.runner() {
        ensure(
            source.moves > 0.0
                && distance(u.at(), to) == 1.0
                && if u.size > 1.0 {
                    can_place(s, u, to, catalog)
                } else {
                    crate::geometry::inside(to)
                },
            "每次沿四向移动一格，不能越界或使大体型重叠。",
        )?;
        return Ok(None);
    }
    let path = if u.size > 1.0 {
        ensure(
            source.moves > 0.0 && distance(u.at(), to) == 1.0 && place(to),
            "2×2每次整体向一个方向移动一小格，四格均须合法且不能重叠。",
        )?;
        vec![u.at(), to]
    } else {
        (if query.is_some() {
            source
                .field
                .get_or_init(|| {
                    crate::geometry::MovementField::new(
                        u,
                        source.limit,
                        u.has("13") && !u.silenced,
                        place,
                    )
                })
                .path(to)
        } else {
            crate::geometry::movement_with_place(
                u,
                to,
                source.limit,
                u.has("13") && !u.silenced,
                place,
            )
        })
        .ok_or(Failure::Invalid("移动距离、路径、占位或独行侠禁区不合法。"))?
    };
    Ok(Some(path))
}
pub fn prepare_move(s: &State, c: &Command, catalog: &Catalog) -> Result<Prepared, Failure> {
    let u = actor(s, c, catalog)?.fork();
    let to = crate::shrines::point(c)?;
    let source = move_source(s, u, catalog)?;
    let path = move_destination(s, &source, to, catalog, None)?;
    Ok(Prepared { source, to, path })
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
    let mut r = reserve(u, kind, catalog);
    r.charge = 0.0;
    r.ready_charge = 0.0;
    if *kind == u.kind {
        u.reserve = r;
    } else {
        u.ability_charges
            .get_or_insert_with(Default::default)
            .insert(kind.key(), r);
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
        let eligible = catalog.by_kind(&u.kind).landmark.is_none() || u.live();
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
        if u.hp <= u.max_hp && u.extra.contains_key("overMaxFromBanner") {
            u.extra.remove("overMaxFromBanner");
        }
    }
}

/// 完整的切片命令：先只读准备，再复制并结算；任何失败不改输入，也不执行 TS 回退。
pub fn apply(previous: &State, c: &Command, catalog: &Catalog) -> Result<State, Failure> {
    let prepared = if c.kind == "finish-mode" {
        prepare_finish(previous, c, catalog)?;
        None
    } else {
        Some(prepare_move(previous, c, catalog)?)
    };
    let mut s = previous.fork();
    s.events.clear();
    if c.kind == "finish-mode" {
        let u = s.unit_mut(c.unit_id.as_deref().unwrap()).unwrap();
        let was_move = u.mode == "move";
        finish(u);
        if was_move && u.weapon("u16") {
            u.bonus_attacks += 1.0;
        }
    } else {
        let p = prepared.unwrap();
        let u = s
            .units
            .iter_mut()
            .find(|u| u.id == p.source.unit.id)
            .unwrap();
        let from = u.at();
        let actor = u.actor_event();
        if p.source.starting {
            u.mode = "move".into();
            u.shots = 0.0;
            u.moves = 0.0;
        }
        let id = format!("e{}", s.serial);
        s.serial += 1;
        s.events.push(json!({"type":"move","id":id,"causeId":id,"actor":actor,"subject":actor,"from":from,"to":p.to,"path":p.path,"unitId":u.id,"owner":u.owner}));
        u.x = p.to.x;
        u.y = p.to.y;
        if let Some(kind) = &p.source.charge
            && (u.size == 1.0 || p.source.starting)
        {
            consume(u, kind, catalog);
        }
        if u.size > 1.0 {
            u.moves = p.source.moves - 1.0;
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
