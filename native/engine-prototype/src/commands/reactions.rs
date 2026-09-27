use crate::combat::{Options, perform};
use crate::damage::{Source, alive, damage, kill, protected};
use crate::geometry::{
    Target, attack_path, can_place, distance, empty_for, inside, neighbors, targets,
};
use crate::model::{Catalog, Command, Failure, Point, State, Unit, ensure, number};
use crate::movement::{actor, consume, finish, movement_stats, reserve, stage, sync};
use crate::resolution::{Resolution, add_unit, normalize_guards, template, terminal};
use crate::stats::stats;
use serde_json::{Value, json};

fn point(c: &Command) -> Result<Point, Failure> {
    let (Some(x), Some(y)) = (c.x, c.y) else {
        return Err(Failure::Invalid("请选择棋盘格。"));
    };
    ensure(x.fract() == 0.0 && y.fract() == 0.0, "请选择棋盘格。")?;
    Ok(Point { x, y })
}
fn can_enter(s: &State, u: &Unit, p: Point, catalog: &Catalog) -> bool {
    if u.size > 1.0 {
        can_place(s, u, p, catalog)
    } else {
        inside(p)
    }
}
fn reachable_exit(s: &State, u: &Unit, steps: f64, catalog: &Catalog) -> bool {
    let mut queue = vec![(u.at(), 0)];
    let mut seen = vec![u.at()];
    let mut i = 0;
    while i < queue.len() {
        let (p, n) = queue[i];
        i += 1;
        let mut moved = u.clone();
        moved.x = p.x;
        moved.y = p.y;
        if empty_for(s, &moved, catalog) {
            return true;
        }
        if n as f64 >= steps {
            continue;
        }
        for next in neighbors(p) {
            if !seen.contains(&next) && can_enter(s, u, next, catalog) {
                seen.push(next);
                queue.push((next, n + 1));
            }
        }
    }
    false
}
fn collision(s: &State, u: &Unit, to: Point) -> Option<Target> {
    targets(s).into_iter().find(|t| {
        t.id != u.id
            && t.unit
                .as_ref()
                .map(|v| v.side() != u.owner)
                .unwrap_or(t.owner != u.owner)
            && t.footprint().contains(&to)
    })
}
fn finish_rush(s: &mut State, id: &str, catalog: &Catalog) {
    if let Some(u) = s.unit(id).cloned()
        && u.moves == 0.0
        && empty_for(s, &u, catalog)
    {
        let u = s.unit_mut(id).unwrap();
        finish(u);
        if u.weapon("u16") {
            u.bonus_attacks += 1.0;
        }
    }
}

/// 冲撞逐格执行，占位必须保证剩余步数内可离开；碰撞与弹出反应保留相同预算。
/// 路径准备只读，实际伤害和随机只写副本；任何失败都保留调用方旧局面。
pub fn move_runner(
    previous: &State,
    c: &Command,
    catalog: &Catalog,
    preview: bool,
) -> Result<State, Failure> {
    stage(previous, c, catalog)?;
    let mut u = actor(previous, c, catalog)?.clone();
    let to = point(c)?;
    ensure(catalog[&u.kind.key()].landmark.is_none(), "地标不能移动。")?;
    let starting = u.mode == "none";
    ensure(
        starting || u.mode == "move",
        "本回合已选择另一操作模式，剩余攻击不能换成移动或技能。",
    )?;
    if starting {
        ensure(
            movement_stats(previous, &u, catalog).1 > 0.0,
            "本回合操作已用完。",
        )?;
        u.mode = "move".into();
        u.shots = 0.0;
        u.moves = 0.0;
    }
    let charged = u
        .kinds()
        .into_iter()
        .find(|k| catalog[&k.key()].movement > 0.0 && catalog[&k.key()].movement.fract() != 0.0);
    if starting {
        ensure(
            charged.as_ref().is_some_and(|k| {
                let r = reserve(&u, k, catalog);
                number(&r["readyCharge"]) >= 1.0 && r["chargeType"] == "move"
            }),
            "需要在回合开始已有1层移动蓄力。",
        )?;
        u.moves = 5.0;
    }
    ensure(
        u.moves > 0.0 && distance(u.at(), to) == 1.0 && can_enter(previous, &u, to, catalog),
        "每次沿四向移动一格，不能越界或使大体型重叠。",
    )?;
    let mut s = previous.clone();
    normalize_guards(&mut s);
    s.events.clear();
    let mut ctx = Resolution {
        preview,
        facts: Some(json!({"actor":u.actor_event(),"action":"rush"})),
        ..Default::default()
    };
    if starting {
        consume(&mut u, charged.as_ref().unwrap(), catalog);
    }
    let target = collision(&s, &u, to);
    ctx.emit(&mut s,json!({"type":"move","from":u.actor_event(),"to":to,"unitId":u.id,"owner":u.owner,"text":"冲撞"}),None);
    u.x = to.x;
    u.y = to.y;
    u.moves -= 1.0;
    *s.unit_mut(&u.id).unwrap() = u.clone();
    if let Some(t) = &target {
        damage(
            &mut s,
            t,
            30.0,
            &Source::new(&u, "collision"),
            catalog,
            &mut ctx,
        )?;
    }
    if alive(&s, &u.id) {
        let u = s.unit(&u.id).unwrap().clone();
        if u.has("u12") && target.is_some() {
            s.pending.insert(
                0,
                json!({"kind":"bounce","owner":u.owner,"source":u,"amount":30}),
            );
        } else {
            ensure(
                empty_for(&s, &u, catalog)
                    || (u.moves > 0.0 && reachable_exit(&s, &u, u.moves, catalog)),
                "剩余移动次数无法返回空地，不能进行这次冲撞。",
            )?;
        }
        finish_rush(&mut s, &u.id, catalog);
    }
    ctx.facts = None;
    sync(&mut s, catalog);
    terminal(&mut s, &mut ctx);
    Ok(s)
}

/// 与 TS hutSpawnPoints 共用规则顺序；同一公开局面的参数树可复用落点结果。
pub fn hut_points(s: &State, r: &Value, catalog: &Catalog) -> Vec<Point> {
    let Some(hut) = s.units.iter().find(|u| r["source"]["id"] == u.id) else {
        return vec![];
    };
    let owner = number(&r["owner"]) as usize;
    if hut.silenced || hut.owner != owner || hut.max_hp < 10.0 {
        return vec![];
    }
    let ghost = template(s, "20", owner, Point { x: 1.0, y: 1.0 }, "preview", catalog);
    let range = stats(s, hut, catalog).range;
    crate::geometry::all_cells()
        .filter(|p| {
            can_place(s, &ghost, *p, catalog)
                && attack_path(
                    s,
                    hut,
                    &crate::geometry::point_target(*p),
                    range,
                    None,
                    false,
                )
                .is_some()
        })
        .collect()
}
pub fn prepare_hut(
    s: &State,
    r: &Value,
    c: &Command,
    destinations: &[Point],
) -> Result<(), Failure> {
    let Some(hut) = s.units.iter().find(|u| r["source"]["id"] == u.id) else {
        return Ok(());
    };
    if destinations.is_empty() {
        return Ok(());
    }
    if c.x.is_none() {
        return ensure(!hut.has("citadel"), "王城死亡召唤必须选择合法落点。");
    }
    let to = point(c)?;
    ensure(
        destinations.contains(&to),
        "小屋召唤须在其范围内的合法空地。",
    )?;
    ensure(hut.max_hp >= 10.0, "小屋生命上限不足。")
}
/// 从权威队首反应读取来源和固定目标，不信任命令替换目标；修改外层命令副本。
/// 小屋/死亡射击/弹出可继续排队；结束效果的最后一个反应完成后才切换实际回合。
pub fn react(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    ensure(!s.pending.is_empty(), "没有待结算反应。")?;
    let r = s.pending.remove(0);
    let source: Unit = serde_json::from_value(r["source"].clone())
        .map_err(|_| Failure::Unsupported("reaction-source"))?;
    let owner = r["owner"].as_u64().unwrap() as usize;
    match r["kind"].as_str().unwrap_or("") {
        "bounce" => {
            if let Some(mut u) = s.unit(&source.id).cloned() {
                let to = point(c)?;
                ensure(
                    distance(u.at(), to) == 1.0 && can_enter(s, &u, to, catalog),
                    "必须向四向相邻合法格弹出，不能越界或使大体型重叠。",
                )?;
                let target = collision(s, &u, to);
                ctx.emit(s,json!({"type":"move","from":u.actor_event(),"to":to,"unitId":u.id,"owner":u.owner,"text":"免费弹出"}),None);
                u.x = to.x;
                u.y = to.y;
                *s.unit_mut(&u.id).unwrap() = u.clone();
                if let Some(t) = &target {
                    damage(s, t, 30.0, &Source::new(&u, "collision"), catalog, ctx)?;
                }
                if let Some(u) = s.unit(&u.id).cloned() {
                    if target.is_some() || !empty_for(s, &u, catalog) {
                        s.pending.insert(
                            0,
                            json!({"kind":"bounce","owner":u.owner,"source":u,"amount":30}),
                        );
                    } else {
                        finish_rush(s, &u.id, catalog);
                    }
                }
            }
        }
        "hit-pull" => {
            if c.mode.as_deref() == Some("pull") {
                let u = s.unit(&source.id).cloned();
                let victim = r["targetId"].as_str().and_then(|id| s.unit(id)).cloned();
                let pair = u.zip(victim).filter(|(u, v)| {
                    !v.has("5") && u.owner == owner && !u.silenced && v.side() != owner
                });
                let destination = pair.and_then(|(u, v)| {
                    let to = Point {
                        x: u.x,
                        y: if u.owner == 1 {
                            u.y + u.size
                        } else {
                            u.y - v.size
                        },
                    };
                    can_place(s, &v, to, catalog).then_some((u, v, to))
                });
                let (u, v, to) = destination.ok_or(Failure::Invalid(
                    "原命中目标或来源已失效，或身前没有合法完整落位。",
                ))?;
                if !protected(
                    s,
                    &Target::from(&v),
                    &Source::new(&u, "skill"),
                    catalog,
                    ctx,
                )? {
                    ctx.emit(s,json!({"type":"move","action":"pull","stage":"trigger","from":v.actor_event(),"to":to,"unitId":v.id,"owner":v.owner}),None);
                    let v = s.unit_mut(&v.id).unwrap();
                    v.x = to.x;
                    v.y = to.y;
                }
            }
        }
        "hut-spawn" => {
            let destinations = hut_points(s, &r, catalog);
            prepare_hut(s, &r, c, &destinations)?;
            if let Some(hut) = s.unit(&source.id).cloned()
                && !destinations.is_empty()
            {
                if c.x.is_none() {
                    ctx.emit(
                        s,
                        json!({"type":"skill","owner":owner,"text":"放弃召唤"}),
                        None,
                    );
                } else {
                    let to = point(c)?;
                    add_unit(s, "20", owner, to, catalog, ctx)?;
                    let u = s.unit_mut(&hut.id).unwrap();
                    u.max_hp = (u.max_hp - 10.0).max(0.0);
                    u.hp = u.hp.min(u.max_hp);
                    let u = u.clone();
                    if u.hp <= 0.0 {
                        kill(s, &u, &Source::new(&u, "sacrifice"), catalog, ctx)?;
                    }
                }
            }
        }
        "death-shot" | "reflect" => {
            if let Some(target) = c.target_id.as_ref().filter(|id| !id.is_empty()) {
                let t = crate::geometry::find_target(s, target)
                    .ok_or(Failure::Invalid("请选择有效的目标。"))?;
                if r["kind"] == "death-shot" {
                    perform(
                        s,
                        &source,
                        &t,
                        &Options {
                            reactive: true,
                            unlimited: true,
                            ..Default::default()
                        },
                        catalog,
                        ctx,
                    )?;
                } else {
                    ensure(
                        t.unit
                            .as_ref()
                            .map(|u| u.side() != owner)
                            .unwrap_or(t.owner != owner),
                        "伤害转化只能指向敌方或中立。",
                    )?;
                    let path = attack_path(
                        s,
                        &source,
                        &t,
                        stats(s, &source, catalog).range,
                        None,
                        false,
                    )
                    .ok_or(Failure::Invalid("目标不在伤害转化器范围内。"))?;
                    ctx.emit(s,json!({"type":"attack","from":source.actor_event(),"to":t.actor(),"path":path,"owner":owner,"text":"伤害转化"}),None);
                    damage(
                        s,
                        &t,
                        number(&r["amount"]),
                        &Source::new(&source, "reflect"),
                        catalog,
                        ctx,
                    )?;
                }
            } else {
                ctx.emit(
                    s,
                    json!({"type":"skill","owner":owner,"text":"放弃效果"}),
                    None,
                );
            }
        }
        _ => return Err(Failure::Unsupported("reaction-kind")),
    }
    if s.pending.is_empty()
        && s.summon_slots == -1.0
        && number(&s.bases["1"]) > 0.0
        && number(&s.bases["2"]) > 0.0
    {
        crate::lifecycle::switch(s, catalog, ctx)?;
    }
    Ok(())
}
