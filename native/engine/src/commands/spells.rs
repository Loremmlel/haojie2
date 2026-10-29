//! 法术先校验全部参数，再反制、支付和结算。伤害来源保留基地身份，AOE 按覆盖格逐包处理。
use crate::abilities::{target, unit};
use crate::damage::{Source, area_damage, heal, kill, protected};
use crate::geometry::{Target, inside, targets};
use crate::model::{Catalog, Command, Failure, Point, State, Unit, ensure, extra_number};
use crate::preparation::{card, draw, kind, remove_card};
use crate::resolution::{Resolution, add_effect};
use serde_json::{Value, json};

pub fn cast(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let previous = ctx.replace_facts(|| {
        let selected = card(s, c);
        let mut facts = json!({"stage":"apply"});
        if let Some(card) = selected.as_ref() {
            let k = kind(&card["kind"]).key();
            facts["ability"] = card["kind"].clone();
            facts["action"] = json!(match k.as_str() {
                "8" => "bomb",
                "17" => "ward",
                "18" => "execution",
                "22" => "conversion",
                "u9" => "storm",
                "u17" => "clock",
                "u26" => "inner-fire",
                _ => "buff",
            });
            if k == "8" || k == "u9" {
                facts["area"] = json!(
                    crate::lifecycle::all_cells()
                        .filter(|p| hits(&k, c, *p))
                        .collect::<Vec<_>>()
                );
            }
        }
        if let Some(t) = c
            .target_id
            .as_deref()
            .and_then(|id| crate::geometry::find_target(s, id))
        {
            facts["subject"] = t.actor();
        }
        facts
    });
    resolve(s, c, catalog, ctx)?;
    ctx.facts = previous;
    Ok(())
}
struct Prepared {
    owner: usize,
    card: Value,
    target: Option<Target>,
}
/// 对应 TS prepareCast；选择校验完整保留在反制随机数与支付之前。
fn prepare(s: &State, c: &Command, catalog: &Catalog) -> Result<Prepared, Failure> {
    let owner = s.active;
    let card = card(s, c)
        .filter(|v| catalog[&kind(&v["kind"]).key()].spell.is_some())
        .ok_or(Failure::Invalid("请选择法术牌。"))?;
    let k = kind(&card["kind"]).key();
    let t = if c.target_id.is_some() {
        Some(target(s, c.target_id.as_deref())?)
    } else {
        None
    };
    if ["17", "18", "22", "u17"].contains(&k.as_str()) {
        ensure(
            t.as_ref()
                .and_then(|t| t.unit.as_ref())
                .is_some_and(|u| u.side() == owner),
            "请选择友方随从。",
        )?;
    }
    if k == "22" {
        ensure(
            !t.as_ref().unwrap().unit.as_ref().unwrap().has("5"),
            "大肉比不能使用策反。",
        )?;
    }
    if k == "u26" {
        ensure(
            t.as_ref().is_some_and(|t| t.unit.is_some()),
            "心灵之火只能选择随从。",
        )?;
    }
    if k == "8" {
        let p = crate::shrines::point(c)?;
        ensure(
            inside(p) && p.x <= 8.0 && p.y <= 12.0,
            "爆弹须选择完整2×2区域的左上格。",
        )?;
    }
    if k == "u9" {
        ensure(
            (c.mode.as_deref() == Some("row")
                && c.row
                    .is_some_and(|v| v.fract() == 0.0 && (1.0..=13.0).contains(&v)))
                || (c.mode.as_deref() == Some("column")
                    && c.column
                        .is_some_and(|v| v.fract() == 0.0 && (1.0..=9.0).contains(&v))),
            "烈焰风暴须选择整行或整列。",
        )?;
    }
    if k == "25" && c.mode.as_deref() == Some("double") {
        let ids = c.sacrifice_ids.as_deref().unwrap_or(&[]);
        ensure(
            ids.len() == 2 && ids[0] != ids[1],
            "请选择两个不同的半血以上友方。",
        )?;
        let units = ids
            .iter()
            .map(|id| unit(s, Some(id)))
            .collect::<Result<Vec<_>, _>>()?;
        ensure(
            units
                .iter()
                .all(|v| v.side() == owner && v.hp * 2.0 >= v.max_hp && !v.kind.is("u25")),
            "不可献祭克隆军团，且友方需至少半血。",
        )?;
    }
    Ok(Prepared {
        owner,
        card,
        target: t,
    })
}
fn counters(s: &State, owner: usize) -> Vec<Unit> {
    let mut mages: Vec<_> = s
        .units
        .iter()
        .filter(|u| {
            u.owner != owner
                && !u.silenced
                && (u.has("archmage") || u.has("u3"))
                && !(u.has("archmage") && s.effect(u, "freeze"))
        })
        .cloned()
        .collect();
    mages.sort_by(|a, b| extra_number(a, "deployedAt").total_cmp(&extra_number(b, "deployedAt")));
    mages
}
fn hits(k: &str, c: &Command, p: Point) -> bool {
    if k == "8" {
        c.x.is_some_and(|x| p.x >= x && p.x <= x + 1.0)
            && c.y.is_some_and(|y| p.y >= y && p.y <= y + 1.0)
    } else if c.mode.as_deref() == Some("row") {
        Some(p.y) == c.row
    } else {
        Some(p.x) == c.column
    }
}
fn victims(s: &State, owner: usize, k: &str, c: &Command) -> Vec<Target> {
    targets(s)
        .into_iter()
        .filter(|t| {
            t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != owner
                && t.footprint().into_iter().any(|p| hits(k, c, p))
        })
        .collect()
}
/// 对应 TS hasCastVictim；只读存在性查询不构造拥有型目标和覆盖格数组。
fn has_victim(s: &State, owner: usize, k: &str, c: &Command) -> bool {
    s.pieces()
        .any(|u| u.side() != owner && crate::geometry::cell_iter(u, u.at()).any(|p| hits(k, c, p)))
        || hits(k, c, crate::geometry::base_point(3 - owner))
}
/// 对应 TS prepareCastInspection；纯附魔无需模拟，抽牌与反制停在原随机边界。
pub fn prepare_inspection(s: &State, c: &Command, catalog: &Catalog) -> Result<bool, Failure> {
    let prepared = prepare(s, c, catalog)?;
    if !counters(s, prepared.owner).is_empty() {
        return Err(Failure::Uncertain);
    }
    let k = kind(&prepared.card["kind"]).key();
    if ["17", "18", "22"].contains(&k.as_str()) {
        return Ok(true);
    }
    if k == "25" && c.mode.as_deref() != Some("double") && c.chosen_kind.is_none() {
        return Err(Failure::Uncertain);
    }
    Ok((k == "8" || k == "u9") && !has_victim(s, prepared.owner, &k, c))
}
/// 对应 TS prepareCastPoints：爆弹在无反制时共用卡牌与公共选择资格，受击落点仍完整预检。
pub fn points(
    s: &State,
    c: &Command,
    catalog: &Catalog,
) -> Result<Option<Vec<(Point, bool)>>, Failure> {
    let Some(card) = card(s, c).filter(|card| card["kind"] == 8) else {
        return Ok(None);
    };
    // 坐标由点域提供；其余前缀条件仍走与正式施法相同的准备入口。
    let prepared = prepare(
        s,
        &Command {
            x: Some(1.0),
            y: Some(1.0),
            ..c.clone()
        },
        catalog,
    )?;
    if !counters(s, prepared.owner).is_empty() {
        return Ok(None);
    }
    let k = kind(&card["kind"]).key();
    Ok(Some(
        crate::geometry::all_cells()
            .filter(|at| at.x <= 8.0 && at.y <= 12.0)
            .map(|at| {
                let candidate = Command {
                    x: Some(at.x),
                    y: Some(at.y),
                    ..c.clone()
                };
                (at, !has_victim(s, prepared.owner, &k, &candidate))
            })
            .collect(),
    ))
}
fn resolve(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    let Prepared {
        owner,
        card,
        target: t,
    } = prepare(s, c, catalog)?;
    let k = kind(&card["kind"]).key();
    let source = Source::effect(owner, "spell");
    for mage in counters(s, owner) {
        let chance = if mage.has("archmage") {
            catalog.rule("/archmageCounterChance")
        } else {
            1.0 / 3.0
        };
        if s.random_with(ctx.preview, &[0., chance, 1.])? < chance {
            if mage.has("archmage") {
                s.unit_mut(&mage.id).unwrap().max_hp += catalog.rule("/archmageCounterHealth");
                heal(
                    s,
                    &crate::geometry::Target::from(&mage),
                    catalog.rule("/archmageCounterHealth"),
                    catalog,
                    ctx,
                )?;
            }
            remove_card(s, &card);
            ctx.emit(s,json!({"type":"shield","to":mage.actor_event(),"owner":mage.owner,"action":"counter","stage":"blocked","text":"法术反制"}),Some(format!("{}被反制并消耗",catalog[&k].name)));
            return Ok(());
        }
    }
    match k.as_str() {
        "8" | "u9" => {
            if k == "8" {
                ctx.emit(s,json!({"type":"skill","to":{"x":c.x.unwrap()+0.5,"y":c.y.unwrap()+0.5},"owner":owner,"text":"爆弹"}),None);
            } else {
                let hazard = json!({"id":format!("hazard{}",s.serial),"owner":owner,"axis":c.mode,"line":if c.mode.as_deref()==Some("row"){c.row}else{c.column},"due":s.ply+2.0});
                s.serial += 1;
                s.extra
                    .get_mut("hazards")
                    .unwrap()
                    .as_array_mut()
                    .unwrap()
                    .push(hazard);
            }
            let victims = victims(s, owner, &k, c);
            area_damage(
                s,
                &victims,
                |p| if hits(&k, c, p) { 20.0 } else { 0.0 },
                &source,
                catalog,
                ctx,
            )?;
        }
        "17" | "18" | "22" => {
            let t = t.as_ref().unwrap();
            let effect = match k.as_str() {
                "17" => "immune",
                "18" => "execute",
                _ => "convert",
            };
            s.unit_mut(&t.id)
                .unwrap()
                .effects
                .retain(|e| e.kind != effect);
            add_effect(
                s,
                &t.id,
                effect,
                owner,
                if k == "17" { 2.0 } else { 1.0 },
                None,
                None,
                k != "17",
            );
            if k != "17" {
                let ply = s.ply;
                let e = s.unit_mut(&t.id).unwrap().effects.last_mut().unwrap();
                e.from = ply + 2.0;
                e.until = ply + 3.0;
            }
            ctx.emit(
                s,
                json!({"type":"shield","to":t.actor(),"owner":owner,"text":catalog[&k].name}),
                None,
            );
        }
        "25" => {
            let double = c.mode.as_deref() == Some("double");
            if double {
                for id in c.sacrifice_ids.as_ref().unwrap() {
                    let v = unit(s, Some(id))?;
                    kill(s, &v, &Source::effect(owner, "sacrifice"), catalog, ctx)?;
                }
            }
            let ultimate = s.extra.get("mode") == Some(&json!("shrine"));
            for i in 0..if double { 2 } else { 1 } {
                draw(
                    s,
                    ultimate,
                    if i == 0 { c.chosen_kind.as_ref() } else { None },
                    catalog,
                    ctx,
                )?;
            }
        }
        "u17" => crate::lifecycle::advance(s, &t.as_ref().unwrap().id, catalog, ctx)?,
        "u26" => {
            let t = t.as_ref().unwrap();
            if !protected(s, t, &source, catalog, ctx)? {
                let now = s.ply + s.unit(&t.id).unwrap().offset;
                s.unit_mut(&t.id)
                    .unwrap()
                    .effects
                    .retain(|e| e.kind != "inner-fire");
                add_effect(
                    s,
                    &t.id,
                    "inner-fire",
                    owner,
                    9007199254740991.0 - now,
                    None,
                    None,
                    false,
                );
            }
        }
        _ => return Err(Failure::Invalid("未定义的法术。")),
    }
    remove_card(s, &card);
    ctx.emit(s,json!({"type":"skill","owner":owner,"text":catalog[&k].name,"ultimate":catalog[&k].tier=="ultimate"}),Some(format!("施放{}",catalog[&k].name)));
    Ok(())
}
