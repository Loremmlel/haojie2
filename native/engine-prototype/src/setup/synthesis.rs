//! 配方由 TS 初始化注入。材料移除不是死亡，先在独立视图校验落点，再原子提交。
use crate::geometry::can_deploy;
use crate::model::{Catalog, Command, Failure, Point, State, ensure, number};
use crate::preparation::{hand, hand_mut, kind};
use crate::resolution::{Resolution, deploy_unit, template};
use serde_json::{Value, json};

pub fn materials(s: &State, recipe: &Value) -> Vec<String> {
    if recipe["id"] == "laoqian" && s.aura(s.active, "laoqian") {
        return vec![];
    }
    if recipe["source"] == "board" {
        s.units
            .iter()
            .filter(|u| u.kind == kind(&recipe["material"]) && u.side() == s.active)
            .map(|u| u.id.clone())
            .collect()
    } else {
        hand(s)
            .iter()
            .filter(|c| {
                c["kind"] == recipe["material"]
                    && c.get("expiresAt")
                        .is_none_or(|v| number(v) > s.turns[&s.active.to_string()])
            })
            .map(|c| c["id"].as_str().unwrap().into())
            .collect()
    }
}
pub fn available(s: &State, catalog: &Catalog) -> bool {
    catalog.recipes.iter().any(|r| materials(s, r).len() >= 3)
}
/// 与规则结算共用材料资格和部署几何；这里只返回材料移除后的落点，不修改输入。
pub fn destinations(s: &State, recipe: &Value, ids: &[String], catalog: &Catalog) -> Vec<Point> {
    let valid = materials(s, recipe);
    if ids.len() != 3
        || ids.iter().collect::<std::collections::HashSet<_>>().len() != 3
        || ids.iter().any(|id| !valid.contains(id))
    {
        return vec![];
    }
    let mut view = s.clone();
    if recipe["source"] == "board" {
        view.units.retain(|u| !ids.contains(&u.id));
    }
    let result = kind(&recipe["result"]).key();
    if catalog[&result].aura {
        return vec![];
    }
    let ghost = template(
        s,
        &result,
        s.active,
        Point { x: 0.0, y: 0.0 },
        "preview",
        catalog,
    );
    crate::geometry::all_cells()
        .filter(|p| can_deploy(&view, &ghost, *p, catalog))
        .collect()
}
pub fn synthesize(
    s: &mut State,
    c: &Command,
    catalog: &Catalog,
    ctx: &mut Resolution,
) -> Result<(), Failure> {
    ensure(
        s.phase == "synthesis" && s.pending.is_empty(),
        "合成仅限己方回合开始的合成窗口。",
    )?;
    let recipe = catalog
        .recipes
        .iter()
        .find(|r| r["id"].as_str() == c.recipe_id.as_deref())
        .ok_or(Failure::Invalid("请选择有效的合成配方。"))?;
    let ids = c.material_ids.as_deref().unwrap_or(&[]);
    let result = kind(&recipe["result"]).key();
    let aura = catalog[&result].aura;
    let point = if aura {
        Point { x: 0.0, y: 0.0 }
    } else {
        crate::shrines::point(c)?
    };
    let valid = materials(s, recipe);
    let mut view = s.clone();
    if recipe["source"] == "board" {
        view.units.retain(|u| !ids.contains(&u.id));
    }
    let ghost = template(s, &result, s.active, point, "preview", catalog);
    ensure(
        ids.len() == 3
            && ids.iter().collect::<std::collections::HashSet<_>>().len() == 3
            && ids.iter().all(|id| valid.contains(id))
            && (aura || can_deploy(&view, &ghost, point, catalog)),
        "需要3个不同的合法材料，以及移除材料后己方召唤区域内的合法落点。",
    )?;
    if recipe["source"] == "board" {
        s.units.retain(|u| !ids.contains(&u.id));
        s.siphons.retain(|l| {
            ["sourceId", "fromId", "toId"]
                .iter()
                .all(|k| !ids.iter().any(|id| l[k] == *id))
        });
        s.extra
            .get_mut("iceMarks")
            .unwrap()
            .as_array_mut()
            .unwrap()
            .retain(|m| !ids.iter().any(|id| m["sourceId"] == *id));
    } else {
        hand_mut(s).retain(|c| !ids.iter().any(|id| c["id"] == *id));
    }
    if result == "laoqian" {
        s.extra
            .entry("auras")
            .or_insert_with(|| json!({"1":[],"2":[]}))[s.active.to_string()]
        .as_array_mut()
        .unwrap()
        .push(json!({"kind":"laoqian"}));
        ctx.emit(
            s,
            json!({"type":"skill","owner":s.active,"text":"合成 · 牢千K"}),
            Some("3名改判小法师合成永久自选召唤光环牢千K".into()),
        );
    } else {
        let id = format!("u{}", s.serial);
        deploy_unit(s, &result, s.active, point, None, false, catalog, ctx);
        let u = s.unit(&id).unwrap();
        ctx.emit(s,json!({"type":"skill","action":"buff","to":u.actor_event(),"unitId":id,"owner":s.active,"text":format!("合成 · {}",catalog[&result].name)}),Some(format!("3个{}合成为{}",catalog[&kind(&recipe["material"]).key()].name,catalog[&result].name)));
    }
    if !available(s, catalog) {
        s.phase = "summon".into();
    }
    Ok(())
}
