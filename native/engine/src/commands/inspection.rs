//! 对应 TS inspectionQueries。每个实例只服务一个不可变局面，缓存成功与规则拒绝；不保存正式 RNG。
use crate::model::{Catalog, Command, Failure, Kind, Point, State, Unit};
use crate::stats::Stats;
use crate::{abilities, combat, geometry, lifecycle, movement, preparation, reactions};
use rustc_hash::FxHashMap as HashMap;
use std::cell::OnceCell;
use std::rc::Rc;

type Id = Option<Rc<str>>;
type SkillKey = (Id, Option<Kind>);
type ChargeKey = (Id, Option<Kind>, Option<String>);

#[derive(Default)]
pub struct Queries {
    pub attacks: geometry::AttackQuery,
    // 只读原局面的实体地址作键；临时能力/模式投影不能写入这份属性缓存。
    attributes: HashMap<*const Unit, Rc<Stats>>,
    transit: OnceCell<Option<String>>,
    actors: HashMap<Id, Result<Unit, Failure>>,
    moves: HashMap<Id, Result<movement::MoveSource, Failure>>,
    attackers: HashMap<Id, Result<(Unit, Rc<Stats>), Failure>>,
    charges: HashMap<ChargeKey, Result<(), Failure>>,
    skills: HashMap<SkillKey, Result<abilities::SkillInspection, Failure>>,
    deployments: HashMap<Id, Result<Unit, Failure>>,
    placement: OnceCell<geometry::PlacementQuery>,
    hut: OnceCell<Vec<Point>>,
    deterministic_attacks: OnceCell<bool>,
}

impl Queries {
    pub fn attributes(&mut self, s: &State, u: &Unit, catalog: &Catalog) -> Rc<Stats> {
        Rc::clone(
            self.attributes
                .entry(u as *const Unit)
                .or_insert_with(|| Rc::new(crate::stats::stats(s, u, catalog))),
        )
    }
    /// 与 TS points 共用来源和精确落点判定；false 表示冲撞仍须完整预检。
    pub fn points(
        &mut self,
        s: &State,
        c: &Command,
        catalog: &Catalog,
    ) -> Result<Option<Vec<(Point, bool)>>, Failure> {
        if c.kind != "move" && c.kind != "deploy" && c.kind != "skill" && c.kind != "cast" {
            return Ok(None);
        }
        movement::stage_with_transit(s, c, || {
            self.transit
                .get_or_init(|| movement::transit(s, catalog).map(str::to_owned))
                .clone()
        })?;
        if c.kind == "cast" {
            return crate::spells::points(s, c, catalog);
        }
        if c.kind == "skill" {
            let kind = match &c.ability {
                Some(kind) => kind,
                None => {
                    &s.unit(c.unit_id.as_deref().unwrap_or(""))
                        .ok_or(Failure::Invalid("请选择仍在场上的随从。"))?
                        .kind
                }
            };
            if !kind.is("u19") && !kind.is("19") && !kind.is("u24") {
                return Ok(None);
            }
            return self
                .skills
                .entry((c.unit_id.clone(), c.ability.clone()))
                .or_insert_with(|| abilities::prepare_inspection(s, c, catalog))
                .as_mut()
                .map_err(|error| error.clone())?
                .points(c, catalog);
        }
        let query = self
            .placement
            .get_or_init(|| geometry::PlacementQuery::new(s));
        if c.kind == "deploy" {
            let ghost = self
                .deployments
                .entry(c.card_id.clone())
                .or_insert_with(|| {
                    let card = preparation::deploy_card(s, c, catalog)?;
                    Ok(crate::resolution::template(
                        s,
                        &preparation::kind(&card["kind"]).key(),
                        s.active,
                        Point { x: 1.0, y: 1.0 },
                        "preview",
                        catalog,
                    ))
                })
                .as_ref()
                .map_err(Clone::clone)?;
            return Ok(Some(
                geometry::all_cells()
                    .filter(|&at| query.can_deploy(s, ghost, at, catalog))
                    .map(|at| (at, true))
                    .collect(),
            ));
        }
        let u = self
            .actors
            .entry(c.unit_id.clone())
            .or_insert_with(|| movement::actor(s, c, catalog).map(Unit::fork))
            .as_ref()
            .map_err(Clone::clone)?;
        let source = self
            .moves
            .entry(c.unit_id.clone())
            .or_insert_with(|| movement::move_source(s, u.fork(), catalog))
            .as_ref()
            .map_err(Clone::clone)?;
        let mut points = vec![];
        for at in geometry::all_cells() {
            match movement::move_contains(s, source, at, catalog, query) {
                Ok(complete) => points.push((at, complete)),
                Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(Some(points))
    }
    pub fn inspect(&mut self, s: &State, c: &Command, catalog: &Catalog) -> Result<(), Failure> {
        movement::stage_with_transit(s, c, || {
            self.transit
                .get_or_init(|| movement::transit(s, catalog).map(str::to_owned))
                .clone()
        })?;
        match c.kind.as_str() {
            "deploy" => {
                preparation::prepare_deploy(s, c, catalog)?;
                return Ok(());
            }
            "equip" => {
                preparation::prepare_equip(s, c, catalog)?;
                return Ok(());
            }
            "finish-mode" => return movement::prepare_finish(s, c, catalog),
            "charge" => {
                return self
                    .charges
                    .entry((c.unit_id.clone(), c.ability.clone(), c.mode.clone()))
                    .or_insert_with(|| preparation::prepare_charge(s, c, catalog).map(|_| ()))
                    .clone();
            }
            "move" => {
                let u = self
                    .actors
                    .entry(c.unit_id.clone())
                    .or_insert_with(|| movement::actor(s, c, catalog).map(Unit::fork))
                    .as_ref()
                    .map_err(Clone::clone)?;
                let to = crate::shrines::point(c)?;
                let query = self
                    .placement
                    .get_or_init(|| geometry::PlacementQuery::new(s));
                let source = self
                    .moves
                    .entry(c.unit_id.clone())
                    .or_insert_with(|| movement::move_source(s, u.fork(), catalog))
                    .as_ref()
                    .map_err(Clone::clone)?;
                if movement::move_destination(s, source, to, catalog, Some(query))?.is_some() {
                    return Ok(());
                }
            }
            "skill" => {
                abilities::skill_coordinate(s, c)?;
                if self
                    .skills
                    .entry((c.unit_id.clone(), c.ability.clone()))
                    .or_insert_with(|| abilities::prepare_inspection(s, c, catalog))
                    .as_mut()
                    .map_err(|e| e.clone())?
                    .inspect(c, catalog)?
                {
                    return Ok(());
                }
            }
            "cast" => {
                if crate::spells::prepare_inspection(s, c, catalog)? {
                    return Ok(());
                }
            }
            "attack" => {
                let raw_attributes = &mut self.attributes;
                let (u, attributes) =
                    self.attackers
                        .entry(c.unit_id.clone())
                        .or_insert_with(|| {
                            combat::projected_attacker(s, c, catalog, |raw| {
                                Rc::clone(raw_attributes.entry(raw as *const Unit).or_insert_with(
                                    || Rc::new(crate::stats::stats(s, raw, catalog)),
                                ))
                            })
                        })
                        .as_ref()
                        .map_err(Clone::clone)?;
                // 同批预检复用模式选择原时点的属性；正式结算和连锁攻击仍按自身状态计算。
                let Some(prepared) = combat::prepare_inspection(
                    s,
                    c,
                    u,
                    catalog,
                    &mut self.attacks,
                    attributes,
                    || {
                        *self
                            .deterministic_attacks
                            .get_or_init(|| combat::deterministic_attack_position(s))
                    },
                )?
                else {
                    return Ok(());
                };
                return combat::apply_prepared(s, c, catalog, true, Some(prepared)).map(|_| ());
            }
            "react" => {
                if let Some(r) = s.pending.first().filter(|r| r.kind == "hut-spawn") {
                    let points = self
                        .hut
                        .get_or_init(|| reactions::hut_points(s, r, catalog));
                    reactions::prepare_hut(s, r, c, points)?;
                }
            }
            "summon" => {
                preparation::prepare_summon(s, c)?;
                // 对应 TS：未自选时，共同资格后必先进入首次随机边界。
                if c.chosen_kind.is_none() {
                    return Err(Failure::Uncertain);
                }
            }
            "extra-summon" => {
                preparation::prepare_extra(s, c)?;
                if c.chosen_kind.is_none() {
                    return Err(Failure::Uncertain);
                }
            }
            "reroll" => {
                preparation::prepare_reroll(s, c, catalog)?;
                if c.chosen_kind.is_none() {
                    return Err(Failure::Uncertain);
                }
            }
            "end" => {
                lifecycle::prepare_end(s, catalog)?;
            }
            "clock" => {
                crate::shrines::prepare_clock(s, c, catalog)?;
            }
            "begin" | "skip-synthesis" => {
                preparation::prepare_phase(s, &c.kind)?;
                return Ok(());
            }
            "finish-shrine-setup" => preparation::prepare_phase(s, &c.kind)?,
            _ => {}
        }
        // 攻击、技能、反应、冲撞和随机边界仍由相同完整规则结算，不能拿公共资格代替后果。
        crate::command::execute(s, c, catalog, true).map(|_| ())
    }
}

#[cfg(test)]
mod raw_attribute_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn raw_attributes_share_root_and_attack_without_crossing_state_or_projection() {
        let catalog = crate::catalog::embedded().unwrap();
        let mut state = crate::runtime::create(731280049, false, &catalog).unwrap();
        state.phase = "play".into();
        state.summon_slots = 0.0;
        state.ply = 5.0;
        state.turns.insert("1".into(), 3.0);
        state.turns.insert("2".into(), 2.0);
        let mut source = crate::resolution::template(
            &state,
            "26",
            1,
            Point { x: 4.0, y: 4.0 },
            "source",
            &catalog,
        );
        source.kills = 1.0;
        source.extra.insert("extraOperations".into(), json!(1));
        let mut victim = crate::resolution::template(
            &state,
            "1",
            2,
            Point { x: 4.0, y: 5.0 },
            "victim",
            &catalog,
        );
        victim.hp = 1.0;
        let runner = crate::resolution::template(
            &state,
            "20",
            2,
            Point { x: 5.0, y: 4.0 },
            "runner",
            &catalog,
        );
        let healer = crate::resolution::template(
            &state,
            "2",
            1,
            Point { x: 3.0, y: 4.0 },
            "healer",
            &catalog,
        );
        state.units = vec![source, victim, runner, healer];
        for u in &mut state.units {
            u.born = 0.0;
        }
        let attack = crate::stats::stats(&state, state.unit("source").unwrap(), &catalog).attack;
        state.unit_mut("runner").unwrap().hp = attack + 1.0;
        let before = serde_json::to_value(&state).unwrap();
        let mut queries = Queries::default();
        let source = state.unit("source").unwrap();
        let original = queries.attributes(&state, source, &catalog);
        let mut kill = Command::new("attack");
        kill.unit_id = Some(Rc::from("source"));
        kill.target_id = Some(Rc::from("victim"));
        let mut probe = kill.clone();
        probe.target_id = Some(Rc::from("runner"));
        let mut invalid = probe.clone();
        invalid.mode = Some("invalid".into());
        for command in [&probe, &kill, &invalid, &probe] {
            let expected = crate::command::execute(&state, command, &catalog, true).map(|_| ());
            assert_eq!(
                format!("{:?}", queries.inspect(&state, command, &catalog)),
                format!("{expected:?}"),
            );
        }
        let projected = queries
            .attackers
            .get(&kill.unit_id)
            .unwrap()
            .as_ref()
            .unwrap();
        assert!(Rc::ptr_eq(&original, &projected.1));
        assert_eq!(queries.attributes.len(), 1);
        assert!(
            queries
                .attributes
                .keys()
                .all(|key| state.pieces().any(|u| std::ptr::eq(*key, u)))
        );

        let next = crate::command::apply_runtime(&state, &kill, &catalog).unwrap();
        let mut next_queries = Queries::default();
        let changed = next_queries.attributes(&next, next.unit("source").unwrap(), &catalog);
        assert_eq!(changed.attack, original.attack + 5.0);
        assert!(!Rc::ptr_eq(&original, &changed));
        assert!(matches!(
            next_queries.inspect(&next, &probe, &catalog),
            Err(Failure::Uncertain)
        ));
        assert!(queries.inspect(&state, &probe, &catalog).is_ok());
        assert_eq!(serde_json::to_value(&state).unwrap(), before);
    }
}
