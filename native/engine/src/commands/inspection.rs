//! 对应 TS inspectionQueries。每个实例只服务一个不可变局面，缓存成功与规则拒绝；不保存正式 RNG。
use crate::model::{Catalog, Command, Failure, Kind, Point, State, Unit};
use crate::{abilities, combat, geometry, lifecycle, movement, preparation, reactions};
use std::cell::OnceCell;
use std::collections::HashMap;

type SkillKey = (Option<String>, Option<Kind>);
type ChargeKey = (Option<String>, Option<Kind>, Option<String>);

#[derive(Default)]
pub struct Queries {
    transit: OnceCell<Option<String>>,
    actors: HashMap<Option<String>, Result<Unit, Failure>>,
    moves: HashMap<Option<String>, Result<movement::MoveSource, Failure>>,
    attackers: HashMap<Option<String>, Result<Unit, Failure>>,
    charges: HashMap<ChargeKey, Result<(), Failure>>,
    skills: HashMap<SkillKey, Result<abilities::SkillInspection, Failure>>,
    deployments: HashMap<Option<String>, Result<Unit, Failure>>,
    placement: OnceCell<geometry::PlacementQuery>,
    hut: OnceCell<Vec<Point>>,
}
impl Queries {
    /// 与 TS points 共用来源和精确落点判定；false 表示冲撞仍须完整预检。
    pub fn points(
        &mut self,
        s: &State,
        c: &Command,
        catalog: &Catalog,
    ) -> Result<Option<Vec<(Point, bool)>>, Failure> {
        if c.kind != "move" && c.kind != "deploy" {
            return Ok(None);
        }
        movement::stage_with_transit(s, c, || {
            self.transit
                .get_or_init(|| movement::transit(s, catalog).map(str::to_owned))
                .clone()
        })?;
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
            match movement::move_destination(s, source, at, catalog, Some(query)) {
                Ok(path) => points.push((at, path.is_some())),
                Err(Failure::Invalid(_) | Failure::InvalidOwned(_)) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(Some(points))
    }
    pub fn inspect(&mut self, s: &State, c: &Command, catalog: &Catalog) -> Result<(), Failure> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Preparation);
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
                self.skills
                    .entry((c.unit_id.clone(), c.ability.clone()))
                    .or_insert_with(|| abilities::prepare_inspection(s, c, catalog))
                    .as_mut()
                    .map_err(|e| e.clone())?
                    .inspect(c, catalog)?;
            }
            "cast" => {
                if crate::spells::prepare_inspection(s, c, catalog)? {
                    return Ok(());
                }
            }
            "attack" => {
                let u = self
                    .attackers
                    .entry(c.unit_id.clone())
                    .or_insert_with(|| combat::projected_attacker(s, c, catalog))
                    .as_ref()
                    .map_err(Clone::clone)?;
                let prepared = combat::prepare_inspection(s, c, u, catalog)?;
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
            }
            "extra-summon" => {
                preparation::prepare_extra(s, c)?;
            }
            "end" => {
                lifecycle::prepare_end(s, catalog)?;
            }
            "clock" => {
                crate::shrines::prepare_clock(s, c, catalog)?;
            }
            "begin" | "skip-synthesis" | "finish-shrine-setup" => {
                preparation::prepare_phase(s, &c.kind)?
            }
            _ => {}
        }
        // 攻击、技能、反应、冲撞和随机边界仍由相同完整规则结算，不能拿公共资格代替后果。
        crate::command::execute(s, c, catalog, true).map(|_| ())
    }
}
