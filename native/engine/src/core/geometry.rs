use crate::model::{Catalog, Point, State, Unit};
use rustc_hash::FxHashMap as HashMap;
use std::cell::OnceCell;

pub fn inside(p: Point) -> bool {
    p.x.fract() == 0.0
        && p.y.fract() == 0.0
        && (1.0..=9.0).contains(&p.x)
        && (1.0..=13.0).contains(&p.y)
}
pub fn distance(a: Point, b: Point) -> f64 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}
pub fn cells(u: &Unit, at: Point) -> Vec<Point> {
    cell_iter(u, at).collect()
}
/// 与 TS 数值逐格判定同序；只读遍历不需要创建拥有型坐标列表。
pub fn cell_iter(u: &Unit, at: Point) -> impl Iterator<Item = Point> + Clone + use<> {
    square_cells(u.size as usize, at)
}
fn square_cells(size: usize, at: Point) -> impl Iterator<Item = Point> + Clone {
    (0..size).flat_map(move |y| {
        (0..size).map(move |x| Point {
            x: at.x + x as f64,
            y: at.y + y as f64,
        })
    })
}
pub fn adjacent(a: &Unit, b: &Unit) -> bool {
    cell_iter(a, a.at()).any(|p| cell_iter(b, b.at()).any(|q| distance(p, q) == 1.0))
}
pub fn covers(u: &Unit, p: Point) -> bool {
    let dx = p.x - u.x;
    let dy = p.y - u.y;
    dx.fract() == 0.0 && dy.fract() == 0.0 && dx >= 0.0 && dy >= 0.0 && dx < u.size && dy < u.size
}
pub fn ring(u: &Unit, points: &[Point]) -> bool {
    let own = cells(u, u.at());
    own.iter().any(|a| {
        points
            .iter()
            .any(|b| (a.x - b.x).abs().max((a.y - b.y).abs()) == 1.0)
    }) && !own.iter().any(|a| points.contains(a))
}
pub fn square(u: &Unit, p: Point) -> bool {
    inside(p) && (p.x - u.x).abs() <= 5.0 && (p.y - u.y).abs() <= 5.0
}
pub fn point_target(p: Point) -> Target {
    Target {
        id: String::new(),
        owner: 0,
        at: p,
        unit: None,
    }
}
pub fn valid_attack_route(u: &Unit, path: &[Point], limit: f64) -> bool {
    if path.is_empty()
        || path.len() as f64 > (limit + 1.0).min(117.0)
        || path.iter().any(|p| !inside(*p))
        || !covers(u, path[0])
    {
        return false;
    }
    for i in 1..path.len() {
        if path[..i].contains(&path[i])
            || distance(path[i - 1], path[i]) != 1.0
            || covers(u, path[i])
            || (i < path.len() - 1 && path[i] == base_point(3 - u.owner))
        {
            return false;
        }
    }
    true
}
pub fn piercing_targets(
    s: &State,
    u: &Unit,
    path: &[Point],
    catalog: &Catalog,
) -> Vec<(Target, Vec<Point>)> {
    let available: Vec<_> = targets(s)
        .into_iter()
        .filter(|t| {
            t.id != u.id
                && t.unit.as_ref().map(Unit::side).unwrap_or(t.owner) != u.owner
                && (top_target(s, t, catalog)
                    || t.unit
                        .as_ref()
                        .is_some_and(|v| catalog.by_kind(&v.kind).landmark.is_some()))
        })
        .collect();
    let mut hits: Vec<(Target, Vec<Point>)> = vec![];
    for i in 1..path.len() {
        for t in &available {
            if !hits.iter().any(|(v, _)| v.id == t.id) && t.footprint().contains(&path[i]) {
                hits.push((t.clone(), path[..=i].to_vec()));
            }
        }
    }
    hits
}

pub struct Target {
    pub id: String,
    pub owner: usize,
    pub at: Point,
    pub unit: Option<Unit>,
}
impl Clone for Target {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            owner: self.owner,
            at: self.at,
            unit: self.unit.as_ref().map(Unit::fork),
        }
    }
}
impl Target {
    pub fn from(u: &Unit) -> Self {
        Self {
            id: u.id.clone(),
            owner: u.owner,
            at: u.at(),
            unit: Some(u.fork()),
        }
    }
    pub fn footprint(&self) -> Vec<Point> {
        self.footprint_iter().collect()
    }
    fn footprint_iter(&self) -> impl Iterator<Item = Point> + Clone + use<> {
        self.unit
            .as_ref()
            .map(|u| square_cells(u.size as usize, u.at()))
            .unwrap_or_else(|| square_cells(1, self.at))
    }
    pub fn actor(&self) -> serde_json::Value {
        let mut actor = serde_json::json!({"id":self.id,"owner":self.owner,"x":self.at.x,"y":self.at.y,"size":self.unit.as_ref().map(|u|u.size).unwrap_or(1.0)});
        if let Some(u) = &self.unit {
            actor["kind"] = serde_json::json!(u.kind);
        }
        actor
    }
}
pub fn targets(s: &State) -> Vec<Target> {
    s.pieces()
        .map(Target::from)
        .chain((1..=2).map(|owner| Target {
            id: format!("base-{owner}"),
            owner,
            at: base_point(owner),
            unit: None,
        }))
        .collect()
}
/// 当前目标持有只读 Rc 句柄；写入自动分离，规则需要冻结的死亡/反击快照仍由规则入口创建。
/// 保持普通棋子、地标、基地的查找顺序。
pub fn find_target(s: &State, id: &str) -> Option<Target> {
    if let Some(unit) = s.unit(id) {
        return Some(Target::from(unit));
    }
    let owner = match id {
        "base-1" => 1,
        "base-2" => 2,
        _ => return None,
    };
    Some(Target {
        id: id.into(),
        owner,
        at: base_point(owner),
        unit: None,
    })
}
pub fn base_point(owner: usize) -> Point {
    Point {
        x: 5.0,
        y: if owner == 1 { 1.0 } else { 13.0 },
    }
}
pub fn frontal(path: &[Point], owner: usize) -> bool {
    path.len() >= 2
        && if owner == 1 {
            path[path.len() - 1].y < path[path.len() - 2].y
        } else {
            path[path.len() - 1].y > path[path.len() - 2].y
        }
}
pub fn top_target(s: &State, t: &Target, catalog: &Catalog) -> bool {
    let Some(u) = &t.unit else {
        return true;
    };
    let over = s.units.iter().find(|v| covers(v, t.at));
    if catalog.by_kind(&u.kind).landmark.is_some() {
        over.is_none_or(|v| v.side() != t.owner)
    } else {
        over.is_some_and(|v| v.id == t.id)
    }
}
fn index(p: Point) -> usize {
    (p.y as usize - 1) * 9 + p.x as usize - 1
}

/// 完全沿用 TS 最短路径/方向候选顺序；正面优先只影响普通攻击的路径选择。
pub fn attack_path(
    s: &State,
    u: &Unit,
    t: &Target,
    limit: f64,
    wanted: Option<&str>,
    pierce: bool,
) -> Option<Vec<Point>> {
    attack_search(s, u, t, limit, pierce, wanted, false, None).0
}

/// 同一来源的纯落点共用一次广搜；敌方占位可作终点，敌方基地仍封锁。
pub struct PointAttackQuery {
    reached: [bool; 117],
}
impl PointAttackQuery {
    pub fn new(s: &State, u: &Unit, limit: f64) -> Self {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Paths);
        let mut blocked = [false; 117];
        let mut reached = [false; 117];
        let mut depth = [-1_i16; 117];
        let mut queue = [0_usize; 117];
        for v in s.pieces() {
            if v.id != u.id && v.side() != u.owner {
                for p in cell_iter(v, v.at()) {
                    if inside(p) {
                        blocked[index(p)] = true;
                    }
                }
            }
        }
        let enemy_base = index(base_point(3 - u.owner));
        blocked[enemy_base] = true;
        let mut tail = 0;
        for p in cell_iter(u, u.at()) {
            let cell = index(p);
            reached[cell] = true;
            depth[cell] = 0;
            queue[tail] = cell;
            tail += 1;
        }
        let mut head = 0;
        while head < tail {
            let cell = queue[head];
            head += 1;
            let n = depth[cell];
            if n as f64 >= limit {
                continue;
            }
            for dir in 0..4 {
                let Some(next) = neighbor_cell(cell, dir) else {
                    continue;
                };
                if next == enemy_base {
                    continue;
                }
                reached[next] = true;
                if blocked[next] || depth[next] != -1 {
                    continue;
                }
                depth[next] = n + 1;
                queue[tail] = next;
                tail += 1;
            }
        }
        Self { reached }
    }
    pub fn has(&self, to: Point) -> bool {
        inside(to) && self.reached[index(to)]
    }
}

fn base(p: Point) -> bool {
    p.x == 5.0 && (p.y == 1.0 || p.y == 13.0)
}
fn near(a: Point, b: Point) -> bool {
    (a.x - b.x).abs().max((a.y - b.y).abs()) <= 1.0
}
pub fn neighbors(p: Point) -> impl Iterator<Item = Point> {
    [
        Point {
            x: p.x,
            y: p.y + 1.0,
        },
        Point {
            x: p.x,
            y: p.y - 1.0,
        },
        Point {
            x: p.x + 1.0,
            y: p.y,
        },
        Point {
            x: p.x - 1.0,
            y: p.y,
        },
    ]
    .into_iter()
    .filter(|p| inside(*p))
}
pub fn all_cells() -> impl Iterator<Item = Point> {
    (1..=13).flat_map(|y| {
        (1..=9).map(move |x| Point {
            x: x as f64,
            y: y as f64,
        })
    })
}
/// 按 BFS 首次到达顺序保留每个入射方向，与 TS attackRoutes 的 Map 插入顺序一致。
#[cfg(test)]
pub fn attack_routes(s: &State, u: &Unit, t: &Target, limit: f64, pierce: bool) -> Vec<String> {
    attack_route_paths(s, u, t, limit, pierce)
        .into_iter()
        .map(|route| route.direction.into())
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttackRoute {
    pub direction: &'static str,
    pub path: Vec<Point>,
}
fn attack_route_paths(
    s: &State,
    u: &Unit,
    t: &Target,
    limit: f64,
    pierce: bool,
) -> Vec<AttackRoute> {
    attack_search(s, u, t, limit, pierce, None, true, None).1
}

#[cfg(test)]
mod attack_query_tests {
    use super::*;

    #[test]
    fn point_field_matches_paths_for_both_sides_sizes_blockers_and_fractional_range() {
        let catalog = crate::catalog::embedded().unwrap();
        for owner in [1, 2] {
            for size in [1, 2] {
                let mut state = crate::runtime::create(73127, false, &catalog).unwrap();
                let source = crate::resolution::template(
                    &state,
                    if size == 1 { "26" } else { "5" },
                    owner,
                    Point { x: 3.0, y: 4.0 },
                    "source",
                    &catalog,
                );
                state.units = vec![source.clone()];
                for (kind, side, at, id) in [
                    ("wall", 3 - owner, Point { x: 5.0, y: 5.0 }, "wall"),
                    ("5", 3 - owner, Point { x: 6.0, y: 6.0 }, "giant"),
                    ("u25", owner, Point { x: 5.0, y: 7.0 }, "clone-a"),
                    ("u25", owner, Point { x: 5.0, y: 7.0 }, "clone-b"),
                ] {
                    let unit = crate::resolution::template(&state, kind, side, at, id, &catalog);
                    state.units.push(unit);
                }
                state.landmarks = Some(vec![crate::resolution::template(
                    &state,
                    "s1",
                    3 - owner,
                    Point { x: 5.0, y: 6.0 },
                    "landmark",
                    &catalog,
                )]);
                let before = serde_json::to_value(&state).unwrap();
                for limit in [-1.0, 0.0, 0.5, 1.5, 3.0, 5.5, 20.0] {
                    let query = PointAttackQuery::new(&state, &source, limit);
                    for at in
                        all_cells().chain([Point { x: 0.0, y: 1.0 }, Point { x: 1.5, y: 2.0 }])
                    {
                        assert_eq!(
                            query.has(at),
                            attack_path(&state, &source, &point_target(at), limit, None, false)
                                .is_some(),
                            "{owner}:{size}:{limit}:{at:?}"
                        );
                    }
                }
                assert_eq!(serde_json::to_value(&state).unwrap(), before);
            }
        }
    }

    #[test]
    fn shared_routes_preserve_auto_front_overlap_piercing_and_owned_results() {
        let catalog = crate::catalog::embedded().unwrap();
        for owner in [1, 2] {
            for overlap in [false, true] {
                let mut state = crate::runtime::create(73127, false, &catalog).unwrap();
                let mut source = crate::resolution::template(
                    &state,
                    "26",
                    owner,
                    Point {
                        x: 4.0,
                        y: if owner == 1 { 4.0 } else { 8.0 },
                    },
                    "source",
                    &catalog,
                );
                let mut victim = crate::resolution::template(
                    &state,
                    "24",
                    3 - owner,
                    Point { x: 4.0, y: 6.0 },
                    "victim",
                    &catalog,
                );
                if overlap {
                    source.size = 2.0;
                    victim.x = source.x + 1.0;
                    victim.y = source.y;
                }
                state.units = vec![
                    source.clone(),
                    victim.clone(),
                    crate::resolution::template(
                        &state,
                        "wall",
                        3 - owner,
                        Point { x: 5.0, y: 7.0 },
                        "blocker",
                        &catalog,
                    ),
                ];
                for (kind, side, at, id) in [
                    ("u25", 3 - owner, victim.at(), "stacked"),
                    ("5", 3 - owner, Point { x: 6.0, y: 6.0 }, "giant"),
                    ("u25", owner, Point { x: 6.0, y: 6.0 }, "friend"),
                    ("grave", owner, Point { x: 3.0, y: 6.0 }, "neutral"),
                    ("26", 3 - owner, base_point(3 - owner), "base-overlay"),
                ] {
                    state.units.push(crate::resolution::template(
                        &state, kind, side, at, id, &catalog,
                    ));
                }
                state.landmarks = Some(vec![crate::resolution::template(
                    &state,
                    "s1",
                    3 - owner,
                    victim.at(),
                    "landmark",
                    &catalog,
                )]);
                let mut query = AttackQuery::default();
                for target in targets(&state) {
                    for limit in [0.0, 1.5, 4.0, 7.0, 20.0] {
                        for pierce in [true, false] {
                            let expected =
                                attack_route_paths(&state, &source, &target, limit, pierce);
                            let directions: Vec<_> =
                                expected.iter().map(|route| route.direction).collect();
                            assert_eq!(
                                query.directions(&state, &source, &target, limit, pierce),
                                directions
                            );
                            assert_eq!(
                                query.routes(&state, &source, &target, limit, pierce),
                                expected
                            );
                            if let Some(mut path) =
                                query.path(&state, &source, &target, limit, None, pierce)
                                && let Some(point) = path.first_mut()
                            {
                                point.x = -1.0;
                            }
                            assert_eq!(
                                query.path(&state, &source, &target, limit, None, pierce),
                                attack_path(&state, &source, &target, limit, None, pierce)
                            );
                            assert_eq!(
                                attack_routes(&state, &source, &target, limit, pierce),
                                expected
                                    .iter()
                                    .map(|route| route.direction.to_owned())
                                    .collect::<Vec<_>>()
                            );
                            for direction in
                                [None, Some("down"), Some("up"), Some("right"), Some("left")]
                            {
                                let expected =
                                    attack_path(&state, &source, &target, limit, direction, pierce);
                                assert_eq!(
                                    query.path(&state, &source, &target, limit, direction, pierce),
                                    expected
                                );
                                let mut projected = source.clone();
                                projected.mode = "attack".into();
                                projected.shots = 0.0;
                                assert_eq!(
                                    query.path(
                                        &state, &projected, &target, limit, direction, pierce
                                    ),
                                    expected
                                );
                            }
                            let mut changed = query.routes(&state, &source, &target, limit, pierce);
                            if let Some(route) = changed.first_mut() {
                                route.path[0].x = -1.0;
                                route.direction = if route.direction == "left" {
                                    "right"
                                } else {
                                    "left"
                                };
                            }
                            assert_eq!(
                                query.routes(&state, &source, &target, limit, pierce),
                                expected
                            );
                            let mut changed =
                                query.directions(&state, &source, &target, limit, pierce);
                            changed.reverse();
                            changed.push("up");
                            assert_eq!(
                                query.directions(&state, &source, &target, limit, pierce),
                                directions
                            );
                            assert_eq!(
                                query.routes(&state, &source, &target, limit, pierce),
                                expected
                            );
                        }
                    }
                }
                let mut moved = source.clone();
                moved.x = 1.0;
                moved.y = 2.0;
                let target = Target::from(&victim);
                assert_eq!(
                    query.path(&state, &moved, &target, 7.0, None, false),
                    attack_path(&state, &moved, &target, 7.0, None, false)
                );
                assert_eq!(
                    query.directions(&state, &moved, &target, 7.0, false),
                    attack_route_paths(&state, &moved, &target, 7.0, false)
                        .iter()
                        .map(|route| route.direction)
                        .collect::<Vec<_>>()
                );
                let mut projected = victim.clone();
                projected.x = 1.0;
                projected.y = 3.0;
                let projected = Target::from(&projected);
                assert_eq!(
                    query.directions(&state, &source, &projected, 7.0, false),
                    attack_route_paths(&state, &source, &projected, 7.0, false)
                        .iter()
                        .map(|route| route.direction)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn base_id_collision_keeps_the_original_whole_entity_exclusion() {
        let catalog = crate::catalog::embedded().unwrap();
        let mut state = crate::runtime::create(73127, false, &catalog).unwrap();
        let source = crate::resolution::template(
            &state,
            "26",
            1,
            Point { x: 5.0, y: 10.0 },
            "source",
            &catalog,
        );
        let blocker = crate::resolution::template(
            &state,
            "26",
            2,
            Point { x: 5.0, y: 11.0 },
            "base-2",
            &catalog,
        );
        state.units = vec![source.clone(), blocker.clone()];
        let base = targets(&state)
            .into_iter()
            .find(|target| target.id == "base-2" && target.unit.is_none())
            .unwrap();
        let mut query = AttackQuery::default();
        let expected = attack_path(&state, &source, &base, 4.0, None, false);
        assert!(expected.as_ref().unwrap().contains(&blocker.at()));
        assert_eq!(
            query.path(&state, &source, &base, 4.0, None, false),
            expected
        );
        assert_eq!(
            query.routes(&state, &source, &base, 4.0, false),
            attack_route_paths(&state, &source, &base, 4.0, false)
        );
        for target in [base, Target::from(&blocker)] {
            assert_eq!(
                query.directions(&state, &source, &target, 4.0, false),
                attack_route_paths(&state, &source, &target, 4.0, false)
                    .iter()
                    .map(|route| route.direction)
                    .collect::<Vec<_>>()
            );
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AttackTarget {
    Unit(crate::entities::EntityHandle),
    Base(usize),
}
#[derive(Default)]
struct AttackPaths {
    routes: Option<Vec<AttackRoute>>,
    auto: Option<Option<Vec<Point>>>,
}
type AttackKey = (crate::entities::EntityHandle, AttackTarget, u64, bool);
/// 对应 TS createAttackQuery；只服务同一不可变局面，允许仅修改操作预算的攻击者投影。
/// 方向共享完整路径，自动选路保留原正面优先与重叠规则；返回值拥有独立路径。
#[derive(Default)]
pub struct AttackQuery {
    paths: HashMap<AttackKey, AttackPaths>,
    blockers: HashMap<crate::entities::EntityHandle, std::cell::OnceCell<[bool; 117]>>,
}
impl AttackQuery {
    fn entry(
        &mut self,
        s: &State,
        u: &Unit,
        t: &Target,
        limit: f64,
        pierce: bool,
    ) -> Option<(&mut AttackPaths, &std::cell::OnceCell<[bool; 117]>)> {
        let source = s.entities.handle(s, &u.id)?;
        let original = s.entity(source)?;
        if original.owner != u.owner
            || original.x != u.x
            || original.y != u.y
            || original.size != u.size
        {
            return None;
        }
        let target = if let Some(unit) = &t.unit {
            let target = s.entities.handle(s, &t.id)?;
            let original = s.entity(target)?;
            if !std::ptr::eq::<crate::model::UnitData>(&**original, &**unit)
                || t.owner != unit.owner
            {
                return None;
            }
            AttackTarget::Unit(target)
        } else {
            if t.id != format!("base-{}", t.owner)
                || t.at != base_point(t.owner)
                || s.entities.handle(s, &t.id).is_some()
            {
                return None;
            }
            AttackTarget::Base(t.owner)
        };
        let range = if limit == 0.0 {
            0
        } else if limit.is_nan() {
            f64::NAN.to_bits()
        } else {
            limit.to_bits()
        };
        Some((
            self.paths
                .entry((source, target, range, pierce))
                .or_default(),
            self.blockers.entry(source).or_default(),
        ))
    }
    pub fn directions(
        &mut self,
        s: &State,
        u: &Unit,
        t: &Target,
        limit: f64,
        pierce: bool,
    ) -> Vec<&'static str> {
        let Some((paths, blockers)) = self.entry(s, u, t, limit, pierce) else {
            return attack_route_paths(s, u, t, limit, pierce)
                .into_iter()
                .map(|route| route.direction)
                .collect();
        };
        paths
            .routes
            .get_or_insert_with(|| {
                attack_search(s, u, t, limit, pierce, None, true, Some(blockers)).1
            })
            .iter()
            .map(|route| route.direction)
            .collect()
    }
    pub fn routes(
        &mut self,
        s: &State,
        u: &Unit,
        t: &Target,
        limit: f64,
        pierce: bool,
    ) -> Vec<AttackRoute> {
        let Some((paths, blockers)) = self.entry(s, u, t, limit, pierce) else {
            return attack_route_paths(s, u, t, limit, pierce);
        };
        paths
            .routes
            .get_or_insert_with(|| {
                attack_search(s, u, t, limit, pierce, None, true, Some(blockers)).1
            })
            .clone()
    }
    pub fn path(
        &mut self,
        s: &State,
        u: &Unit,
        t: &Target,
        limit: f64,
        wanted: Option<&str>,
        pierce: bool,
    ) -> Option<Vec<Point>> {
        let Some((paths, blockers)) = self.entry(s, u, t, limit, pierce) else {
            return attack_path(s, u, t, limit, wanted, pierce);
        };
        if let Some(direction) = wanted {
            return paths
                .routes
                .get_or_insert_with(|| {
                    attack_search(s, u, t, limit, pierce, None, true, Some(blockers)).1
                })
                .iter()
                .find(|route| route.direction == direction)
                .map(|route| route.path.clone());
        }
        paths
            .auto
            .get_or_insert_with(|| {
                attack_search(s, u, t, limit, pierce, None, false, Some(blockers)).0
            })
            .clone()
    }
}
/// 与 TS attackSearch 相同的数字前驱广搜；路径仅在确定结果时展开。
#[allow(clippy::too_many_arguments)]
fn attack_search(
    s: &State,
    u: &Unit,
    t: &Target,
    limit: f64,
    pierce: bool,
    wanted: Option<&str>,
    all_routes: bool,
    source: Option<&std::cell::OnceCell<[bool; 117]>>,
) -> (Option<Vec<Point>>, Vec<AttackRoute>) {
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Paths);
    let ends = t.footprint_iter();
    let starts = cell_iter(u, u.at());
    if ends.clone().all(|to| {
        starts
            .clone()
            .all(|from| distance(from, to) > limit.ceil().max(0.0))
    }) {
        return (None, vec![]);
    }
    let mut end = [false; 117];
    let mut blocked = [false; 117];
    for p in ends.clone() {
        if inside(p) {
            end[index(p)] = true;
        }
    }
    if let Some(source) = source {
        if !pierce {
            // 来源阻挡场不含目标例外；复制后仍按原顺序清目标格和封锁基地。
            blocked = *source.get_or_init(|| {
                let mut blocked = [false; 117];
                for v in s.pieces() {
                    if v.id != u.id && v.side() != u.owner {
                        for p in cell_iter(v, v.at()) {
                            blocked[index(p)] = true;
                        }
                    }
                }
                blocked
            });
        }
    } else {
        for v in s.pieces() {
            if !pierce && v.id != u.id && v.id != t.id && v.side() != u.owner {
                for p in cell_iter(v, v.at()) {
                    blocked[index(p)] = true;
                }
            }
        }
    }
    for p in ends {
        if inside(p) {
            blocked[index(p)] = false;
        }
    }
    if t.id != format!("base-{}", 3 - u.owner) {
        blocked[index(base_point(3 - u.owner))] = true;
    }
    let directional = all_routes || wanted.is_some();
    let mut parent = [-1_i16; 117];
    let mut depth = [0_u8; 117];
    let mut queue = [0_usize; 117];
    let mut tail = 0;
    for p in starts {
        let cell = index(p);
        if directional && end[cell] {
            continue;
        }
        parent[cell] = cell as i16;
        queue[tail] = cell;
        tail += 1;
    }
    let mut best = None;
    let mut routes = vec![];
    let mut directions = 0_u8;
    let mut fallback_depth = usize::MAX;
    let mut head = 0;
    while head < tail {
        let cell = queue[head];
        head += 1;
        let n = depth[cell] as usize;
        if !directional && end[cell] {
            return (Some(grid_path(&parent, cell, None)), routes);
        }
        if n as f64 >= limit || (!directional && n >= fallback_depth) {
            continue;
        }
        for dir in 0..4 {
            let Some(next) = neighbor_cell(cell, dir) else {
                continue;
            };
            if blocked[next] {
                continue;
            }
            if end[next] {
                if directional {
                    if directions & (1 << dir) == 0 {
                        directions |= 1 << dir;
                        let direction = ["down", "up", "right", "left"][dir];
                        if wanted == Some(direction) {
                            return (Some(grid_path(&parent, cell, Some(next))), routes);
                        }
                        routes.push(AttackRoute {
                            direction,
                            path: if all_routes {
                                grid_path(&parent, cell, Some(next))
                            } else {
                                vec![]
                            },
                        });
                        if directions == 15 {
                            return (best, routes);
                        }
                    }
                } else {
                    if best.is_none() {
                        best = Some(grid_path(&parent, cell, Some(next)));
                        fallback_depth = n + 1;
                    }
                    if t.unit.as_ref().is_none_or(|v| {
                        !v.has("24") || v.silenced || if t.owner == 1 { dir == 1 } else { dir == 0 }
                    }) {
                        return (Some(grid_path(&parent, cell, Some(next))), routes);
                    }
                }
            } else if parent[next] == -1 {
                parent[next] = cell as i16;
                depth[next] = depth[cell] + 1;
                queue[tail] = next;
                tail += 1;
            }
        }
    }
    (best, routes)
}
fn neighbor_cell(cell: usize, dir: usize) -> Option<usize> {
    match dir {
        0 => (cell < 108).then(|| cell + 9),
        1 => (cell >= 9).then(|| cell - 9),
        2 => (cell % 9 < 8).then(|| cell + 1),
        _ => (!cell.is_multiple_of(9)).then(|| cell - 1),
    }
}
fn grid_point(cell: usize) -> Point {
    Point {
        x: (cell % 9 + 1) as f64,
        y: (cell / 9 + 1) as f64,
    }
}
fn grid_path(parent: &[i16; 117], mut cell: usize, end: Option<usize>) -> Vec<Point> {
    let mut path = vec![];
    if let Some(end) = end {
        path.push(grid_point(end));
    }
    loop {
        path.push(grid_point(cell));
        if parent[cell] == cell as i16 {
            break;
        }
        cell = parent[cell] as usize;
    }
    path.reverse();
    path
}
pub struct MovementField {
    parent: [i16; 117],
    depth: [u8; 117],
    source: usize,
    straight: bool,
}
impl MovementField {
    pub fn new(u: &Unit, limit: f64, straight: bool, place: impl Fn(Point) -> bool) -> Self {
        let mut f = Self {
            parent: [-1; 117],
            depth: [0; 117],
            source: index(u.at()),
            straight,
        };
        let mut queue = [0; 117];
        queue[0] = f.source;
        f.parent[f.source] = f.source as i16;
        let mut tail = 1;
        let mut head = 0;
        while head < tail {
            let cell = queue[head];
            head += 1;
            if if straight {
                f.depth[cell] >= 3
            } else {
                f.depth[cell] as f64 >= limit
            } {
                continue;
            }
            for dir in 0..4 {
                let Some(next) = neighbor_cell(cell, dir) else {
                    continue;
                };
                let p = grid_point(next);
                if (straight && p.x != u.x && p.y != u.y) || f.parent[next] != -1 || !place(p) {
                    continue;
                }
                f.parent[next] = cell as i16;
                f.depth[next] = f.depth[cell] + 1;
                queue[tail] = next;
                tail += 1;
            }
        }
        f
    }
    pub fn has(&self, to: Point) -> bool {
        if !inside(to) {
            return false;
        }
        let cell = index(to);
        cell != self.source && self.parent[cell] != -1 && (!self.straight || self.depth[cell] == 3)
    }
    pub fn path(&self, to: Point) -> Option<Vec<Point>> {
        if !self.has(to) {
            return None;
        }
        Some(grid_path(&self.parent, index(to), None))
    }
}

pub fn expansion_anchors(s: &State, u: &Unit, catalog: &Catalog) -> Vec<Point> {
    if u.size != 1.0 || catalog.by_kind(&u.kind).landmark.is_some() {
        return vec![];
    }
    let mut giant = u.clone();
    giant.size = 2.0;
    [
        u.at(),
        Point {
            x: u.x - 1.0,
            y: u.y,
        },
        Point {
            x: u.x,
            y: u.y - 1.0,
        },
        Point {
            x: u.x - 1.0,
            y: u.y - 1.0,
        },
    ]
    .into_iter()
    .filter(|p| can_place(s, &giant, *p, catalog))
    .collect()
}
pub fn deployment_rows(s: &State, owner: usize) -> Vec<usize> {
    let mut counts = [0_i32; 14];
    for u in &s.units {
        if u.side() == 0 {
            continue;
        }
        for dy in 0..u.size as usize {
            let row = u.y as usize + dy;
            if row < 14 {
                counts[row] += if u.side() == owner { 1 } else { -1 };
            }
        }
    }
    (1..=13)
        .filter(|&y| (if owner == 1 { y <= 8 } else { y >= 6 }) || counts[y] >= 2)
        .collect()
}

/// 与 TS placement 共用部署/移动判定顺序；地标单独占层，克隆和禁区按原数组判定。
pub fn can_place(s: &State, u: &Unit, at: Point, catalog: &Catalog) -> bool {
    placement(s, u, at, catalog, false, None)
}
pub fn can_deploy(s: &State, u: &Unit, at: Point, catalog: &Catalog) -> bool {
    placement(s, u, at, catalog, true, None)
}
/// 与 TS createPlacementQuery 对齐；索引只属于单一不可变观察，不限制每格实体数量。
pub struct PlacementQuery {
    occupants: Vec<Vec<usize>>,
    friends: [Vec<usize>; 3],
    loners: [Vec<usize>; 3],
    rows: [OnceCell<Vec<usize>>; 3],
}
impl PlacementQuery {
    pub fn new(s: &State) -> Self {
        let mut q = Self {
            occupants: vec![vec![]; 117],
            friends: Default::default(),
            loners: Default::default(),
            rows: Default::default(),
        };
        for (i, u) in s.units.iter().enumerate() {
            for p in cells(u, u.at()) {
                if inside(p) {
                    q.occupants[index(p)].push(i);
                }
            }
            let side = u.side();
            if side == 1 || side == 2 {
                q.friends[side].push(i);
                if u.has("23") && !u.silenced {
                    q.loners[side].push(i);
                }
            }
        }
        q
    }
    pub fn can_place(&self, s: &State, u: &Unit, at: Point, catalog: &Catalog) -> bool {
        placement(s, u, at, catalog, false, Some(self))
    }
    pub fn can_deploy(&self, s: &State, u: &Unit, at: Point, catalog: &Catalog) -> bool {
        placement(s, u, at, catalog, true, Some(self))
    }
}
fn placement(
    s: &State,
    u: &Unit,
    at: Point,
    catalog: &Catalog,
    deployment: bool,
    query: Option<&PlacementQuery>,
) -> bool {
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Paths);
    let footprint = cells(u, at);
    if footprint.iter().any(|&p| !inside(p) || base(p)) {
        return false;
    }
    if let Some(rule) = &catalog.by_kind(&u.kind).landmark {
        if !deployment {
            return u.at() == at;
        }
        let square = rule
            .get("allowed")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter().any(|p| {
                    crate::model::number(&p["x"]) == at.x && crate::model::number(&p["y"]) == at.y
                })
            })
            .unwrap_or((6.0..=8.0).contains(&at.y));
        if !square || s.landmarks().iter().any(|l| l.at() == at) {
            return false;
        }
        let over: Vec<_> = s.units.iter().filter(|v| covers(v, at)).collect();
        return over.len() <= 1 && over.iter().all(|v| v.side() == u.owner);
    }
    let uncached_rows;
    let rows = if !deployment {
        &[][..]
    } else if let Some(q) = query {
        q.rows[u.owner].get_or_init(|| deployment_rows(s, u.owner))
    } else {
        uncached_rows = deployment_rows(s, u.owner);
        &uncached_rows
    };
    if deployment
        && !u.has("u27")
        && footprint.iter().any(|p| {
            !rows.contains(&(p.y as usize))
                && !s
                    .landmarks()
                    .iter()
                    .any(|l| l.at() == *p && l.live() && l.side() == u.owner)
        })
    {
        return false;
    }
    for &p in &footprint {
        if let Some(l) = s.landmarks().iter().find(|l| l.at() == p)
            && (u.size > 1.0
                || (deployment && l.live() && l.side() != u.owner)
                || s.units.iter().any(|v| v.id != u.id && covers(v, p)))
        {
            return false;
        }
    }
    let overlapping = |v: &Unit| {
        v.id != u.id
            && footprint.iter().any(|&p| covers(v, p))
            && !(u.kind.is("u25")
                && v.kind.is("u25")
                && u.owner == v.owner
                && u.size == 1.0
                && v.size == 1.0)
    };
    let occupied = if let Some(q) = query {
        footprint
            .iter()
            .flat_map(|p| &q.occupants[index(*p)])
            .any(|i| overlapping(&s.units[*i]))
    } else {
        s.units.iter().any(overlapping)
    };
    if occupied {
        return false;
    }
    let isolated = |v: &Unit| {
        v.id != u.id
            && v.side() == u.owner
            && ((v.has("23") && !v.silenced) || (u.has("23") && !u.silenced))
            && cells(v, v.at())
                .iter()
                .any(|&a| footprint.iter().any(|&b| near(a, b)))
    };
    !if let Some(q) = query {
        let indices = if u.has("23") && !u.silenced {
            &q.friends[u.owner]
        } else {
            &q.loners[u.owner]
        };
        indices.iter().any(|i| isolated(&s.units[*i]))
    } else {
        s.units.iter().any(isolated)
    }
}
pub fn empty_for(s: &State, u: &Unit, catalog: &Catalog) -> bool {
    can_place(s, u, u.at(), catalog)
        && cells(u, u.at()).iter().all(|&p| {
            !s.landmarks().iter().any(|l| l.at() == p)
                && s.units.iter().all(|v| v.id == u.id || !covers(v, p))
        })
}

/// 四向顺序、最短路径和分数上限保留 TS 语义；117格访问表只替换字符串 Set 的表示。
pub fn movement_path(
    s: &State,
    u: &Unit,
    to: Point,
    limit: f64,
    straight: bool,
    catalog: &Catalog,
) -> Option<Vec<Point>> {
    movement_with_place(u, to, limit, straight, |p| can_place(s, u, p, catalog))
}
pub fn movement_with_place(
    u: &Unit,
    to: Point,
    limit: f64,
    straight: bool,
    place: impl Fn(Point) -> bool,
) -> Option<Vec<Point>> {
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Paths);
    if !inside(to) || u.at() == to || !place(to) {
        return None;
    }
    if straight {
        if distance(u.at(), to) != 3.0 || (u.x != to.x && u.y != to.y) {
            return None;
        }
        let dx = if to.x == u.x {
            0.0
        } else {
            (to.x - u.x).signum()
        };
        let dy = if to.y == u.y {
            0.0
        } else {
            (to.y - u.y).signum()
        };
        let path: Vec<_> = (0..4)
            .map(|i| Point {
                x: u.x + dx * i as f64,
                y: u.y + dy * i as f64,
            })
            .collect();
        return path.iter().skip(1).all(|&p| place(p)).then_some(path);
    }
    if distance(u.at(), to) > limit.ceil().max(0.0) {
        return None;
    }
    let index = |p: Point| ((p.y as usize - 1) * 9) + (p.x as usize - 1);
    let mut seen = [false; 117];
    seen[index(u.at())] = true;
    let mut queue = vec![vec![u.at()]];
    let mut i = 0;
    while i < queue.len() {
        if (queue[i].len() - 1) as f64 >= limit {
            i += 1;
            continue;
        }
        for p in neighbors(*queue[i].last().unwrap()) {
            if seen[index(p)] || !place(p) {
                continue;
            }
            let mut next = queue[i].clone();
            next.push(p);
            if p == to {
                return Some(next);
            }
            seen[index(p)] = true;
            queue.push(next);
        }
        i += 1;
    }
    None
}
