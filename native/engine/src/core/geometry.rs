use crate::model::{Catalog, Point, State, Unit};

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
    (0..u.size as usize)
        .flat_map(|y| {
            (0..u.size as usize).map(move |x| Point {
                x: at.x + x as f64,
                y: at.y + y as f64,
            })
        })
        .collect()
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
        self.unit
            .as_ref()
            .map(|u| cells(u, u.at()))
            .unwrap_or_else(|| vec![self.at])
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
    attack_search(s, u, t, limit, pierce, wanted, false).0
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
pub fn attack_routes(s: &State, u: &Unit, t: &Target, limit: f64, pierce: bool) -> Vec<String> {
    attack_search(s, u, t, limit, pierce, None, true).1
}
/// 与 TS attackSearch 相同的数字前驱广搜；路径仅在确定结果时展开。
fn attack_search(
    s: &State,
    u: &Unit,
    t: &Target,
    limit: f64,
    pierce: bool,
    wanted: Option<&str>,
    all_routes: bool,
) -> (Option<Vec<Point>>, Vec<String>) {
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Paths);
    let ends = t.footprint();
    let starts = cells(u, u.at());
    if ends.iter().all(|&to| {
        starts
            .iter()
            .all(|&from| distance(from, to) > limit.ceil().max(0.0))
    }) {
        return (None, vec![]);
    }
    let mut end = [false; 117];
    let mut blocked = [false; 117];
    for p in &ends {
        if inside(*p) {
            end[index(*p)] = true;
        }
    }
    for v in s.pieces() {
        if !pierce && v.id != u.id && v.id != t.id && v.side() != u.owner {
            for p in cells(v, v.at()) {
                blocked[index(p)] = true;
            }
        }
    }
    for p in &ends {
        if inside(*p) {
            blocked[index(*p)] = false;
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
                        routes.push(direction.to_string());
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
    pub fn path(&self, to: Point) -> Option<Vec<Point>> {
        if !inside(to) {
            return None;
        }
        let cell = index(to);
        if cell == self.source
            || self.parent[cell] == -1
            || (self.straight && self.depth[cell] != 3)
        {
            return None;
        }
        Some(grid_path(&self.parent, cell, None))
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
}
impl PlacementQuery {
    pub fn new(s: &State) -> Self {
        let mut q = Self {
            occupants: vec![vec![]; 117],
            friends: Default::default(),
            loners: Default::default(),
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
    let rows = if deployment {
        deployment_rows(s, u.owner)
    } else {
        vec![]
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
