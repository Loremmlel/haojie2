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
                        .is_some_and(|v| catalog[&v.kind.key()].landmark.is_some()))
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

#[derive(Clone)]
pub struct Target {
    pub id: String,
    pub owner: usize,
    pub at: Point,
    pub unit: Option<Unit>,
}
impl Target {
    pub fn from(u: &Unit) -> Self {
        Self {
            id: u.id.clone(),
            owner: u.owner,
            at: u.at(),
            unit: Some(u.clone()),
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
/// 单目标查询先借用状态定位，再只复制命中对象；保持普通棋子、地标、基地的查找顺序。
/// 不能为找一个 ID 构建全盘深拷贝，攻击和每份反伤都会经过此路径。
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
    if catalog[&u.kind.key()].landmark.is_some() {
        over.is_none_or(|v| v.side() != t.owner)
    } else {
        over.is_some_and(|v| v.id == t.id)
    }
}
fn index(p: Point) -> usize {
    (p.y as usize - 1) * 9 + p.x as usize - 1
}
pub fn direction(path: &[Point]) -> &'static str {
    let a = path[path.len() - 2];
    let b = path[path.len() - 1];
    if b.y < a.y {
        "up"
    } else if b.y > a.y {
        "down"
    } else if b.x < a.x {
        "left"
    } else {
        "right"
    }
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
    let ends = t.footprint();
    let starts = cells(u, u.at());
    if ends.iter().all(|&to| {
        starts
            .iter()
            .all(|&from| distance(from, to) > limit.ceil().max(0.0))
    }) {
        return None;
    }
    let mut blocked = [false; 117];
    for v in s.pieces() {
        if !pierce && v.id != u.id && v.id != t.id && v.side() != u.owner {
            for p in cells(v, v.at()) {
                blocked[index(p)] = true;
            }
        }
    }
    for &p in &ends {
        blocked[index(p)] = false;
    }
    if t.id != format!("base-{}", 3 - u.owner) {
        blocked[index(base_point(3 - u.owner))] = true;
    }
    let mut queue: Vec<Vec<Point>> = starts
        .into_iter()
        .filter(|p| wanted.is_none() || !ends.contains(p))
        .map(|p| vec![p])
        .collect();
    let mut seen = [false; 117];
    for path in &queue {
        seen[index(path[0])] = true;
    }
    let mut fallback: Option<Vec<Point>> = None;
    let mut i = 0;
    while i < queue.len() {
        let path = queue[i].clone();
        i += 1;
        let depth = path.len() - 1;
        if wanted.is_none() && ends.contains(&path[depth]) {
            return Some(path);
        }
        if depth as f64 >= limit
            || (wanted.is_none() && fallback.as_ref().is_some_and(|p| depth >= p.len() - 1))
        {
            continue;
        }
        for p in neighbors(path[depth]) {
            if blocked[index(p)] {
                continue;
            }
            let mut next = path.clone();
            next.push(p);
            if ends.contains(&p) {
                if let Some(wanted) = wanted {
                    if direction(&next) == wanted {
                        return Some(next);
                    }
                } else {
                    if fallback.is_none() {
                        fallback = Some(next.clone());
                    }
                    if t.unit
                        .as_ref()
                        .is_none_or(|v| !v.has("24") || v.silenced || frontal(&next, t.owner))
                    {
                        return Some(next);
                    }
                }
                continue;
            }
            if !seen[index(p)] {
                seen[index(p)] = true;
                queue.push(next);
            }
        }
    }
    if wanted.is_none() { fallback } else { None }
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
    placement(s, u, at, catalog, false)
}
pub fn can_deploy(s: &State, u: &Unit, at: Point, catalog: &Catalog) -> bool {
    placement(s, u, at, catalog, true)
}
fn placement(s: &State, u: &Unit, at: Point, catalog: &Catalog, deployment: bool) -> bool {
    let footprint = cells(u, at);
    if footprint.iter().any(|&p| !inside(p) || base(p)) {
        return false;
    }
    if let Some(rule) = &catalog[&u.kind.key()].landmark {
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
    for v in &s.units {
        if v.id == u.id {
            continue;
        }
        if footprint.iter().any(|&p| covers(v, p))
            && !(u.kind.is("u25")
                && v.kind.is("u25")
                && u.owner == v.owner
                && u.size == 1.0
                && v.size == 1.0)
        {
            return false;
        }
        if v.side() == u.owner
            && ((v.has("23") && !v.silenced) || (u.has("23") && !u.silenced))
            && cells(v, v.at())
                .iter()
                .any(|&a| footprint.iter().any(|&b| near(a, b)))
        {
            return false;
        }
    }
    true
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
    if !inside(to) || u.at() == to || !can_place(s, u, to, catalog) {
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
        return path
            .iter()
            .skip(1)
            .all(|&p| can_place(s, u, p, catalog))
            .then_some(path);
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
            if seen[index(p)] || !can_place(s, u, p, catalog) {
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
