use crate::{
    geometry::{self, Target},
    model::{Catalog, Point, State, Unit, extra_number},
    stats,
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

struct Cache {
    root: usize,
    members: HashSet<usize>,
    stats: HashMap<usize, stats::Stats>,
    windows: HashMap<(usize, bool), State>,
    fields: HashMap<String, [i16; 117]>,
}
thread_local! {static CACHE:RefCell<Option<Cache>>=const{RefCell::new(None)};}
pub fn with_cache<T>(s: &State, run: impl FnOnce() -> T) -> T {
    let root = std::ptr::from_ref(s) as usize;
    if CACHE.with(|c| c.borrow().as_ref().is_some_and(|c| c.root == root)) {
        return run();
    }
    struct Restore(Option<Cache>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CACHE.with(|c| {
                *c.borrow_mut() = self.0.take();
            });
        }
    }
    let cache = Cache {
        root,
        members: s
            .units
            .iter()
            .map(|u| std::ptr::from_ref(&**u) as usize)
            .collect(),
        stats: HashMap::new(),
        windows: HashMap::new(),
        fields: HashMap::new(),
    };
    let old = CACHE.with(|c| c.replace(Some(cache)));
    let _restore = Restore(old);
    run()
}
pub fn stats_for(s: &State, u: &Unit, catalog: &Catalog) -> stats::Stats {
    let root = std::ptr::from_ref(s) as usize;
    let id = std::ptr::from_ref(&**u) as usize;
    let cached = CACHE.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|c| c.root == root && c.members.contains(&id))
            .and_then(|c| c.stats.get(&id).cloned())
    });
    if let Some(st) = cached {
        return st;
    }
    let st = stats::stats(s, u, catalog);
    CACHE.with(|c| {
        if let Some(c) = c
            .borrow_mut()
            .as_mut()
            .filter(|c| c.root == root && c.members.contains(&id))
        {
            c.stats.insert(id, st.clone());
        }
    });
    st
}

fn index(p: Point) -> usize {
    ((p.y - 1.) * 9. + p.x - 1.) as usize
}
pub fn action_window(s: &State, side: usize, next: bool) -> State {
    let root = std::ptr::from_ref(s) as usize;
    if let Some(view) = CACHE.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|c| c.root == root)
            .and_then(|c| c.windows.get(&(side, next)).map(State::fork))
    }) {
        return view;
    }
    let view = make_window(s, side, next);
    CACHE.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut().filter(|c| c.root == root) {
            c.windows.insert((side, next), view.fork());
        }
    });
    view
}
fn make_window(s: &State, side: usize, next: bool) -> State {
    let delta = if s.active != side {
        1.
    } else if next {
        2.
    } else {
        0.
    };
    let mut view = s.fork();
    if delta == 0. && s.phase == "play" {
        return view;
    }
    view.ply += delta;
    view.active = side;
    view.phase = "play".into();
    view.pending.clear();
    view.turns
        .insert(side.to_string(), s.turn(side) + f64::from(delta != 0.));
    view.units
        .retain(|u| !u.extra.contains_key("expiresAt") || extra_number(u, "expiresAt") > view.ply);
    if delta != 0. {
        for u in &mut view.units {
            if u.owner == side {
                crate::resolution::reset_unit(view.ply, u);
            }
        }
    }
    view
}
pub fn attack_field(s: &State, u: &Unit, range: f64, ignore: &str) -> [i16; 117] {
    let root = std::ptr::from_ref(s) as usize;
    let key = format!(
        "{}:{}:{}:{}:{}:{range}:{ignore}:{}",
        u.id,
        u.owner,
        u.x,
        u.y,
        u.size,
        u.weapon("u28")
    );
    if let Some(field) = CACHE.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|c| c.root == root)
            .and_then(|c| c.fields.get(&key).copied())
    }) {
        return field;
    }
    let field = make_field(s, u, range, ignore);
    CACHE.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut().filter(|c| c.root == root) {
            c.fields.insert(key, field);
        }
    });
    field
}
fn make_field(s: &State, u: &Unit, range: f64, ignore: &str) -> [i16; 117] {
    let mut distances = [-1; 117];
    let mut queue = Vec::new();
    let mut blocked = [false; 117];
    blocked[index(geometry::base_point(3 - u.owner))] = true;
    if !u.weapon("u28") {
        for v in &s.units {
            if v.id != u.id && v.id != ignore && v.side() != u.owner {
                for p in geometry::cell_iter(v, v.at()).filter(|p| geometry::inside(*p)) {
                    blocked[index(p)] = true;
                }
            }
        }
    }
    for p in geometry::cell_iter(u, u.at()).filter(|p| geometry::inside(*p)) {
        let i = index(p);
        distances[i] = 0;
        queue.push(p);
    }
    let mut cursor = 0;
    while cursor < queue.len() {
        let from = queue[cursor];
        cursor += 1;
        let depth = distances[index(from)];
        if f64::from(depth) >= range {
            continue;
        }
        for p in geometry::neighbors(from) {
            let i = index(p);
            if distances[i] >= 0 {
                continue;
            }
            distances[i] = depth + 1;
            if !blocked[i] {
                queue.push(p);
            }
        }
    }
    distances
}
pub fn hit_distance(s: &State, u: &Unit, t: &Target, ignore: &str, c: &Catalog) -> f64 {
    let range = stats_for(s, u, c).range;
    let ends = t.footprint();
    if u.piercing() && !u.weapon("u28") {
        let mut shortest = f64::INFINITY;
        for a in geometry::cell_iter(u, u.at()) {
            for b in &ends {
                let d = geometry::distance(a, *b);
                if (a.x == b.x || a.y == b.y) && (d > 0. || t.id != u.id) {
                    shortest = shortest.min(d);
                }
            }
        }
        return if shortest <= range {
            shortest
        } else {
            f64::INFINITY
        };
    }
    let field = attack_field(s, u, range, ignore);
    ends.iter()
        .filter(|p| geometry::inside(**p))
        .map(|p| field[index(*p)])
        .filter(|d| *d >= 0)
        .map(f64::from)
        .fold(f64::INFINITY, f64::min)
}
pub fn front_hit(s: &State, u: &Unit, t: &Target, ignore: &str, c: &Catalog) -> bool {
    if !u.piercing() || u.weapon("u28") {
        let mut view = s.fork();
        if !ignore.is_empty() {
            view.units.retain(|v| v.id != ignore);
        }
        let routes = geometry::AttackQuery::default().routes(
            &view,
            u,
            t,
            stats_for(s, u, c).range,
            u.weapon("u28"),
        );
        return !routes.is_empty() && routes.iter().all(|r| geometry::frontal(&r.path, t.owner));
    }
    let d = hit_distance(s, u, t, ignore, c);
    if !d.is_finite() || d < 1. {
        return false;
    }
    let field = attack_field(s, u, stats_for(s, u, c).range, ignore);
    t.footprint().iter().any(|p| {
        let before = Point {
            x: p.x,
            y: p.y + if t.owner == 1 { 1. } else { -1. },
        };
        geometry::inside(before) && f64::from(field[index(before)]) == d - 1.
    })
}
