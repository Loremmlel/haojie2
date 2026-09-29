//! 同 TS 惰性参数树；节点只缓存当前公开局面，不做评分或候选裁剪。
use crate::actions::{self, Action, Step, array, extend, text};
use crate::geometry;
use crate::model::{Catalog, Point, State, Unit};
use serde_json::{Value, json};
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone)]
pub struct Prefix {
    action: usize,
    pub command: Value,
    steps: Vec<Step>,
}
#[derive(Clone)]
pub struct Choice {
    pub key: String,
    pub command: Value,
    pub status: &'static str,
    pub subject: Option<String>,
    pub next: Option<Prefix>,
}
#[derive(Clone)]
pub struct Node {
    pub cursor: Vec<usize>,
    pub prefix: Option<Value>,
    pub stage: &'static str,
    pub choices: Vec<Choice>,
}
impl Node {
    pub fn wire(&self) -> Value {
        let mut value = json!({"cursor":self.cursor,"stage":self.stage,"choices":self.choices.iter().map(|c|{
            let mut v=json!({"key":c.key,"command":c.command,"status":c.status,"next":c.next.is_some()});
            if let Some(s)=&c.subject {v["subject"]=json!(s);}v
        }).collect::<Vec<_>>()});
        if let Some(c) = &self.prefix {
            value["prefix"] = c.clone();
        }
        value
    }
}
pub struct Tree<'a> {
    pub observation: &'a Value,
    pub state: State,
    pub actor: usize,
    pub catalog: &'a Catalog,
    actions: Vec<Action>,
    nodes: HashMap<Vec<usize>, Rc<Node>>,
    queries: RefCell<crate::inspection::Queries>,
    pieces: HashMap<String, Unit>,
    targets: Vec<geometry::Target>,
    target_index: HashMap<String, usize>,
    stats: RefCell<HashMap<String, crate::stats::Stats>>,
    routes: RefCell<HashMap<(String, String), Vec<String>>>,
    pub encoding: OnceCell<Result<crate::encoding::BaseEncoding, String>>,
}
impl<'a> Tree<'a> {
    pub fn new(observation: &'a Value, actor: usize, catalog: &'a Catalog) -> Result<Self, String> {
        if ![1, 2].contains(&actor) {
            return Err("invalid training actor".into());
        }
        let state = actions::position(observation)?;
        let mut queries = crate::inspection::Queries::default();
        let actions = actions::actions(&state, actor, catalog, &mut queries)?;
        let mut pieces = HashMap::new();
        for u in state.pieces() {
            pieces.entry(u.id.clone()).or_insert_with(|| u.fork());
        }
        let targets = geometry::targets(&state);
        let mut target_index = HashMap::new();
        for (i, t) in targets.iter().enumerate() {
            target_index.entry(t.id.clone()).or_insert(i);
        }
        Ok(Self {
            observation,
            state,
            actor,
            catalog,
            actions,
            nodes: HashMap::new(),
            queries: RefCell::new(queries),
            pieces,
            targets,
            target_index,
            stats: RefCell::new(HashMap::new()),
            routes: RefCell::new(HashMap::new()),
            encoding: OnceCell::new(),
        })
    }
    fn routes(&self, c: &Value) -> Vec<String> {
        let key = (
            text(&c["unitId"]).to_string(),
            text(&c["targetId"]).to_string(),
        );
        if let Some(routes) = self.routes.borrow().get(&key) {
            return routes.clone();
        }
        let s = &self.state;
        let Some(u) = self.pieces.get(&key.0) else {
            return vec![];
        };
        let Some(t) = self.target_index.get(&key.1).map(|i| &self.targets[*i]) else {
            return vec![];
        };
        // TS selectableAttackRoutes 有结果时仍来自同一 attackRoutes，否则回退到全方向查询。
        let routes = geometry::attack_routes(s, u, t, self.range(u), u.piercing());
        self.routes.borrow_mut().insert(key, routes.clone());
        routes
    }
    fn range(&self, u: &Unit) -> f64 {
        self.stats
            .borrow_mut()
            .entry(u.id.clone())
            .or_insert_with(|| crate::stats::stats(&self.state, u, self.catalog))
            .range
    }
    fn prepare(&self, mut p: Prefix) -> Prefix {
        while p.steps.first().is_some_and(|s| s.kind == "direction")
            && self.routes(&p.command).is_empty()
        {
            p.steps.remove(0);
        }
        p
    }
    fn choice(
        &self,
        key: String,
        p: Prefix,
        subject: Option<String>,
    ) -> Result<Option<Choice>, String> {
        let p = self.prepare(p);
        let status = if p.steps.is_empty() {
            actions::inspect(
                &self.state,
                self.actor,
                &p.command,
                self.catalog,
                &mut self.queries.borrow_mut(),
            )?
        } else {
            "parameter"
        };
        if status == "invalid" {
            return Ok(None);
        }
        Ok(Some(Choice {
            key,
            command: p.command.clone(),
            status,
            subject,
            next: if p.steps.is_empty() { None } else { Some(p) },
        }))
    }
    fn start(&self, i: usize) -> Prefix {
        let a = &self.actions[i];
        let mut steps = vec![];
        if !a.materials.is_empty() {
            steps.extend(vec![Step::new("material"); 3]);
        }
        steps.extend(a.steps.clone());
        if !a.chosen.is_empty() {
            steps.push(Step::new("chosen"));
        }
        if a.command["type"] == "attack" && a.command.get("path").is_none() {
            steps.push(Step::new("direction"));
        }
        Prefix {
            action: i,
            command: a.command.clone(),
            steps,
        }
    }
    fn expand(&self, p: Prefix) -> Result<Vec<Choice>, String> {
        let c = &p.command;
        let a = &self.actions[p.action];
        let step = &p.steps[0];
        let s = &self.state;
        let mut result = vec![];
        let mut add = |key: String,
                       command: Value,
                       subject: Option<String>,
                       repeat: bool|
         -> Result<(), String> {
            let next = Prefix {
                action: p.action,
                command,
                steps: if repeat {
                    p.steps.clone()
                } else {
                    p.steps[1..].to_vec()
                },
            };
            if let Some(c) = self.choice(key, next, subject)? {
                result.push(c);
            }
            Ok(())
        };
        match step.kind {
            "material" => {
                for id in &a.materials {
                    if !array(&c["materialIds"]).contains(&json!(id)) {
                        let mut ids = array(&c["materialIds"]).to_vec();
                        ids.push(json!(id));
                        add(
                            id.clone(),
                            extend(c, json!({"materialIds":ids})),
                            Some(id.clone()),
                            false,
                        )?;
                    }
                }
            }
            "chosen" => {
                add("random".into(), c.clone(), None, false)?;
                for k in &a.chosen {
                    add(
                        format!("kind:{}", crate::preparation::kind(k).key()),
                        extend(c, json!({"chosenKind":k})),
                        None,
                        false,
                    )?;
                }
            }
            "target" => {
                for t in &self.targets {
                    if step.unit_only && t.unit.is_none() {
                        continue;
                    }
                    let owner = t.unit.as_ref().map(|u| u.side()).unwrap_or(t.owner);
                    if (step.relation == "friend" && owner != self.actor)
                        || (step.relation == "enemy" && owner == self.actor)
                    {
                        continue;
                    }
                    if step.field == "sacrificeIds" && array(&c[step.field]).contains(&json!(t.id))
                    {
                        continue;
                    }
                    let mut command = c.clone();
                    command[step.field] = if step.field == "sacrificeIds" {
                        let mut ids = array(&c[step.field]).to_vec();
                        ids.push(json!(t.id));
                        json!(ids)
                    } else {
                        json!(t.id)
                    };
                    add(t.id.clone(), command, Some(t.id.clone()), false)?;
                }
            }
            "point" => {
                let points = if c["type"] == "synthesize" && array(&c["materialIds"]).len() == 3 {
                    let r = self
                        .catalog
                        .recipes
                        .iter()
                        .find(|r| r["id"] == c["recipeId"])
                        .ok_or("unknown recipe")?;
                    crate::synthesis::destinations(
                        s,
                        r,
                        &array(&c["materialIds"])
                            .iter()
                            .map(|v| text(v).to_string())
                            .collect::<Vec<_>>(),
                        self.catalog,
                    )
                } else if a.id.split(':').next() == Some("giant") && c.get("targetId").is_some() {
                    self.pieces
                        .get(text(&c["targetId"]))
                        .map(|u| geometry::expansion_anchors(s, u, self.catalog))
                        .unwrap_or_default()
                } else {
                    geometry::all_cells().collect()
                };
                for at in points {
                    add(
                        format!("{},{}", at.x, at.y),
                        extend(c, json!({"x":at.x,"y":at.y})),
                        None,
                        false,
                    )?;
                }
            }
            "death" => {
                for d in array(&s.extra["deaths"]) {
                    let id = text(&d["id"]);
                    add(
                        id.into(),
                        extend(c, json!({"deathId":id})),
                        Some(id.into()),
                        false,
                    )?;
                }
            }
            "row" | "column" => {
                for n in 1..=if step.kind == "row" { 13 } else { 9 } {
                    add(n.to_string(), extend(c, json!({step.kind:n})), None, false)?;
                }
            }
            "direction" => {
                add("auto".into(), c.clone(), None, false)?;
                for d in self.routes(c) {
                    add(d.clone(), extend(c, json!({"direction":d})), None, false)?;
                }
            }
            "path" => {
                if let Some(u) = self.pieces.get(text(&c["unitId"])) {
                    let path: Vec<Point> =
                        serde_json::from_value(c["path"].clone()).map_err(|e| e.to_string())?;
                    if !path.is_empty() {
                        add("commit-path".into(), c.clone(), None, false)?;
                    }
                    let points: Vec<_> = path
                        .last()
                        .map(|p| geometry::neighbors(*p).collect())
                        .unwrap_or_else(|| geometry::cells(u, u.at()));
                    for at in points {
                        let mut extended = path.clone();
                        extended.push(at);
                        if geometry::valid_attack_route(u, &extended, self.range(u)) {
                            add(
                                format!("{},{}", at.x, at.y),
                                extend(c, json!({"path":extended})),
                                None,
                                true,
                            )?;
                        }
                    }
                }
            }
            _ => return Err("unknown decision stage".into()),
        }
        Ok(result)
    }
    pub fn node(&mut self, cursor: &[usize]) -> Result<Rc<Node>, String> {
        if cursor.len() > 256 {
            return Err("action cursor too deep".into());
        }
        if let Some(n) = self.nodes.get(cursor) {
            return Ok(n.clone());
        }
        let node = if cursor.is_empty() {
            let mut choices = vec![];
            for i in 0..self.actions.len() {
                if let Some(c) = self.choice(i.to_string(), self.start(i), None)? {
                    choices.push(c);
                }
            }
            Node {
                cursor: vec![],
                prefix: None,
                stage: "action",
                choices,
            }
        } else {
            let parent = self.node(&cursor[..cursor.len() - 1])?;
            let p = parent
                .choices
                .get(*cursor.last().unwrap())
                .and_then(|c| c.next.clone())
                .ok_or("invalid or terminal action cursor")?;
            let p = self.prepare(p);
            Node {
                cursor: cursor.to_vec(),
                prefix: Some(p.command.clone()),
                stage: p.steps[0].kind,
                choices: self.expand(p)?,
            }
        };
        // TS 的 Map 返回同一不可变节点；这里共享所有权，避免深拷贝整组候选。
        let node = Rc::new(node);
        self.nodes.insert(cursor.to_vec(), node.clone());
        Ok(node)
    }
}
