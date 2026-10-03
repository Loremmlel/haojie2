//! 同 TS 惰性参数树；节点只缓存当前公开局面，不做评分或候选裁剪。
use crate::actions::{self, Action, Step, array, text};
use crate::geometry;
use crate::model::{Catalog, Command, Point, State, Unit};
use serde_json::{Value, json};
use std::borrow::Cow;
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone)]
pub struct Prefix {
    action: usize,
    pub command: Command,
    steps: Vec<Step>,
}
#[derive(Clone)]
pub struct Choice {
    pub key: String,
    pub command: Command,
    pub status: &'static str,
    pub subject: Option<String>,
    pub next: Option<Prefix>,
}
#[derive(Clone)]
pub struct Node {
    pub cursor: Vec<usize>,
    pub prefix: Option<Command>,
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
            value["prefix"] = json!(c);
        }
        value
    }
}
pub struct Tree<'a> {
    pub observation: Option<&'a Value>,
    pub state: Cow<'a, State>,
    pub actor: usize,
    pub catalog: &'a Catalog,
    actions: Vec<Action>,
    nodes: HashMap<Vec<usize>, Rc<Node>>,
    queries: RefCell<crate::inspection::Queries>,
    targets: Vec<geometry::Target>,
    target_index: HashMap<String, usize>,
    stats: RefCell<HashMap<crate::entities::EntityHandle, crate::stats::Stats>>,
    routes: RefCell<HashMap<(crate::entities::EntityHandle, usize), Vec<String>>>,
    pub encoding: OnceCell<Result<crate::encoding::BaseEncoding, String>>,
    pub encoding_workspace: RefCell<crate::encoding::Workspace>,
}
impl<'a> Tree<'a> {
    pub fn new(observation: &'a Value, actor: usize, catalog: &'a Catalog) -> Result<Self, String> {
        if ![1, 2].contains(&actor) {
            return Err("invalid training actor".into());
        }
        Self::with_position(
            Some(observation),
            Cow::Owned(actions::position(observation)?),
            actor,
            catalog,
        )
    }
    pub fn from_view(
        view: &'a crate::runtime::PublicPosition,
        catalog: &'a Catalog,
    ) -> Result<Self, String> {
        Self::with_position(None, Cow::Borrowed(view.position()), view.viewer, catalog)
    }
    fn with_position(
        observation: Option<&'a Value>,
        state: Cow<'a, State>,
        actor: usize,
        catalog: &'a Catalog,
    ) -> Result<Self, String> {
        let mut queries = crate::inspection::Queries::default();
        let actions = actions::actions(&state, actor, catalog, &mut queries)?;
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
            targets,
            target_index,
            stats: RefCell::new(HashMap::new()),
            routes: RefCell::new(HashMap::new()),
            encoding: OnceCell::new(),
            encoding_workspace: RefCell::default(),
        })
    }
    fn routes(&self, c: &Command) -> Vec<String> {
        let s = &self.state;
        let Some(source) = c.unit_id.as_deref().and_then(|id| s.entities.handle(s, id)) else {
            return vec![];
        };
        let Some(target) = c
            .target_id
            .as_deref()
            .and_then(|id| self.target_index.get(id))
            .copied()
        else {
            return vec![];
        };
        let key = (source, target);
        if let Some(routes) = self.routes.borrow().get(&key) {
            return routes.clone();
        }
        let Some(u) = s.entity(source) else {
            return vec![];
        };
        let t = &self.targets[target];
        // TS selectableAttackRoutes 有结果时仍来自同一 attackRoutes，否则回退到全方向查询。
        let routes = geometry::attack_routes(s, u, t, self.range(u), u.piercing());
        self.routes.borrow_mut().insert(key, routes.clone());
        routes
    }
    fn range(&self, u: &Unit) -> f64 {
        self.stats
            .borrow_mut()
            .entry(
                self.state
                    .entities
                    .handle(&self.state, &u.id)
                    .expect("当前只读实体"),
            )
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
        // 叶子没有后续前缀，直接交付这份命令；参数节点仍保留自己的不可变前缀。
        let (command, next) = if p.steps.is_empty() {
            (p.command, None)
        } else {
            (p.command.clone(), Some(p))
        };
        Ok(Some(Choice {
            key,
            command,
            status,
            subject,
            next,
        }))
    }
    fn start(&self, i: usize) -> Prefix {
        let a = &self.actions[i];
        let mut steps = vec![];
        if !a.materials.is_empty() {
            steps.extend(std::iter::repeat_n(Step::new("material"), 3));
        }
        steps.extend(a.steps.iter().cloned());
        if !a.chosen.is_empty() {
            steps.push(Step::new("chosen"));
        }
        if a.command.kind == "attack" && a.command.path.is_none() {
            steps.push(Step::new("direction"));
        }
        Prefix {
            action: i,
            command: a.command.clone(),
            steps,
        }
    }
    fn expand(&self, p: Prefix) -> Result<Vec<Choice>, String> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Candidates);
        let c = &p.command;
        let a = &self.actions[p.action];
        let step = &p.steps[0];
        let s = &self.state;
        let mut result = vec![];
        let mut add = |key: String,
                       command: Command,
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
                    if !c.material_ids.as_deref().unwrap_or(&[]).contains(id) {
                        let mut ids = c.material_ids.clone().unwrap_or_default();
                        ids.push(id.clone());
                        add(
                            id.clone(),
                            Command {
                                material_ids: Some(ids),
                                ..c.clone()
                            },
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
                        format!("kind:{}", k.key()),
                        Command {
                            chosen_kind: Some(k.clone()),
                            ..c.clone()
                        },
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
                    if step.field == "sacrificeIds"
                        && c.sacrifice_ids.as_deref().unwrap_or(&[]).contains(&t.id)
                    {
                        continue;
                    }
                    let mut command = c.clone();
                    match step.field {
                        "sacrificeIds" => command
                            .sacrifice_ids
                            .get_or_insert_with(Vec::new)
                            .push(t.id.clone()),
                        "secondId" => command.second_id = Some(t.id.clone()),
                        "targetId" => command.target_id = Some(t.id.clone()),
                        _ => return Err("unknown target field".into()),
                    }
                    add(t.id.clone(), command, Some(t.id.clone()), false)?;
                }
            }
            "point" => {
                let points = if c.kind == "synthesize"
                    && c.material_ids.as_deref().unwrap_or(&[]).len() == 3
                {
                    let r = self
                        .catalog
                        .recipes
                        .iter()
                        .find(|r| r["id"].as_str() == c.recipe_id.as_deref())
                        .ok_or("unknown recipe")?;
                    crate::synthesis::destinations(
                        s,
                        r,
                        c.material_ids.as_deref().unwrap_or(&[]),
                        self.catalog,
                    )
                } else if a.id.split(':').next() == Some("giant") && c.target_id.is_some() {
                    self.state
                        .unit(c.target_id.as_deref().unwrap_or(""))
                        .map(|u| geometry::expansion_anchors(s, u, self.catalog))
                        .unwrap_or_default()
                } else {
                    geometry::all_cells().collect()
                };
                for at in points {
                    add(
                        format!("{},{}", at.x, at.y),
                        Command {
                            x: Some(at.x),
                            y: Some(at.y),
                            ..c.clone()
                        },
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
                        Command {
                            death_id: Some(id.into()),
                            ..c.clone()
                        },
                        Some(id.into()),
                        false,
                    )?;
                }
            }
            "row" | "column" => {
                for n in 1..=if step.kind == "row" { 13 } else { 9 } {
                    let mut command = c.clone();
                    if step.kind == "row" {
                        command.row = Some(n as f64);
                    } else {
                        command.column = Some(n as f64);
                    }
                    add(n.to_string(), command, None, false)?;
                }
            }
            "direction" => {
                add("auto".into(), c.clone(), None, false)?;
                for d in self.routes(c) {
                    add(
                        d.clone(),
                        Command {
                            direction: Some(d),
                            ..c.clone()
                        },
                        None,
                        false,
                    )?;
                }
            }
            "path" => {
                if let Some(u) = self.state.unit(c.unit_id.as_deref().unwrap_or("")) {
                    let path: &[Point] = c.path.as_deref().unwrap_or(&[]);
                    if !path.is_empty() {
                        add("commit-path".into(), c.clone(), None, false)?;
                    }
                    let points: Vec<_> = path
                        .last()
                        .map(|p| geometry::neighbors(*p).collect())
                        .unwrap_or_else(|| geometry::cells(u, u.at()));
                    for at in points {
                        let mut extended = path.to_vec();
                        extended.push(at);
                        if geometry::valid_attack_route(u, &extended, self.range(u)) {
                            add(
                                format!("{},{}", at.x, at.y),
                                Command {
                                    path: Some(extended),
                                    ..c.clone()
                                },
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
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Candidates);
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
