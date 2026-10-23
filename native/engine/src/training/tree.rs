//! 同 TS 惰性参数树；节点只缓存当前公开局面，不做评分或候选裁剪。
use crate::actions::{self, Action, Step, text};
use crate::geometry;
use crate::model::{Catalog, Command, Point, State, Unit};
use rustc_hash::FxHashMap as HashMap;
use serde::Serialize;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::cell::{OnceCell, RefCell};
use std::rc::Rc;

#[derive(Clone)]
pub struct Prefix {
    action: usize,
    pub command: Command,
    offset: usize,
}
#[derive(Clone)]
pub enum ChoiceKey {
    Owned(String),
    Root(usize),
    Point(Point),
    SharedId(Rc<str>),
    Static(&'static str),
}
impl ChoiceKey {
    pub fn is_commit_path(&self) -> bool {
        match self {
            Self::Static(value) => *value == "commit-path",
            Self::Owned(value) => value == "commit-path",
            Self::SharedId(value) => value.as_ref() == "commit-path",
            Self::Root(_) | Self::Point(_) => false,
        }
    }
}
impl From<String> for ChoiceKey {
    fn from(value: String) -> Self {
        Self::Owned(value)
    }
}
impl From<&'static str> for ChoiceKey {
    fn from(value: &'static str) -> Self {
        Self::Static(value)
    }
}
impl Serialize for ChoiceKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Owned(value) => serializer.serialize_str(value),
            Self::Root(index) => serializer.serialize_str(&index.to_string()),
            Self::Point(at) => serializer.serialize_str(&format!("{},{}", at.x, at.y)),
            Self::SharedId(value) => serializer.serialize_str(value),
            Self::Static(value) => serializer.serialize_str(value),
        }
    }
}
#[derive(Clone)]
pub struct Choice {
    pub key: ChoiceKey,
    pub command: Command,
    pub status: &'static str,
    pub subject: Option<Rc<str>>,
    pub next: Option<Rc<Prefix>>,
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
type RouteKey = (crate::entities::EntityHandle, Rc<str>);

pub struct Tree<'a> {
    pub observation: Option<&'a Value>,
    pub state: Cow<'a, State>,
    pub actor: usize,
    pub catalog: &'a Catalog,
    actions: Vec<Action>,
    nodes: HashMap<Vec<usize>, Rc<Node>>,
    queries: RefCell<crate::inspection::Queries>,
    targets: OnceCell<Vec<geometry::Target>>,
    routes: RefCell<HashMap<RouteKey, Vec<&'static str>>>,
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
    /// 可暂停运行时拥有公开副本；不借用权威状态，也不跨线程移动 Rc。
    pub fn own_view(
        view: crate::runtime::PublicPosition,
        catalog: &'a Catalog,
    ) -> Result<Self, String> {
        let viewer = view.viewer;
        Self::with_position(None, Cow::Owned(view.into_position()), viewer, catalog)
    }
    fn with_position(
        observation: Option<&'a Value>,
        state: Cow<'a, State>,
        actor: usize,
        catalog: &'a Catalog,
    ) -> Result<Self, String> {
        let mut queries = crate::inspection::Queries::default();
        let mut actions = actions::actions(&state, actor, catalog, &mut queries)?;
        // 同一动作的全部前缀共享固定步骤，分支只推进 offset；隐式参数在过滤后补齐。
        for action in &mut actions {
            let chosen = usize::from(!action.chosen.is_empty());
            let direction =
                usize::from(action.command.kind == "attack" && action.command.path.is_none());
            if action.materials.is_empty() && chosen == 0 && direction == 0 {
                continue;
            }
            let mut steps = Vec::with_capacity(3 + action.steps.len() + chosen + direction);
            if !action.materials.is_empty() {
                steps.extend(std::iter::repeat_n(Step::new("material"), 3));
            }
            steps.extend_from_slice(&action.steps);
            if chosen != 0 {
                steps.push(Step::new("chosen"));
            }
            if direction != 0 {
                steps.push(Step::new("direction"));
            }
            action.steps = Cow::Owned(steps);
        }
        Ok(Self {
            observation,
            state,
            actor,
            catalog,
            actions,
            nodes: HashMap::default(),
            queries: RefCell::new(queries),
            targets: OnceCell::new(),
            routes: RefCell::new(HashMap::default()),
            encoding: OnceCell::new(),
            encoding_workspace: RefCell::default(),
        })
    }
    fn routes(&self, c: &Command) -> Vec<&'static str> {
        let s = &self.state;
        let Some(source) = c.unit_id.as_deref().and_then(|id| s.entities.handle(s, id)) else {
            return vec![];
        };
        let Some(target) = &c.target_id else {
            return vec![];
        };
        let key = (source, Rc::clone(target));
        if let Some(routes) = self.routes.borrow().get(&key) {
            return routes.clone();
        }
        let Some(u) = s.entity(source) else {
            return vec![];
        };
        // 对应 TS：先按完整目标域的实体顺序找首项，休眠同名地标不遮住后续活跃实体。
        let Some(t) = s
            .pieces()
            .find(|u| u.id.as_str() == target.as_ref())
            .map(geometry::Target::from)
            .or_else(|| geometry::find_target(s, target))
        else {
            return vec![];
        };
        // 与 TS 一样，方向候选和叶子攻击预检共享本决策内的精确路径。
        let range = self.range(u);
        let routes = self
            .queries
            .borrow_mut()
            .attacks
            .directions(s, u, &t, range, u.piercing());
        self.routes.borrow_mut().insert(key, routes.clone());
        routes
    }
    fn range(&self, u: &Unit) -> f64 {
        self.queries
            .borrow_mut()
            .attributes(&self.state, u, self.catalog)
            .range
    }
    fn prepare(&self, mut p: Prefix) -> Prefix {
        while self.actions[p.action]
            .steps
            .get(p.offset)
            .is_some_and(|s| s.kind == "direction")
            && self.routes(&p.command).is_empty()
        {
            p.offset += 1;
        }
        p
    }
    fn choice(
        &self,
        key: ChoiceKey,
        p: Prefix,
        subject: Option<Rc<str>>,
        initial: bool,
    ) -> Result<Option<Choice>, String> {
        let p = self.prepare(p);
        let status = if p.offset == self.actions[p.action].steps.len() {
            // 根动作命令没有经过参数扩展，复用同一只读局面已经完成的预检。
            match initial
                .then_some(self.actions[p.action].initial_inspection)
                .flatten()
            {
                Some(status) => status,
                None => actions::inspect(
                    &self.state,
                    self.actor,
                    &p.command,
                    self.catalog,
                    &mut self.queries.borrow_mut(),
                )?,
            }
        } else {
            "parameter"
        };
        if status == "invalid" {
            return Ok(None);
        }
        // 叶子没有后续前缀，直接交付这份命令；参数节点仍保留自己的不可变前缀。
        let (command, next) = if p.offset == self.actions[p.action].steps.len() {
            (p.command, None)
        } else {
            (p.command.clone(), Some(Rc::new(p)))
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
        Prefix {
            action: i,
            command: a.command.clone(),
            offset: 0,
        }
    }
    fn expand(&self, p: &Prefix) -> Result<Vec<Choice>, String> {
        #[cfg(feature = "kernel-profile")]
        let _profile = crate::profile::scope(crate::profile::Phase::Candidates);
        let c = &p.command;
        let a = &self.actions[p.action];
        let step = &a.steps[p.offset];
        let s = &self.state;
        if step.kind == "point"
            && a.steps.len() - p.offset == 1
            && let Some(domain) = actions::points(
                s,
                self.actor,
                c,
                self.catalog,
                &mut self.queries.borrow_mut(),
            )?
        {
            return Ok(domain
                .into_iter()
                .map(|(at, status)| Choice {
                    key: ChoiceKey::Point(at),
                    command: Command {
                        x: Some(at.x),
                        y: Some(at.y),
                        ..c.clone()
                    },
                    status,
                    subject: None,
                    next: None,
                })
                .collect());
        }
        let mut result = vec![];
        let mut add = |key: ChoiceKey,
                       command: Command,
                       subject: Option<Rc<str>>,
                       repeat: bool|
         -> Result<(), String> {
            let next = Prefix {
                action: p.action,
                command,
                offset: p.offset + usize::from(!repeat),
            };
            if let Some(c) = self.choice(key, next, subject, false)? {
                result.push(c);
            }
            Ok(())
        };
        match step.kind {
            "material" => {
                for id in &a.materials {
                    if !c.material_ids.as_deref().unwrap_or(&[]).contains(id) {
                        let shared_id: Rc<str> = Rc::from(id.as_str());
                        let mut ids = c.material_ids.clone().unwrap_or_default();
                        ids.push(id.clone());
                        add(
                            ChoiceKey::SharedId(Rc::clone(&shared_id)),
                            {
                                let mut command = c.clone();
                                command.material_ids = Some(ids);
                                command
                            },
                            Some(shared_id),
                            false,
                        )?;
                    }
                }
            }
            "chosen" => {
                add("random".into(), c.clone(), None, false)?;
                for k in &a.chosen {
                    add(
                        format!("kind:{}", k.key()).into(),
                        {
                            let mut command = c.clone();
                            command.chosen_kind = Some(k.clone());
                            command
                        },
                        None,
                        false,
                    )?;
                }
            }
            "target" => {
                // 只有实际进入目标参数节点时才构造整表；保留全部重复项与原有顺序。
                for t in self.targets.get_or_init(|| geometry::targets(s)) {
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
                    let shared_id: Rc<str> = Rc::from(t.id.as_str());
                    let mut command = c.clone();
                    match step.field {
                        "sacrificeIds" => command
                            .sacrifice_ids
                            .get_or_insert_with(Vec::new)
                            .push(t.id.clone()),
                        "secondId" => command.second_id = Some(t.id.clone()),
                        "targetId" => command.target_id = Some(Rc::clone(&shared_id)),
                        _ => return Err("unknown target field".into()),
                    }
                    add(
                        ChoiceKey::SharedId(Rc::clone(&shared_id)),
                        command,
                        Some(shared_id),
                        false,
                    )?;
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
                        ChoiceKey::Point(at),
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
                for d in s.deaths.iter() {
                    let id = text(&d["id"]);
                    let shared_id: Rc<str> = Rc::from(id);
                    add(
                        ChoiceKey::SharedId(Rc::clone(&shared_id)),
                        {
                            let mut command = c.clone();
                            command.death_id = Some(id.into());
                            command
                        },
                        Some(shared_id),
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
                    add(n.to_string().into(), command, None, false)?;
                }
            }
            "direction" => {
                add("auto".into(), c.clone(), None, false)?;
                for d in self.routes(c) {
                    add(
                        ChoiceKey::Static(d),
                        {
                            let mut command = c.clone();
                            command.direction = Some(Cow::Borrowed(d));
                            command
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
                                ChoiceKey::Point(at),
                                {
                                    let mut command = c.clone();
                                    command.path = Some(extended);
                                    command
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
                if let Some(c) = self.choice(ChoiceKey::Root(i), self.start(i), None, true)? {
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
                .and_then(|c| c.next.as_ref())
                .ok_or("invalid or terminal action cursor")?;
            Node {
                cursor: cursor.to_vec(),
                prefix: Some(p.command.clone()),
                stage: self.actions[p.action].steps[p.offset].kind,
                choices: self.expand(p)?,
            }
        };
        // TS 的 Map 返回同一不可变节点；这里共享所有权，避免深拷贝整组候选。
        let node = Rc::new(node);
        self.nodes.insert(cursor.to_vec(), node.clone());
        Ok(node)
    }

    /// 在线状态机只保留当前栈的节点；已排除的分支不再持有整组候选。
    pub fn retain_path(&mut self, cursor: &[usize]) {
        self.nodes.retain(|path, _| cursor.starts_with(path));
    }

    #[cfg(test)]
    pub(crate) fn inject_node(&mut self, node: Node) {
        self.nodes.insert(node.cursor.clone(), Rc::new(node));
    }
}

#[cfg(test)]
mod key_tests {
    use super::*;

    #[test]
    fn deferred_choice_keys_preserve_wire_and_path_commit_semantics() {
        for (key, expected, commit) in [
            (ChoiceKey::Owned("7".into()), "7", false),
            (ChoiceKey::Root(7), "7", false),
            (ChoiceKey::Point(Point { x: 9.0, y: 13.0 }), "9,13", false),
            (ChoiceKey::SharedId(Rc::from("随从-1")), "随从-1", false),
            (ChoiceKey::Static("auto"), "auto", false),
            (ChoiceKey::Static("commit-path"), "commit-path", true),
            (ChoiceKey::Owned("commit-path".into()), "commit-path", true),
            (
                ChoiceKey::SharedId(Rc::from("commit-path")),
                "commit-path",
                true,
            ),
        ] {
            assert_eq!(serde_json::to_value(&key).unwrap(), json!(expected));
            assert_eq!(key.is_commit_path(), commit);
        }
    }
}
