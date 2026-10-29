//! 来源、动作、参数的惰性合法解码树；参数前缀不是游戏状态或搜索深度。
use crate::{
    inspection::Queries,
    model::{Catalog, Command, State},
    options::{self, Action, Step},
};
use serde_json::{Value, json};

#[derive(Clone)]
enum Stage {
    Root,
    Source(Vec<usize>),
    Parameters(usize, usize, usize),
    Leaf,
}
#[derive(Clone)]
struct Node {
    command: Command,
    stage: Stage,
    children: Option<Vec<usize>>,
}
pub struct Decoder {
    actions: Vec<Action>,
    steps: Vec<Vec<Step>>,
    nodes: Vec<Node>,
    queries: Queries,
}

impl Decoder {
    pub fn new(s: &State, catalog: &Catalog) -> Result<Self, String> {
        if s.extra.contains_key("mode") {
            return Err("decoder supports classic only".into());
        }
        let mut queries = Queries::default();
        let actions = options::actions(s, crate::runtime::viewer(s), catalog, &mut queries)?;
        let steps = actions
            .iter()
            .map(|a| {
                let mut result = vec![];
                if !a.materials.is_empty() {
                    result.extend(std::iter::repeat_n(Step::new("material"), 3));
                }
                result.extend_from_slice(&a.steps);
                if !a.chosen.is_empty() {
                    result.push(Step::new("chosen"));
                }
                if a.command.kind == "attack" && a.command.path.is_none() {
                    result.push(Step::new("direction"));
                }
                result
            })
            .collect();
        Ok(Self {
            actions,
            steps,
            queries,
            nodes: vec![Node {
                command: Command::new(""),
                stage: Stage::Root,
                children: None,
            }],
        })
    }
    fn completion(
        &mut self,
        s: &State,
        catalog: &Catalog,
        a: usize,
        c: Command,
        offset: usize,
        depth: usize,
    ) -> Result<Option<Command>, String> {
        if depth > 256 {
            return Err("command parameter depth exceeds rule representation".into());
        }
        if offset == self.steps[a].len() {
            return Ok((options::inspect(
                s,
                crate::runtime::viewer(s),
                &c,
                catalog,
                &mut self.queries,
            )? != "invalid")
                .then_some(c));
        }
        let choices = super::parameters::expand(
            s,
            &c,
            &self.actions[a],
            self.steps[a][offset],
            offset + 1 == self.steps[a].len(),
            catalog,
            &mut self.queries,
        )?;
        for (next, repeat) in choices {
            if let Some(found) = self.completion(
                s,
                catalog,
                a,
                next,
                offset + usize::from(!repeat),
                depth + 1,
            )? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }
    fn push(&mut self, command: Command, stage: Stage) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            command,
            stage,
            children: None,
        });
        id
    }
    pub fn choices(
        &mut self,
        s: &State,
        catalog: &Catalog,
        prefix: usize,
    ) -> Result<Value, String> {
        let node = self
            .nodes
            .get(prefix)
            .ok_or("unknown decoder prefix")?
            .clone();
        let stage = match &node.stage {
            Stage::Root => 0,
            Stage::Source(_) => 1,
            _ => 2,
        };
        let focus = match &node.stage {
            Stage::Root => "source",
            Stage::Parameters(a, offset, _) => {
                let step = self.steps[*a][*offset];
                if step.kind == "target" {
                    step.field
                } else {
                    step.kind
                }
            }
            _ => "action",
        };
        if matches!(node.stage, Stage::Leaf) {
            return Ok(json!({"leaf":true,"command":node.command}));
        }
        if node.children.is_none() {
            let mut children = vec![];
            match node.stage {
                Stage::Root => {
                    let mut groups: Vec<(String, Command, Vec<usize>)> = vec![];
                    for a in 0..self.actions.len() {
                        let c = self.actions[a].command.clone();
                        if self.completion(s, catalog, a, c.clone(), 0, 0)?.is_none() {
                            continue;
                        }
                        let key = if let Some(id) = c.unit_id.as_deref() {
                            format!("unit:{id}")
                        } else if let Some(id) = c.card_id.as_deref() {
                            format!("card:{id}")
                        } else {
                            "global".into()
                        };
                        if let Some(group) = groups.iter_mut().find(|g| g.0 == key) {
                            group.2.push(a);
                        } else {
                            let mut source = Command::new("");
                            source.unit_id = c.unit_id.clone();
                            source.card_id = c.card_id.clone();
                            groups.push((key, source, vec![a]));
                        }
                    }
                    for (_, c, group) in groups {
                        children.push(self.push(c, Stage::Source(group)));
                    }
                }
                Stage::Source(group) => {
                    for a in group {
                        let stage = if self.steps[a].is_empty() {
                            Stage::Leaf
                        } else {
                            Stage::Parameters(a, 0, 0)
                        };
                        children.push(self.push(self.actions[a].command.clone(), stage));
                    }
                }
                Stage::Parameters(a, offset, depth) => {
                    let choices = super::parameters::expand(
                        s,
                        &node.command,
                        &self.actions[a],
                        self.steps[a][offset],
                        offset + 1 == self.steps[a].len(),
                        catalog,
                        &mut self.queries,
                    )?;
                    for (next, repeat) in choices {
                        let offset = offset + usize::from(!repeat);
                        if self
                            .completion(s, catalog, a, next.clone(), offset, depth + 1)?
                            .is_some()
                        {
                            let stage = if offset == self.steps[a].len() {
                                Stage::Leaf
                            } else {
                                Stage::Parameters(a, offset, depth + 1)
                            };
                            children.push(self.push(next, stage));
                        }
                    }
                }
                Stage::Leaf => unreachable!(),
            }
            self.nodes[prefix].children = Some(children);
        }
        let children = self.nodes[prefix].children.as_ref().unwrap();
        Ok(
            json!({"leaf":false,"next":children,"stage":stage,"focus":focus,
            "choices":children.iter().map(|id|&self.nodes[*id].command).collect::<Vec<_>>()}),
        )
    }
    pub fn trace(
        &mut self,
        s: &State,
        catalog: &Catalog,
        command: &Command,
    ) -> Result<Vec<(usize, usize)>, String> {
        // 合成材料是集合；逐前缀匹配集合成员，其他数组保留顺序。
        fn agrees(part: &Value, full: &Value) -> bool {
            part.as_object().is_some_and(|p| {
                p.iter().all(|(k, v)| {
                    if k == "type" && v == "" {
                        return true;
                    }
                    if k == "materialIds" {
                        return v.as_array().is_some_and(|a| {
                            a.iter()
                                .all(|v| full[k].as_array().is_some_and(|f| f.contains(v)))
                        });
                    }
                    if k == "path" || k == "sacrificeIds" {
                        return v
                            .as_array()
                            .is_some_and(|a| full[k].as_array().is_some_and(|f| f.starts_with(a)));
                    }
                    full.get(k) == Some(v)
                })
            })
        }
        fn visit(
            d: &mut Decoder,
            s: &State,
            catalog: &Catalog,
            id: usize,
            target: &Value,
            path: &mut Vec<(usize, usize)>,
        ) -> Result<bool, String> {
            let front = d.choices(s, catalog, id)?;
            if front["leaf"] == true {
                return Ok(agrees(&front["command"], target) && agrees(target, &front["command"]));
            }
            for (i, c) in front["choices"].as_array().unwrap().iter().enumerate() {
                if agrees(c, target) {
                    path.push((id, i));
                    if visit(
                        d,
                        s,
                        catalog,
                        front["next"][i].as_u64().unwrap() as usize,
                        target,
                        path,
                    )? {
                        return Ok(true);
                    }
                    path.pop();
                }
            }
            Ok(false)
        }
        let mut path = vec![];
        if !visit(
            self,
            s,
            catalog,
            0,
            &serde_json::to_value(command).map_err(|e| e.to_string())?,
            &mut path,
        )? {
            return Err("legal teacher command absent from decoder".into());
        }
        Ok(path)
    }
}
