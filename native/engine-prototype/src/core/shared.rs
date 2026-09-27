//! 状态分支共享顶层扩展字段，取得可变引用前复制该字段的 JSON 子树。
//! 与 TS forkPosition 对应；普通 Clone 仍导出独立快照，保持原 Map 的序列化和键操作顺序。
use indexmap::{IndexMap, map::Entry};
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
use serde_json::{Map, Value};
use std::{ops::Index, rc::Rc};

#[derive(Default)]
pub struct ValueMap(IndexMap<String, Rc<Value>>);

impl Clone for ValueMap {
    fn clone(&self) -> Self {
        Self(
            self.0
                .iter()
                .map(|(key, value)| (key.clone(), Rc::new((**value).clone())))
                .collect(),
        )
    }
}

impl ValueMap {
    pub fn fork(&self) -> Self {
        Self(self.0.clone())
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key).map(Rc::as_ref)
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.0.get_mut(key).map(Rc::make_mut)
    }
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn insert(&mut self, key: String, value: Value) {
        self.0.insert(key, Rc::new(value));
    }
    pub fn remove(&mut self, key: &str) {
        self.0.swap_remove(key);
    }
    pub fn entry(&mut self, key: impl Into<String>) -> ValueEntry<'_> {
        ValueEntry(self.0.entry(key.into()))
    }
}

pub struct ValueEntry<'a>(Entry<'a, String, Rc<Value>>);
impl<'a> ValueEntry<'a> {
    pub fn or_insert_with(self, value: impl FnOnce() -> Value) -> &'a mut Value {
        Rc::make_mut(self.0.or_insert_with(|| Rc::new(value())))
    }
}

impl Index<&str> for ValueMap {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.get(key).expect("入口已校验扩展字段")
    }
}
impl<'de> Deserialize<'de> for ValueMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let map = Map::<String, Value>::deserialize(deserializer)?;
        Ok(Self(
            map.into_iter()
                .map(|(key, value)| (key, Rc::new(value)))
                .collect(),
        ))
    }
}
impl Serialize for ValueMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value.as_ref())?;
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn branch_and_snapshot_keep_source_isolated_and_preserve_map_protocol() {
        let input = json!({"hands":{"1":[{"id":"card","kind":18}]},"obsolete":true,"log":[]});
        let source: ValueMap = serde_json::from_value(input.clone()).unwrap();
        let mut branch = source.fork();
        branch.get_mut("hands").unwrap()["1"][0]["kind"] = json!(1);
        branch.remove("obsolete");
        branch
            .entry("log")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .unwrap()
            .push(json!("分支日志"));
        let mut expected = input.as_object().unwrap().clone();
        expected.get_mut("hands").unwrap()["1"][0]["kind"] = json!(1);
        expected.remove("obsolete");
        expected
            .get_mut("log")
            .unwrap()
            .as_array_mut()
            .unwrap()
            .push(json!("分支日志"));
        let snapshot = branch.clone();
        branch.get_mut("hands").unwrap()["1"][0]["kind"] = json!(2);
        assert_eq!(serde_json::to_value(&source).unwrap(), input);
        assert_eq!(
            serde_json::to_string(&snapshot).unwrap(),
            serde_json::to_string(&expected).unwrap()
        );
    }
}
