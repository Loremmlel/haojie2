//! 状态分支共享扩展映射，写入先分离映射，再分离实际修改字段的 JSON 子树。
//! 与 TS forkPosition 对应；不可变键共享，普通 Clone 的值仍独立，保持原 Map 的序列化和键操作顺序。
use indexmap::{IndexMap, map::Entry};
use rustc_hash::FxBuildHasher;
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeMap};
use serde_json::{Map, Value};
use std::{
    ops::{Deref, DerefMut, Index},
    rc::Rc,
};

/// 只读分支共享整份 JSON；首次可变访问才复制，序列化仍是原始 JSON 值。
#[derive(Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SharedValue(Rc<Value>);
impl From<Value> for SharedValue {
    fn from(value: Value) -> Self {
        Self(Rc::new(value))
    }
}
impl Deref for SharedValue {
    type Target = Value;
    fn deref(&self) -> &Value {
        &self.0
    }
}
impl DerefMut for SharedValue {
    fn deref_mut(&mut self) -> &mut Value {
        Rc::make_mut(&mut self.0)
    }
}

#[derive(Default)]
pub struct ValueMap(Rc<IndexMap<Rc<str>, Rc<Value>, FxBuildHasher>>);

impl Clone for ValueMap {
    fn clone(&self) -> Self {
        Self(Rc::new(
            self.0
                .iter()
                .map(|(key, value)| (Rc::clone(key), Rc::new((**value).clone())))
                .collect(),
        ))
    }
}

impl ValueMap {
    pub fn fork(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
    /// 白名单视图只共享选中的字段；没有指回完整权威映射的指针。
    pub fn select(&self, keys: &[&str]) -> Self {
        Self(Rc::new(
            keys.iter()
                .filter_map(|key| {
                    self.0
                        .get_key_value(*key)
                        .map(|(key, value)| (Rc::clone(key), Rc::clone(value)))
                })
                .collect(),
        ))
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key).map(Rc::as_ref)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter().map(|(k, v)| (k.as_ref(), v.as_ref()))
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        if !self.0.contains_key(key) {
            return None;
        }
        Rc::make_mut(&mut self.0).get_mut(key).map(Rc::make_mut)
    }
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn insert(&mut self, key: String, value: Value) {
        Rc::make_mut(&mut self.0).insert(key.into(), Rc::new(value));
    }
    pub fn remove(&mut self, key: &str) {
        if self.0.contains_key(key) {
            Rc::make_mut(&mut self.0).swap_remove(key);
        }
    }
    pub fn entry(&mut self, key: impl Into<String>) -> ValueEntry<'_> {
        let key: String = key.into();
        ValueEntry(Rc::make_mut(&mut self.0).entry(key.into()))
    }
}

pub struct ValueEntry<'a>(Entry<'a, Rc<str>, Rc<Value>>);
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
        Ok(Self(Rc::new(
            map.into_iter()
                .map(|(key, value)| (key.into(), Rc::new(value)))
                .collect(),
        )))
    }
}
impl Serialize for ValueMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in self.0.iter() {
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
        let input = json!({"hands":{"1":[{"id":"card","kind":18}]},"obsolete":true,"log":[],"未知🌸":{"nested":[1]},"10":"数字键","2":"另一数字键"});
        let source: ValueMap = serde_json::from_value(input.clone()).unwrap();
        let mut branch = source.fork();
        assert!(Rc::ptr_eq(&source.0, &branch.0));
        assert!(branch.get_mut("absent").is_none());
        branch.remove("absent");
        assert!(Rc::ptr_eq(&source.0, &branch.0));
        let public = source.select(&["hands", "log"]);
        assert!(!Rc::ptr_eq(&source.0, &public.0));
        assert!(!public.contains_key("obsolete"));
        for (key, value) in public.0.iter() {
            let (original_key, original_value) = source.0.get_key_value(key.as_ref()).unwrap();
            assert!(Rc::ptr_eq(key, original_key));
            assert!(Rc::ptr_eq(value, original_value));
        }
        assert_eq!(
            public.iter().map(|(key, _)| key).collect::<Vec<_>>(),
            ["hands", "log"]
        );
        branch.get_mut("hands").unwrap()["1"][0]["kind"] = json!(1);
        assert!(Rc::ptr_eq(
            source.0.get_key_value("hands").unwrap().0,
            branch.0.get_key_value("hands").unwrap().0
        ));
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
        for (key, value) in snapshot.0.iter() {
            let (original_key, original_value) = branch.0.get_key_value(key.as_ref()).unwrap();
            assert!(Rc::ptr_eq(key, original_key));
            assert!(!Rc::ptr_eq(value, original_value));
        }
        branch.get_mut("hands").unwrap()["1"][0]["kind"] = json!(2);
        branch.get_mut("未知🌸").unwrap()["nested"][0] = json!(2);
        assert_eq!(serde_json::to_value(&source).unwrap(), input);
        assert_eq!(
            serde_json::to_string(&snapshot).unwrap(),
            serde_json::to_string(&expected).unwrap()
        );
    }
}
