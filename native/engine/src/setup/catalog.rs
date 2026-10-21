//! 共用规则包边界：完整校验后返回不可变目录，开发会话与训练宿主分别持有它。
use crate::{
    encoding,
    model::{self, Catalog, Definition, PROTOCOL, RULESET},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const RULES: &[u8] = include_bytes!("../../data/rules.json");

pub fn rules_hash() -> String {
    format!("{:x}", Sha256::digest(RULES))
}

/// 使用随二进制交付的规则；先核对原始字节哈希，再执行与开发初始化相同的校验。
pub fn embedded() -> Result<Catalog, String> {
    let manifest: Value = serde_json::from_slice(include_bytes!("../../data/manifest.json"))
        .map_err(|e| e.to_string())?;
    if manifest["sha256"] != rules_hash() {
        return Err("embedded rules package hash mismatch".into());
    }
    from_request(serde_json::from_slice(RULES).map_err(|e| e.to_string())?)
}

/// 消费一份初始化请求；不修改驻留状态，任何字段失败都丢弃本次构建结果。
pub fn from_request(mut request: Value) -> Result<Catalog, String> {
    if request["protocol"] != PROTOCOL || request["ruleset"] != RULESET {
        return Err("protocol/ruleset mismatch".into());
    }
    let printed: Vec<Value> =
        serde_json::from_value(request["catalog"].clone()).map_err(|e| e.to_string())?;
    let definitions: Vec<Definition> =
        serde_json::from_value(request["catalog"].take()).map_err(|e| e.to_string())?;
    let mut entries = BTreeMap::new();
    for (mut d, p) in definitions.into_iter().zip(&printed) {
        d.printed_mage = p.get("mage").and_then(Value::as_bool);
        d.printed_aura = p.get("aura").and_then(Value::as_bool);
        if entries.insert(d.id.key(), std::rc::Rc::new(d)).is_some() {
            return Err("duplicate catalog kind".into());
        }
    }
    if entries.is_empty() {
        return Err("empty catalog".into());
    }
    let combat = request["combat"].take();
    let mut rule_values = rustc_hash::FxHashMap::default();
    for path in [
        "/accumulator/attack",
        "/accumulator/range",
        "/accumulator/max",
        "/goldSpellChance",
        "/sageAuraAttack",
        "/littleGoldImmunity",
        "/kingAttackImmunity",
        "/frontDamageCap",
        "/slayerReflectRate",
        "/catapultMarkDamage",
        "/charger/heavyChance",
        "/charger/criticalChance",
        "/charger/heavyBonus",
        "/charger/bonus",
        "/superCritical/lethalChance",
        "/superCritical/doubleChance",
        "/superCritical/lethalDamage",
        "/vampire/base",
        "/vampire/perKill",
        "/minerBaseDamage",
        "/sacrificeMaxHpCost",
        "/giantAreaDamage",
        "/firelord/radius",
        "/firelord/damage",
        "/firelord/splash",
        "/archmageCounterChance",
        "/archmageCounterHealth",
    ] {
        let value = combat.pointer(path);
        if !value.is_some_and(Value::is_number) {
            return Err(format!("missing combat rule: {path}"));
        }
        // 只缓存已校验的固定路径；扩展路径继续使用原 JSON Pointer 规则。
        if let Some(value) = value.and_then(Value::as_f64) {
            rule_values.insert(path, value);
        }
    }
    let pools: BTreeMap<String, Vec<model::Kind>> =
        serde_json::from_value(request["summonPools"].take()).map_err(|e| e.to_string())?;
    for name in ["normal", "ultimate", "shrine"] {
        if !pools.get(name).is_some_and(|pool| {
            !pool.is_empty() && pool.iter().all(|k| entries.contains_key(&k.key()))
        }) {
            return Err(format!("invalid summon pool: {name}"));
        }
    }
    if pools["shrine"].len() < 3 {
        return Err("shrine pool requires at least 3 entries".into());
    }
    let recipes: Vec<Value> =
        serde_json::from_value(request["recipes"].take()).map_err(|e| e.to_string())?;
    if recipes.is_empty()
        || recipes.iter().any(|r| {
            r["id"].as_str().is_none_or(str::is_empty)
                || !r["source"]
                    .as_str()
                    .is_some_and(|s| ["board", "hand"].contains(&s))
                || ["material", "result"].iter().any(|k| {
                    serde_json::from_value::<model::Kind>(r[k].clone())
                        .map_or(true, |k| !entries.contains_key(&k.key()))
                })
        })
    {
        return Err("invalid synthesis recipes".into());
    }
    let encoding = request["encoding"].take();
    if !encoding.is_null() {
        encoding::validate_schema(&encoding)?;
    }
    Ok(Catalog {
        numbers: std::array::from_fn(|i| entries.get(&i.to_string()).cloned()),
        kind_codes: encoding
            .get("kinds")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(i, k)| (crate::preparation::kind(k), i + 1))
            .collect(),
        vocab_codes: encoding
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(key, value)| {
                value.as_array().map(|vs| {
                    (
                        key.clone(),
                        vs.iter()
                            .enumerate()
                            .filter_map(|(i, v)| v.as_str().map(|v| (v.to_owned(), i + 1)))
                            .collect(),
                    )
                })
            })
            .collect(),
        entries,
        combat,
        rule_values,
        pools,
        recipes,
        encoding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn numeric_rule_cache_preserves_loaded_values_pointer_fallback_and_validation_order() {
        let mut request: Value = serde_json::from_slice(RULES).unwrap();
        request["combat"]["sageAuraAttack"] = json!(-0.0);
        request["combat"]["charger"]["heavyChance"] = json!(0.125);
        request["combat"]["extension"] = json!({"~2": -3.75, "a/b": [0.1]});
        let combat = request["combat"].clone();
        let catalog = from_request(request).unwrap();
        assert_eq!(catalog.combat, combat);
        assert!(catalog.rule_values.contains_key("/charger/heavyChance"));
        assert!(!catalog.rule_values.contains_key("/extension/~2"));
        for path in [
            "/sageAuraAttack",
            "/charger/heavyChance",
            "/extension/~2",
            "/extension/~02",
            "/extension/a~1b/0",
        ] {
            assert_eq!(
                catalog.rule(path).to_bits(),
                combat
                    .pointer(path)
                    .and_then(Value::as_f64)
                    .unwrap()
                    .to_bits(),
                "{path}"
            );
        }

        let mut invalid: Value = serde_json::from_slice(RULES).unwrap();
        invalid["combat"]["accumulator"]["attack"] = Value::Null;
        invalid["combat"]["goldSpellChance"] = Value::Null;
        assert_eq!(
            from_request(invalid).err().as_deref(),
            Some("missing combat rule: /accumulator/attack")
        );
    }
}
