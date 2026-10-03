//! 保留 factorized-v1 的实体顺序、引用重编号和存在掩码；词表及印刷值由 TS 注入。
use crate::actions::{array, text};
use crate::model::{Catalog, Kind, State};
use crate::tree::{Node, Tree};
use serde::Serialize;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

pub fn validate_schema(s: &Value) -> Result<(), String> {
    if s["encoding"] != "haojie-entities-factorized-v1"
        || s["entity_features"] != 64
        || s["global_features"] != 32
        || s["action_features"] != 64
        || s["kind_count"] != 256
    {
        return Err("unsupported encoding schema".into());
    }
    for (name, len) in [
        ("roles", 24),
        ("commands", 23),
        ("modes", 13),
        ("phases", 5),
        ("effects", 9),
        ("reactions", 5),
        ("unit_fields", 38),
        ("decision_stages", 10),
    ] {
        let a = array(&s[name]);
        if a.len() != len || a.iter().any(|v| !v.is_string()) {
            return Err(format!("invalid encoding vocabulary {name}"));
        }
    }
    if array(&s["kinds"]).is_empty() || array(&s["kinds"]).len() >= 128 {
        return Err("invalid kind vocabulary".into());
    }
    Ok(())
}
pub fn known(v: &Value, keys: &[&str]) -> Result<(), String> {
    let map = v.as_object().ok_or("encoding expects object")?;
    for k in map.keys() {
        if !keys.contains(&k.as_str()) {
            return Err(format!("unencoded field {k}"));
        }
    }
    Ok(())
}
fn category(v: &Value, words: &[Value]) -> Result<Value, String> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    words
        .iter()
        .position(|w| w == v)
        .map(|i| json!(i + 1))
        .ok_or_else(|| format!("unknown encoding category {v}"))
}
fn cat(v: &Value, words: &[&str]) -> Result<Value, String> {
    category(v, &words.iter().map(|w| json!(w)).collect::<Vec<_>>())
}
fn num(v: &Value) -> Result<f64, String> {
    if let Some(b) = v.as_bool() {
        return Ok(if b { 1.0 } else { 0.0 });
    }
    let n = v
        .as_f64()
        .filter(|n| n.is_finite())
        .ok_or("encoding expects finite number")?;
    Ok(n.signum() * n.abs().ln_1p() / 8.0)
}
fn no_null(v: &Value) -> Result<(), String> {
    match v {
        Value::Null => Err("explicit null is not an optional encoding field".into()),
        Value::Array(a) => {
            for v in a {
                no_null(v)?;
            }
            Ok(())
        }
        Value::Object(o) => {
            for v in o.values() {
                no_null(v)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
fn js_keys(v: &Value) -> Vec<String> {
    let keys: Vec<_> = v
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    js_key_order(keys)
}
fn js_key_order(mut keys: Vec<String>) -> Vec<String> {
    let index = |k: &str| {
        k.parse::<u32>()
            .ok()
            .filter(|n| *n != u32::MAX && n.to_string() == k)
    };
    keys.sort_by(|a, b| match (index(a), index(b)) {
        (Some(a), Some(b)) => a.cmp(&b),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    });
    keys
}
#[derive(Serialize)]
pub struct Input {
    pub entities: Vec<Rc<Vec<f64>>>,
    pub kinds: Vec<usize>,
    pub globals: Vec<f64>,
    pub candidates: Vec<Vec<f64>>,
    pub entity_mask: Vec<bool>,
    pub candidate_mask: Vec<bool>,
    pub sources: Vec<i32>,
    pub targets: Vec<i32>,
}
struct Encoder<'a> {
    s: &'a State,
    catalog: &'a Catalog,
    viewer: usize,
    entities: Vec<Rc<Vec<f64>>>,
    kinds: Vec<usize>,
    indices: Cow<'a, HashMap<String, usize>>,
    identities: HashMap<String, usize>,
    base_identities: Option<&'a HashMap<String, usize>>,
}

#[derive(Default)]
struct RowMeta<'a> {
    id: Option<&'a str>,
    parent: Option<&'a str>,
    source: Option<&'a str>,
    target: Option<&'a str>,
    group: Option<&'a str>,
    owner: Option<usize>,
    kind: Option<&'a Kind>,
    order: usize,
    selectable: bool,
}
struct Row {
    values: Vec<f64>,
    kind: usize,
}
impl Row {
    fn number(&mut self, at: usize, value: Option<f64>) -> Result<(), String> {
        if let Some(v) = value {
            if !v.is_finite() {
                return Err("encoding expects finite number".into());
            }
            self.values[8 + at] = v.signum() * v.abs().ln_1p() / 8.0;
            self.present(at);
        }
        Ok(())
    }
    fn boolean(&mut self, at: usize, value: Option<bool>) {
        if let Some(v) = value {
            self.values[8 + at] = f64::from(v);
            self.present(at);
        }
    }
    fn value(&mut self, at: usize, value: &Value) -> Result<(), String> {
        match value {
            Value::Null => Ok(()),
            Value::Bool(v) => {
                self.boolean(at, Some(*v));
                Ok(())
            }
            _ => self.number(
                at,
                Some(value.as_f64().ok_or("encoding expects finite number")?),
            ),
        }
    }
    fn present(&mut self, at: usize) {
        self.values[60 + at / 13] += (1u32 << (at % 13)) as f64 / 8191.0;
    }
}
impl Encoder<'_> {
    fn reference_id(&mut self, id: Option<&str>) -> Result<usize, String> {
        let Some(id) = id else {
            return Ok(0);
        };
        if id.is_empty() {
            return Err("invalid entity reference".into());
        }
        if let Some(n) = self.base_identities.and_then(|ids| ids.get(id)) {
            return Ok(*n);
        }
        let n = self.base_identities.map_or(0, HashMap::len) + self.identities.len() + 1;
        Ok(*self.identities.entry(id.into()).or_insert(n))
    }
    fn kind_code(&self, kind: Option<&Kind>) -> Result<usize, String> {
        let Some(kind) = kind else {
            return Ok(0);
        };
        self.catalog
            .kind_codes
            .get(kind)
            .copied()
            .ok_or_else(|| format!("unknown kind {}", kind.key()))
    }
    fn category_code(&self, value: Option<&str>, vocabulary: &str) -> Result<Option<f64>, String> {
        value
            .map(|v| {
                self.catalog
                    .vocab_codes
                    .get(vocabulary)
                    .and_then(|m| m.get(v))
                    .map(|i| *i as f64)
                    .ok_or_else(|| format!("unknown encoding category {v}"))
            })
            .transpose()
    }
    fn index_id(&self, id: Option<&str>) -> Result<i32, String> {
        id.map(|id| {
            self.indices
                .get(id)
                .map(|i| *i as i32)
                .ok_or_else(|| format!("missing entity {id}"))
        })
        .unwrap_or(Ok(-1))
    }
    fn row(&mut self, role: &str, meta: RowMeta<'_>) -> Result<Row, String> {
        let role = self.catalog.vocab_codes["roles"]
            .get(role)
            .copied()
            .ok_or("unknown role")?
            - 1;
        let mut values = vec![0.0; 64];
        values[0] = (role + 1) as f64 / 32.0;
        values[1] = match meta.owner {
            None => 0.0,
            Some(p) if p == self.viewer => 1.0,
            Some(1 | 2) => -1.0,
            _ => return Err("invalid encoding owner".into()),
        };
        for (i, id) in [meta.id, meta.parent, meta.source, meta.target]
            .into_iter()
            .enumerate()
        {
            values[2 + i] = self.reference_id(id)? as f64 / 256.0;
        }
        values[6] =
            self.reference_id(meta.group.map(|g| format!("group:{g}")).as_deref())? as f64 / 256.0;
        values[7] = meta.order as f64 / 128.0;
        if meta.selectable
            && let Some(id) = meta.id
        {
            self.indices
                .to_mut()
                .entry(id.into())
                .or_insert(self.entities.len());
        }
        Ok(Row {
            values,
            kind: if meta.kind.is_some() {
                self.kind_code(meta.kind)?
            } else {
                128 + role
            },
        })
    }
    fn push(&mut self, row: Row) {
        self.entities.push(Rc::new(row.values));
        self.kinds.push(row.kind);
    }
    fn vocab(&self, name: &str) -> &[Value] {
        array(&self.catalog.encoding[name])
    }
    fn kind(&self, k: &Value) -> Result<usize, String> {
        if k.is_null() {
            return Ok(0);
        }
        self.vocab("kinds")
            .iter()
            .position(|v| v == k)
            .map(|i| i + 1)
            .ok_or_else(|| format!("unknown kind {k}"))
    }
    fn kind_from_key(&self, key: &str) -> Result<Value, String> {
        self.vocab("kinds")
            .iter()
            .find(|v| {
                if v.is_string() {
                    text(v) == key
                } else {
                    crate::preparation::kind(v).key() == key
                }
            })
            .cloned()
            .ok_or_else(|| format!("unknown kind key {key}"))
    }
    fn owner(&self, v: &Value) -> Result<f64, String> {
        if v.is_null() {
            return Ok(0.0);
        }
        match v.as_u64() {
            Some(p) if p == self.viewer as u64 => Ok(1.0),
            Some(1 | 2) => Ok(-1.0),
            _ => Err("invalid encoding owner".into()),
        }
    }
    fn reference(&mut self, v: &Value) -> Result<usize, String> {
        if v.is_null() {
            return Ok(0);
        }
        let id = v
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("invalid entity reference")?;
        if let Some(n) = self.base_identities.and_then(|ids| ids.get(id)) {
            return Ok(*n);
        }
        let n = self.base_identities.map_or(0, HashMap::len) + self.identities.len() + 1;
        Ok(*self.identities.entry(id.into()).or_insert(n))
    }
    fn add(&mut self, role: &str, o: Value, fields: Vec<Value>) -> Result<(), String> {
        let role = self
            .vocab("roles")
            .iter()
            .position(|v| v == role)
            .ok_or("unknown role")?;
        if fields.len() > 52 {
            return Err("too many entity fields".into());
        }
        let mut row = vec![0.0; 64];
        row[0] = (role + 1) as f64 / 32.0;
        row[1] = self.owner(&o["owner"])?;
        for (i, key) in ["id", "parent", "source", "target"].iter().enumerate() {
            row[2 + i] = self.reference(&o[*key])? as f64 / 256.0;
        }
        row[6] = self.reference(&if o["group"].is_null() {
            Value::Null
        } else {
            json!(format!("group:{}", text(&o["group"])))
        })? as f64
            / 256.0;
        row[7] = o["order"].as_f64().unwrap_or(0.0) / 128.0;
        for (i, v) in fields.iter().enumerate() {
            if !v.is_null() {
                row[8 + i] = num(v)?;
                row[60 + i / 13] += (1u32 << (i % 13)) as f64 / 8191.0;
            }
        }
        if o["selectable"] == true
            && let Some(id) = o["id"].as_str()
        {
            let n = self.entities.len();
            self.indices.to_mut().entry(id.into()).or_insert(n);
        }
        self.kinds.push(if o["kind"].is_null() {
            128 + role
        } else {
            self.kind(&o["kind"])?
        });
        self.entities.push(Rc::new(row));
        Ok(())
    }
    fn printed(&self, k: &Value) -> Result<Vec<Value>, String> {
        let kind = crate::preparation::kind(k);
        let d = self.catalog.get_kind(&kind).ok_or("missing printed kind")?;
        Ok(vec![
            json!(d.attack),
            json!(d.health),
            json!(d.range),
            json!(d.actions),
            json!(d.movement),
            json!(d.size.unwrap_or(1.0)),
            json!(d.printed_mage),
            json!(d.spell),
            json!(d.weapon),
            json!(d.printed_aura),
        ])
    }
    fn effect(&mut self, e: &crate::model::Effect, parent: &str, i: usize) -> Result<(), String> {
        let mut row = self.row(
            "effect",
            RowMeta {
                owner: Some(e.owner),
                parent: Some(parent),
                source: e.source_id.as_deref(),
                order: i,
                ..RowMeta::default()
            },
        )?;
        row.number(0, self.category_code(Some(&e.kind), "effects")?)?;
        row.number(1, Some(e.from))?;
        row.number(2, Some(e.until))?;
        row.number(3, e.amount)?;
        row.boolean(4, e.global);
        self.push(row);
        Ok(())
    }
    fn unit(
        &mut self,
        u: &crate::model::Unit,
        role: &str,
        parent: Option<&str>,
        order: usize,
    ) -> Result<(), String> {
        for key in u.extra.keys() {
            if !self.vocab("unit_fields").iter().any(|v| v == key)
                && ![
                    "attacked",
                    "guardSourceIds",
                    "traits",
                    "abilityUsage",
                    "abilityCharges",
                    "equipmentIds",
                    "receivedDamage",
                    "group",
                ]
                .contains(&key.as_str())
            {
                return Err(format!("unencoded field {key}"));
            }
        }
        let extra = |key: &str| u.extra.get(key).unwrap_or(&Value::Null);
        let id = if role == "snapshot" {
            format!("{}:snapshot:{order}", parent.unwrap())
        } else {
            u.id.clone()
        };
        self.indices
            .to_mut()
            .entry(u.id.clone())
            .or_insert(self.entities.len());
        let mut row = self.row(
            role,
            RowMeta {
                id: Some(&id),
                source: (role == "snapshot").then_some(u.id.as_str()),
                kind: Some(&u.kind),
                owner: Some(u.owner),
                parent,
                group: extra("group").as_str(),
                order,
                ..RowMeta::default()
            },
        )?;
        for (i, n) in [
            (0, u.x),
            (1, u.y),
            (2, u.hp),
            (3, u.max_hp),
            (4, u.size),
            (5, u.born),
            (6, u.offset),
            (10, u.operations),
            (11, u.shots),
            (12, u.moves),
            (13, u.bonus_attacks),
        ] {
            row.number(i, Some(n))?;
        }
        row.number(9, self.category_code(Some(&u.mode), "modes")?)?;
        row.boolean(14, Some(u.bonus_sequence));
        row.boolean(26, Some(u.silenced));
        for (i, n) in [
            (7, u.deployed_at),
            (20, u.upgrades),
            (21, u.kills),
            (22, u.attack_bonus),
            (23, u.range_bonus),
            (27, u.free_used),
        ] {
            row.number(i, Some(n))?;
        }
        for (i, b) in [
            (8, u.charged_on_deploy),
            (15, u.weapon_first_used),
            (24, u.guard_used),
            (28, u.once_used),
        ] {
            row.boolean(i, Some(b));
        }
        for (i, key) in [
            (25, "rerollUsedPly"),
            (29, "extraOperations"),
            (30, "bannerHp"),
            (31, "overMaxFromBanner"),
            (32, "bladeQualified"),
            (33, "expiresAt"),
            (34, "hookReadyAt"),
            (35, "hookExpiresAt"),
            (36, "dormantSince"),
            (37, "rebuildTicks"),
        ] {
            row.value(i, extra(key))?;
        }
        row.number(16, Some(u.reserve.charge))?;
        row.number(17, Some(u.reserve.ready_charge))?;
        row.number(
            18,
            self.category_code(Some(u.reserve.charge_type.as_str()), "modes")?,
        )?;
        row.number(19, Some(u.reserve.last_charge))?;
        if role != "snapshot" {
            let st = crate::stats::stats(self.s, u, self.catalog);
            for (i, n) in [
                st.attack,
                st.range,
                st.actions,
                st.remaining,
                st.movement,
                st.operation_limit,
                st.operations_left,
            ]
            .into_iter()
            .enumerate()
            {
                row.number(38 + i, Some(n))?;
            }
            row.boolean(45, Some(st.sleeping));
            row.boolean(46, Some(st.frozen));
            row.boolean(47, Some(st.stunned));
        }
        self.push(row);
        for (i, e) in u.effects.iter().enumerate() {
            self.effect(e, &id, i)?;
        }
        for (i, t) in u.attacked.iter().enumerate() {
            let row = self.row(
                "attacked",
                RowMeta {
                    parent: Some(&id),
                    target: Some(t),
                    owner: Some(u.owner),
                    order: i,
                    ..RowMeta::default()
                },
            )?;
            self.push(row);
        }
        for (role, key, source) in [("guard", "guardSourceIds", true)] {
            for (i, t) in array(extra(key)).iter().enumerate() {
                let row = self.row(
                    role,
                    RowMeta {
                        parent: Some(&id),
                        target: if source { None } else { t.as_str() },
                        source: if source { t.as_str() } else { None },
                        owner: Some(u.owner),
                        order: i,
                        ..RowMeta::default()
                    },
                )?;
                self.push(row);
            }
        }
        for (i, h) in array(extra("receivedDamage")).iter().enumerate() {
            known(h, &["ply", "amount"])?;
            let mut row = self.row(
                "received",
                RowMeta {
                    parent: Some(&id),
                    owner: Some(u.owner),
                    order: i,
                    ..RowMeta::default()
                },
            )?;
            row.value(0, &h["ply"])?;
            row.value(1, &h["amount"])?;
            self.push(row);
        }
        let mut equipment = u.equipment.clone();
        for key in js_keys(extra("equipmentIds")) {
            let k = crate::preparation::kind(&self.kind_from_key(&key)?);
            if !equipment.contains(&k) {
                equipment.push(k);
            }
        }
        for (i, k) in equipment.iter().enumerate() {
            let key = k.key();
            let mut row = self.row(
                "equipment",
                RowMeta {
                    parent: Some(&id),
                    kind: Some(k),
                    owner: Some(u.owner),
                    id: extra("equipmentIds")[key].as_str(),
                    order: i,
                    ..RowMeta::default()
                },
            )?;
            row.boolean(0, Some(u.equipment.contains(k)));
            self.printed_into(&mut row, 1, k)?;
            self.push(row);
        }
        let mut abilities: Vec<Kind> = u.traits.clone().unwrap_or_default();
        let keys = js_key_order(
            u.ability_usage
                .as_ref()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default(),
        )
        .into_iter()
        .chain(js_key_order(
            u.ability_charges
                .as_ref()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default(),
        ));
        for key in keys {
            let k = crate::preparation::kind(&self.kind_from_key(&key)?);
            if !abilities.contains(&k) {
                abilities.push(k);
            }
        }
        for k in abilities {
            let key = k.key();
            let usage = u.ability_usage.as_ref().and_then(|m| m.get(&key));
            let charge = u.ability_charges.as_ref().and_then(|m| m.get(&key));
            let mut row = self.row(
                "ability",
                RowMeta {
                    parent: Some(&id),
                    kind: Some(&k),
                    owner: Some(u.owner),
                    ..RowMeta::default()
                },
            )?;
            row.boolean(0, Some(u.traits.as_ref().is_some_and(|v| v.contains(&k))));
            row.boolean(1, usage.map(|u| u.once));
            row.number(2, usage.map(|u| u.free))?;
            row.number(3, charge.map(|r| r.charge))?;
            row.number(4, charge.map(|r| r.ready_charge))?;
            row.number(
                5,
                self.category_code(charge.map(|r| r.charge_type.as_str()), "modes")?,
            )?;
            row.number(6, charge.map(|r| r.last_charge))?;
            self.push(row);
        }
        Ok(())
    }
    fn printed_into(&self, row: &mut Row, offset: usize, k: &Kind) -> Result<(), String> {
        let d = self.catalog.by_kind(k);
        for (i, n) in [
            d.attack,
            d.health,
            d.range,
            d.actions,
            d.movement,
            d.size.unwrap_or(1.0),
        ]
        .into_iter()
        .enumerate()
        {
            row.number(offset + i, Some(n))?;
        }
        // 印刷布尔字段的缺失与 false 不等价，直接读取规则包保留的存在性。
        row.boolean(offset + 6, d.printed_mage);
        row.number(offset + 7, d.spell)?;
        row.number(offset + 8, d.weapon)?;
        row.boolean(offset + 9, d.printed_aura);
        Ok(())
    }
    fn card(
        &mut self,
        c: &Value,
        owner: &Value,
        role: &str,
        parent: Option<String>,
        order: usize,
    ) -> Result<(), String> {
        known(
            c,
            &[
                "id",
                "kind",
                "drawnAt",
                "expiresAt",
                "group",
                "rerolled",
                "summonedPly",
                "summonPool",
                "parity",
            ],
        )?;
        let mut fields = vec![
            c["drawnAt"].clone(),
            c["expiresAt"].clone(),
            c["rerolled"].clone(),
            c["summonedPly"].clone(),
            cat(&c["summonPool"], &["normal", "ultimate"])?,
            cat(&c["parity"], &["odd", "even"])?,
        ];
        fields.extend(self.printed(&c["kind"])?);
        self.add(role,json!({"id":c["id"],"kind":c["kind"],"owner":owner,"parent":parent,"group":c["group"],"order":order,"selectable":true}),fields)
    }
}

/// 只保存公开观察的固定编码，不保存前缀派生身份，也不跨树复用。
pub struct BaseEncoding {
    entities: Vec<Rc<Vec<f64>>>,
    kinds: Vec<usize>,
    indices: HashMap<String, usize>,
    identities: HashMap<String, usize>,
    globals: Vec<f64>,
}

/// 只接收公开 Observation；严格拒绝新字段与未共同揭示的对手暗选，不截断任何实体。
fn encode_base(tree: &Tree<'_>) -> Result<BaseEncoding, String> {
    let viewer = tree.actor;
    let catalog = tree.catalog;
    validate_schema(&catalog.encoding)?;
    // 树的状态已从同一公开观察构造；像 TS 一样复用只读实体，避免每个参数节点重建整局。
    let s = &tree.state;
    let v = |key: &str| s.extra.get(key).unwrap_or(&Value::Null);
    if ![1, 2].contains(&viewer) || s.version != 2 {
        return Err("invalid encoding viewer/version".into());
    }
    if let Some(observation) = tree.observation {
        no_null(observation)?;
        known(
            observation,
            &[
                "version",
                "serial",
                "ply",
                "active",
                "phase",
                "mode",
                "landmarks",
                "auras",
                "shrineDraft",
                "shrineSetupDone",
                "regularSummons",
                "summonOffer",
                "clockFrames",
                "summonSlots",
                "turns",
                "bases",
                "baseEffects",
                "heads",
                "hands",
                "bonus",
                "deployRows",
                "units",
                "pending",
                "deaths",
                "hazards",
                "siphons",
                "iceMarks",
                "winner",
            ],
        )?;
    }
    let mut e = Encoder {
        s,
        catalog,
        viewer,
        entities: vec![],
        kinds: vec![],
        indices: Cow::Owned(HashMap::new()),
        identities: HashMap::new(),
        base_identities: None,
    };
    let sides = [viewer, 3 - viewer];
    let sidekeys = [viewer.to_string(), (3 - viewer).to_string()];
    for p in sides {
        e.reference(&json!(format!("base-{p}")))?;
    }
    for u in s.units.iter().chain(s.landmarks()) {
        e.reference_id(Some(&u.id))?;
    }
    for p in &sidekeys {
        for c in array(&v("hands")[p]) {
            e.reference(&c["id"])?;
        }
    }
    for d in array(v("deaths")) {
        e.reference(&d["id"])?;
    }
    for (p, key) in sides.iter().zip(&sidekeys) {
        for pair in ["heads", "hands", "bonus"] {
            known(v(pair), &["1", "2"])?;
        }
        known(&s.bases, &["1", "2"])?;
        known(&s.deploy_rows, &["1", "2"])?;
        let at = crate::geometry::base_point(*p);
        let id = format!("base-{p}");
        let mut row = e.row(
            "base",
            RowMeta {
                id: Some(&id),
                owner: Some(*p),
                selectable: true,
                ..RowMeta::default()
            },
        )?;
        row.number(0, Some(at.x))?;
        row.number(1, Some(at.y))?;
        row.value(2, &s.bases[key])?;
        row.number(3, Some(s.turns[key]))?;
        row.value(4, &v("heads")[key])?;
        row.value(5, &v("bonus")[key])?;
        for i in 1..=13 {
            row.boolean(
                5 + i,
                Some(
                    array(&s.deploy_rows[key])
                        .iter()
                        .any(|n| n.as_u64() == Some(i as u64)),
                ),
            );
        }
        e.push(row);
        if s.base_effects.keys().any(|k| k != "1" && k != "2") {
            return Err("unknown base effect side".into());
        }
        for (i, eff) in s
            .base_effects
            .get(key)
            .ok_or("missing base effect side")?
            .iter()
            .enumerate()
        {
            e.effect(eff, &id, i)?;
        }
    }
    for (i, u) in s.units.iter().enumerate() {
        e.unit(u, "unit", None, i)?;
    }
    for (i, u) in s.landmarks().iter().enumerate() {
        e.unit(u, "landmark", None, i)?;
    }
    for (p, key) in sides.iter().zip(&sidekeys) {
        for (i, c) in array(&v("hands")[key]).iter().enumerate() {
            e.card(c, &json!(p), "card", None, i)?;
        }
    }
    for (i, d) in array(v("deaths")).iter().enumerate() {
        known(d, &["id", "kind", "owner", "ply", "revived", "group"])?;
        let mut fields = vec![d["ply"].clone(), d["revived"].clone()];
        fields.extend(e.printed(&d["kind"])?);
        e.add(
            "death",
            crate::actions::extend(d, json!({"order":i,"selectable":true})),
            fields,
        )?;
    }
    for (i, h) in array(v("hazards")).iter().enumerate() {
        known(h, &["id", "owner", "sourceId", "axis", "line", "due"])?;
        e.add(
            "hazard",
            json!({"id":h["id"],"owner":h["owner"],"source":h["sourceId"],"order":i}),
            vec![
                cat(&h["axis"], &["row", "column"])?,
                h["line"].clone(),
                h["due"].clone(),
            ],
        )?;
    }
    for (i, l) in s.siphons.iter().enumerate() {
        known(l, &["id", "sourceId", "owner", "fromId", "toId"])?;
        e.add("siphon",json!({"id":l["id"],"owner":l["owner"],"parent":l["fromId"],"source":l["sourceId"],"target":l["toId"],"order":i}),vec![])?;
    }
    for (i, m) in array(v("iceMarks")).iter().enumerate() {
        known(m, &["id", "sourceId", "owner", "due", "x", "y"])?;
        e.add(
            "ice",
            json!({"id":m["id"],"owner":m["owner"],"source":m["sourceId"],"order":i}),
            vec![m["x"].clone(), m["y"].clone(), m["due"].clone()],
        )?;
    }
    for (i, r) in s.pending.iter().enumerate() {
        let id = format!("reaction:{i}");
        let mut row = e.row(
            "reaction",
            RowMeta {
                id: Some(&id),
                owner: Some(r.owner),
                source: Some(&r.source.id),
                target: r.target_id.as_deref(),
                order: i,
                ..RowMeta::default()
            },
        )?;
        row.number(0, e.category_code(Some(&r.kind), "reactions")?)?;
        row.number(1, Some(r.amount))?;
        e.push(row);
        e.unit(&r.source, "snapshot", Some(&id), 0)?;
    }
    for pair in ["auras"] {
        if let Some(v) = s.extra.get(pair) {
            known(v, &["1", "2"])?;
        }
    }
    for (p, key) in sides.iter().zip(&sidekeys) {
        for (i, a) in array(&v("auras")[key]).iter().enumerate() {
            known(a, &["kind", "parity", "usedPly"])?;
            e.add(
                "aura",
                json!({"kind":a["kind"],"owner":p,"order":i}),
                vec![cat(&a["parity"], &["odd", "even"])?, a["usedPly"].clone()],
            )?;
        }
        if let Some(all) = &s.clock_frames {
            if all.keys().any(|k| k != "1" && k != "2") {
                return Err("unknown clock frame side".into());
            }
            if let Some(frames) = all.get(key) {
                for (i, (name, frame)) in
                    [("current", &frames.current), ("previous", &frames.previous)]
                        .into_iter()
                        .enumerate()
                {
                    if let Some(f) = frame {
                        if f.turns.keys().any(|k| k != "1" && k != "2") {
                            return Err("unknown clock turn side".into());
                        }
                        let id = format!("frame:{p}:{name}");
                        let mut row = e.row(
                            "frame",
                            RowMeta {
                                id: Some(&id),
                                owner: Some(*p),
                                order: i,
                                ..RowMeta::default()
                            },
                        )?;
                        row.number(0, Some(f.ply))?;
                        row.number(1, f.turns.get(&sidekeys[0]).copied())?;
                        row.number(2, f.turns.get(&sidekeys[1]).copied())?;
                        e.push(row);
                        for (j, u) in f.units.iter().enumerate() {
                            e.unit(u, "snapshot", Some(&id), j)?;
                        }
                    }
                }
            }
        }
    }
    let draft = v("shrineDraft");
    if !draft.is_null() {
        known(draft, &["offers", "committed", "choices", "revealed"])?;
        for key in ["offers", "committed", "choices"] {
            known(&draft[key], &["1", "2"])?;
        }
        if draft["revealed"] != true && draft["choices"].get(&sidekeys[1]).is_some() {
            return Err("opponent unrevealed shrine choice leaked".into());
        }
        for (p, key) in sides.iter().zip(&sidekeys) {
            for (i, k) in array(&draft["offers"][key]).iter().enumerate() {
                e.add("draft-offer", json!({"kind":k,"owner":p,"order":i}), vec![])?;
            }
            if let Some(c) = draft["choices"].get(key) {
                known(c, &["kind", "parity"])?;
                e.add(
                    "draft-choice",
                    json!({"kind":c["kind"],"owner":p}),
                    vec![cat(&c["parity"], &["odd", "even"])?],
                )?;
            }
        }
    }
    if let Some(offer) = s.extra.get("summonOffer") {
        known(offer, &["owner", "groups", "count"])?;
        for (i, g) in array(&offer["groups"]).iter().enumerate() {
            for (j, c) in array(g).iter().enumerate() {
                e.card(
                    c,
                    &offer["owner"],
                    "summon-offer",
                    Some(format!("offer:{i}")),
                    j,
                )?;
            }
        }
    }
    let mut globals = vec![
        viewer as f64 / 2.0,
        if s.active == viewer { 1.0 } else { -1.0 },
    ];
    globals.extend(e.vocab("phases").iter().map(|p| {
        if p.as_str() == Some(s.phase.as_str()) {
            1.0
        } else {
            0.0
        }
    }));
    globals.push(if v("mode") == "shrine" { 1.0 } else { 0.0 });
    for value in [
        s.ply,
        s.summon_slots,
        s.turns[&sidekeys[0]],
        s.turns[&sidekeys[1]],
        crate::model::number(&s.bases[&sidekeys[0]]),
        crate::model::number(&s.bases[&sidekeys[1]]),
        crate::model::number(&v("heads")[&sidekeys[0]]),
        crate::model::number(&v("heads")[&sidekeys[1]]),
        crate::model::number(&v("bonus")[&sidekeys[0]]),
        crate::model::number(&v("bonus")[&sidekeys[1]]),
        crate::model::number(v("regularSummons")),
    ] {
        globals.push(value.signum() * value.abs().ln_1p() / 8.0);
    }
    globals.extend([
        if v("winner").is_null() || v("winner") == "draw" {
            0.0
        } else {
            e.owner(v("winner"))?
        },
        if s.extra.get("winner").is_some() {
            1.0
        } else {
            0.0
        },
        (s.pending.len() as f64).ln_1p() / 8.0,
        num(&json!(array(&v("hands")[&sidekeys[0]]).len()))?,
        num(&json!(array(&v("hands")[&sidekeys[1]]).len()))?,
    ]);
    for p in sides {
        globals.push(if array(v("shrineSetupDone")).contains(&json!(p)) {
            1.0
        } else {
            0.0
        });
    }
    for key in &sidekeys {
        globals.push(if draft["committed"][key] == true {
            1.0
        } else {
            0.0
        });
    }
    globals.extend([
        if draft["revealed"] == true { 1.0 } else { 0.0 },
        num(v("summonOffer").get("count").unwrap_or(&json!(0)))?,
        e.owner(&v("summonOffer")["owner"])?,
        0.0,
    ]);
    Ok(BaseEncoding {
        entities: e.entities,
        kinds: e.kinds,
        indices: e.indices.into_owned(),
        identities: e.identities,
        globals,
    })
}

/// 默认返回独立行；内部同步采样只借用固定行和索引，前缀身份使用追加映射。
pub fn encode(tree: &Tree<'_>, node: &Node) -> Result<Input, String> {
    encode_mode(tree, node, false)
}
pub fn encode_sampling(tree: &Tree<'_>, node: &Node) -> Result<Input, String> {
    encode_mode(tree, node, true)
}
fn encode_mode(tree: &Tree<'_>, node: &Node, borrow: bool) -> Result<Input, String> {
    #[cfg(feature = "kernel-profile")]
    let _profile = crate::profile::scope(crate::profile::Phase::Encoding);
    if node.choices.is_empty() {
        return Err("empty action branch requires backtracking".into());
    }
    let base = tree
        .encoding
        .get_or_init(|| encode_base(tree))
        .as_ref()
        .map_err(Clone::clone)?;
    let viewer = tree.actor;
    let catalog = tree.catalog;
    let mut e = Encoder {
        s: &tree.state,
        catalog,
        viewer,
        entities: if borrow {
            base.entities.clone()
        } else {
            base.entities
                .iter()
                .map(|r| Rc::new((**r).clone()))
                .collect()
        },
        kinds: base.kinds.clone(),
        indices: if borrow {
            Cow::Borrowed(&base.indices)
        } else {
            Cow::Owned(base.indices.clone())
        },
        identities: if borrow {
            HashMap::new()
        } else {
            base.identities.clone()
        },
        base_identities: borrow.then_some(&base.identities),
    };
    // 命令直接写固定数值缓冲；不构造 JSON 字段数组或再次解析内部命令。
    let direction = |v: Option<&str>| -> Result<Option<f64>, String> {
        v.map(|v| {
            ["up", "down", "left", "right"]
                .iter()
                .position(|w| *w == v)
                .map(|i| (i + 1) as f64)
                .ok_or_else(|| "unknown direction".to_string())
        })
        .transpose()
    };
    let parity = |v: Option<&str>| -> Result<Option<f64>, String> {
        match v {
            None => Ok(None),
            Some("odd") => Ok(Some(1.0)),
            Some("even") => Ok(Some(2.0)),
            _ => Err("unknown parity".into()),
        }
    };
    let recipe = |v: Option<&str>| -> Result<Option<f64>, String> {
        v.map(|v| {
            catalog
                .recipes
                .iter()
                .position(|r| r["id"] == v)
                .map(|i| (i + 1) as f64)
                .ok_or_else(|| "unknown recipe".to_string())
        })
        .transpose()
    };
    if let Some(c) = &node.prefix {
        let mut row = e.row(
            "prefix",
            RowMeta {
                id: Some("prefix"),
                owner: Some(viewer),
                source: c.unit_id.as_deref().or(c.card_id.as_deref()),
                target: c.target_id.as_deref(),
                ..RowMeta::default()
            },
        )?;
        row.number(0, e.category_code(Some(&c.kind), "commands")?)?;
        row.number(1, e.category_code(c.mode.as_deref(), "modes")?)?;
        row.number(2, c.x)?;
        row.number(3, c.y)?;
        row.number(4, c.row)?;
        row.number(5, c.column)?;
        row.boolean(6, c.ultimate);
        row.boolean(7, c.charge);
        row.number(8, direction(c.direction.as_deref())?)?;
        for (i, k) in [
            c.ability.as_ref(),
            c.chosen_kind.as_ref(),
            c.shrine_kind.as_ref(),
        ]
        .into_iter()
        .enumerate()
        {
            row.number(
                9 + i,
                k.map(|k| e.kind_code(Some(k)).map(|n| n as f64))
                    .transpose()?,
            )?;
        }
        row.number(12, parity(c.parity.as_deref())?)?;
        row.number(13, recipe(c.recipe_id.as_deref())?)?;
        row.number(14, c.player.map(|p| if p == viewer { 1.0 } else { -1.0 }))?;
        e.push(row);
        let mut argument = |id: &str, field: usize, order: usize| -> Result<(), String> {
            let mut row = e.row(
                "argument",
                RowMeta {
                    parent: Some("prefix"),
                    target: Some(id),
                    order,
                    ..RowMeta::default()
                },
            )?;
            row.number(0, Some(field as f64))?;
            e.push(row);
            Ok(())
        };
        if let Some(id) = &c.second_id {
            argument(id, 1, 0)?;
        }
        if let Some(id) = &c.death_id {
            argument(id, 2, 0)?;
        }
        for (field, ids) in [
            (3, &c.material_ids),
            (4, &c.card_ids),
            (5, &c.sacrifice_ids),
        ] {
            for (i, id) in ids.as_deref().unwrap_or(&[]).iter().enumerate() {
                argument(id, field, i)?;
            }
        }
        for (i, p) in c.path.as_deref().unwrap_or(&[]).iter().enumerate() {
            let mut row = e.row(
                "path",
                RowMeta {
                    parent: Some("prefix"),
                    order: i,
                    ..RowMeta::default()
                },
            )?;
            row.number(0, Some(p.x))?;
            row.number(1, Some(p.y))?;
            e.push(row);
        }
        for (i, n) in c.offer_indices.as_deref().unwrap_or(&[]).iter().enumerate() {
            let mut row = e.row(
                "argument",
                RowMeta {
                    parent: Some("prefix"),
                    order: i,
                    ..RowMeta::default()
                },
            )?;
            row.number(0, Some(6.0))?;
            row.number(1, Some(*n))?;
            e.push(row);
        }
    }
    let mut globals = base.globals.clone();
    globals[31] = f64::from(node.prefix.is_some());
    let mut candidates = Vec::with_capacity(node.choices.len());
    let mut sources = Vec::with_capacity(node.choices.len());
    let mut targets = Vec::with_capacity(node.choices.len());
    let stage = e
        .category_code(Some(node.stage), "decision_stages")?
        .unwrap()
        / 16.0;
    for choice in &node.choices {
        let c = &choice.command;
        let mut row = vec![0.0; 64];
        row[e.category_code(Some(&c.kind), "commands")?.unwrap() as usize - 1] = 1.0;
        if let Some(n) = e.category_code(c.mode.as_deref(), "modes")? {
            row[23 + n as usize - 1] = 1.0;
        }
        row[36] = c.x.unwrap_or(0.0) / 9.0;
        row[37] = c.y.unwrap_or(0.0) / 13.0;
        row[38] = c.row.unwrap_or(0.0) / 13.0;
        row[39] = c.column.unwrap_or(0.0) / 9.0;
        row[40] = c.ultimate.map_or(-1.0, f64::from);
        row[41] = c.charge.map_or(-1.0, f64::from);
        row[42] = direction(c.direction.as_deref())?.unwrap_or(0.0) / 4.0;
        row[43] = e.kind_code(c.ability.as_ref())? as f64 / 128.0;
        row[44] = e.kind_code(c.chosen_kind.as_ref())? as f64 / 128.0;
        row[45] = e.kind_code(c.shrine_kind.as_ref())? as f64 / 128.0;
        row[46] = recipe(c.recipe_id.as_deref())?.unwrap_or(0.0) / 8.0;
        row[47] = match parity(c.parity.as_deref())? {
            Some(1.0) => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
        row[48] = stage;
        row[49] = f64::from(choice.status != "parameter");
        row[50] = f64::from(choice.status == "uncertain");
        for i in 0..2 {
            row[51 + i] = c
                .offer_indices
                .as_deref()
                .unwrap_or(&[])
                .get(i)
                .map_or(-1.0, |n| n / 16.0);
        }
        row[53] = c.material_ids.as_deref().unwrap_or(&[]).len() as f64 / 3.0;
        row[54] = c.sacrifice_ids.as_deref().unwrap_or(&[]).len() as f64 / 2.0;
        row[55] = c.path.as_deref().unwrap_or(&[]).len() as f64 / 117.0;
        row[56] = e.reference_id(c.second_id.as_deref())? as f64 / 256.0;
        row[57] = e.reference_id(c.death_id.as_deref())? as f64 / 256.0;
        row[58] = f64::from(choice.key == "commit-path");
        row[59] = c
            .player
            .map_or(0.0, |p| if p == viewer { 1.0 } else { -1.0 });
        if let Some(p) = c.path.as_deref().unwrap_or(&[]).last() {
            row[60] = p.x / 9.0;
            row[61] = p.y / 13.0;
        }
        let source = c.unit_id.as_deref().or(c.card_id.as_deref()).or_else(|| {
            if c.kind == "react" {
                tree.state.pending.first().map(|r| r.source.id.as_str())
            } else {
                None
            }
        });
        sources.push(e.index_id(source)?);
        targets.push(
            e.index_id(
                choice
                    .subject
                    .as_deref()
                    .or(c.target_id.as_deref())
                    .or(c.death_id.as_deref()),
            )?,
        );
        candidates.push(row);
    }
    Ok(Input {
        entity_mask: vec![true; e.entities.len()],
        candidate_mask: vec![true; candidates.len()],
        entities: e.entities,
        kinds: e.kinds,
        globals,
        candidates,
        sources,
        targets,
    })
}
