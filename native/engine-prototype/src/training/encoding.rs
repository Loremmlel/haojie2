//! 保留 factorized-v1 的实体顺序、引用重编号和存在掩码；词表及印刷值由 TS 注入。
use crate::actions::{array, text};
use crate::model::{Catalog, State};
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
    let mut keys: Vec<_> = v
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
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
impl Encoder<'_> {
    fn vocab(&self, name: &str) -> &[Value] {
        array(&self.catalog.encoding[name])
    }
    fn cat(&self, v: &Value, name: &str) -> Result<Value, String> {
        category(v, self.vocab(name))
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
    fn index(&self, v: &Value) -> Result<i32, String> {
        if v.is_null() {
            return Ok(-1);
        }
        self.indices
            .get(text(v))
            .map(|i| *i as i32)
            .ok_or_else(|| format!("missing entity {v}"))
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
        let d = self
            .catalog
            .printed
            .iter()
            .find(|v| v["id"] == *k)
            .ok_or("missing printed kind")?;
        Ok([
            "attack", "health", "range", "actions", "move", "size", "mage", "spell", "weapon",
            "aura",
        ]
        .iter()
        .map(|name| {
            if *name == "size" {
                d.get(*name).cloned().unwrap_or(json!(1))
            } else {
                d[*name].clone()
            }
        })
        .collect())
    }
    fn effect(&mut self, e: &Value, parent: &str, i: usize) -> Result<(), String> {
        known(
            e,
            &[
                "type", "from", "until", "owner", "amount", "sourceId", "global",
            ],
        )?;
        self.add(
            "effect",
            json!({"owner":e["owner"],"parent":parent,"source":e["sourceId"],"order":i}),
            vec![
                self.cat(&e["type"], "effects")?,
                e["from"].clone(),
                e["until"].clone(),
                e["amount"].clone(),
                e["global"].clone(),
            ],
        )
    }
    fn unit(
        &mut self,
        u: &Value,
        role: &str,
        parent: Option<&str>,
        order: usize,
    ) -> Result<(), String> {
        let mut keys: Vec<&str> = self.vocab("unit_fields").iter().map(text).collect();
        keys.extend([
            "id",
            "kind",
            "owner",
            "attacked",
            "guardSourceIds",
            "effects",
            "equipment",
            "traits",
            "abilityUsage",
            "abilityCharges",
            "equipmentIds",
            "receivedDamage",
            "group",
        ]);
        known(u, &keys)?;
        let mut fields = self
            .vocab("unit_fields")
            .iter()
            .map(|name| {
                let key = text(name);
                if ["mode", "chargeType"].contains(&key) {
                    self.cat(&u[key], "modes")
                } else {
                    Ok(u[key].clone())
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        if role != "snapshot" {
            let unit = if role == "unit" {
                self.s.units.get(order)
            } else {
                self.s.landmarks().get(order)
            }
            .ok_or("entity index mismatch")?;
            let st = crate::stats::stats(self.s, unit, self.catalog);
            fields.extend([
                json!(st.attack),
                json!(st.range),
                json!(st.actions),
                json!(st.remaining),
                json!(st.movement),
                json!(st.operation_limit),
                json!(st.operations_left),
                json!(st.sleeping),
                json!(st.frozen),
                json!(st.stunned),
            ]);
        }
        let id = if role == "snapshot" {
            format!("{}:snapshot:{order}", parent.unwrap())
        } else {
            text(&u["id"]).into()
        };
        self.indices
            .to_mut()
            .entry(text(&u["id"]).into())
            .or_insert(self.entities.len());
        self.add(role,json!({"id":id,"source":if role=="snapshot"{u["id"].clone()}else{Value::Null},"kind":u["kind"],"owner":u["owner"],"parent":parent,"group":u["group"],"order":order}),fields)?;
        for (i, e) in array(&u["effects"]).iter().enumerate() {
            self.effect(e, &id, i)?;
        }
        for (i, t) in array(&u["attacked"]).iter().enumerate() {
            self.add(
                "attacked",
                json!({"parent":id,"target":t,"owner":u["owner"],"order":i}),
                vec![],
            )?;
        }
        for (i, t) in array(&u["guardSourceIds"]).iter().enumerate() {
            self.add(
                "guard",
                json!({"parent":id,"source":t,"owner":u["owner"],"order":i}),
                vec![],
            )?;
        }
        for (i, h) in array(&u["receivedDamage"]).iter().enumerate() {
            known(h, &["ply", "amount"])?;
            self.add(
                "received",
                json!({"parent":id,"owner":u["owner"],"order":i}),
                vec![h["ply"].clone(), h["amount"].clone()],
            )?;
        }
        let mut equipment = array(&u["equipment"]).to_vec();
        for key in js_keys(&u["equipmentIds"]) {
            let k = self.kind_from_key(&key)?;
            if !equipment.contains(&k) {
                equipment.push(k);
            }
        }
        for (i, k) in equipment.iter().enumerate() {
            let key = crate::preparation::kind(k).key();
            let mut fields = vec![json!(array(&u["equipment"]).contains(k))];
            fields.extend(self.printed(k)?);
            self.add("equipment",json!({"parent":id,"kind":k,"owner":u["owner"],"id":u["equipmentIds"][key],"order":i}),fields)?;
        }
        let mut abilities = array(&u["traits"]).to_vec();
        for field in ["abilityUsage", "abilityCharges"] {
            for key in js_keys(&u[field]) {
                let k = self.kind_from_key(&key)?;
                if !abilities.contains(&k) {
                    abilities.push(k);
                }
            }
        }
        for k in abilities {
            let key = crate::preparation::kind(&k).key();
            let usage = &u["abilityUsage"][&key];
            let charge = &u["abilityCharges"][&key];
            if !usage.is_null() {
                known(usage, &["once", "free"])?;
            }
            if !charge.is_null() {
                known(
                    charge,
                    &["charge", "readyCharge", "chargeType", "lastCharge"],
                )?;
            }
            self.add(
                "ability",
                json!({"parent":id,"kind":k,"owner":u["owner"]}),
                vec![
                    json!(array(&u["traits"]).contains(&k)),
                    usage["once"].clone(),
                    usage["free"].clone(),
                    charge["charge"].clone(),
                    charge["readyCharge"].clone(),
                    self.cat(&charge["chargeType"], "modes")?,
                    charge["lastCharge"].clone(),
                ],
            )?;
        }
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
    let observation = tree.observation;
    let viewer = tree.actor;
    let catalog = tree.catalog;
    validate_schema(&catalog.encoding)?;
    no_null(observation)?;
    // 树的状态已从同一公开观察构造；像 TS 一样复用只读实体，避免每个参数节点重建整局。
    let s = &tree.state;
    let v = observation;
    if ![1, 2].contains(&viewer) || v["version"] != 2 {
        return Err("invalid encoding viewer/version".into());
    }
    known(
        v,
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
    for u in array(&v["units"]).iter().chain(array(&v["landmarks"])) {
        e.reference(&u["id"])?;
    }
    for p in &sidekeys {
        for c in array(&v["hands"][p]) {
            e.reference(&c["id"])?;
        }
    }
    for d in array(&v["deaths"]) {
        e.reference(&d["id"])?;
    }
    for (p, key) in sides.iter().zip(&sidekeys) {
        for pair in [
            "turns",
            "bases",
            "heads",
            "hands",
            "bonus",
            "baseEffects",
            "deployRows",
        ] {
            known(&v[pair], &["1", "2"])?;
        }
        let at = crate::geometry::base_point(*p);
        let mut fields = vec![
            json!(at.x),
            json!(at.y),
            v["bases"][key].clone(),
            v["turns"][key].clone(),
            v["heads"][key].clone(),
            v["bonus"][key].clone(),
        ];
        fields.extend((1..=13).map(|i| json!(array(&v["deployRows"][key]).contains(&json!(i)))));
        let id = format!("base-{p}");
        e.add("base", json!({"id":id,"owner":p,"selectable":true}), fields)?;
        for (i, eff) in array(&v["baseEffects"][key]).iter().enumerate() {
            e.effect(eff, &id, i)?;
        }
    }
    for (i, u) in array(&v["units"]).iter().enumerate() {
        e.unit(u, "unit", None, i)?;
    }
    for (i, u) in array(&v["landmarks"]).iter().enumerate() {
        e.unit(u, "landmark", None, i)?;
    }
    for (p, key) in sides.iter().zip(&sidekeys) {
        for (i, c) in array(&v["hands"][key]).iter().enumerate() {
            e.card(c, &json!(p), "card", None, i)?;
        }
    }
    for (i, d) in array(&v["deaths"]).iter().enumerate() {
        known(d, &["id", "kind", "owner", "ply", "revived", "group"])?;
        let mut fields = vec![d["ply"].clone(), d["revived"].clone()];
        fields.extend(e.printed(&d["kind"])?);
        e.add(
            "death",
            crate::actions::extend(d, json!({"order":i,"selectable":true})),
            fields,
        )?;
    }
    for (i, h) in array(&v["hazards"]).iter().enumerate() {
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
    for (i, l) in array(&v["siphons"]).iter().enumerate() {
        known(l, &["id", "sourceId", "owner", "fromId", "toId"])?;
        e.add("siphon",json!({"id":l["id"],"owner":l["owner"],"parent":l["fromId"],"source":l["sourceId"],"target":l["toId"],"order":i}),vec![])?;
    }
    for (i, m) in array(&v["iceMarks"]).iter().enumerate() {
        known(m, &["id", "sourceId", "owner", "due", "x", "y"])?;
        e.add(
            "ice",
            json!({"id":m["id"],"owner":m["owner"],"source":m["sourceId"],"order":i}),
            vec![m["x"].clone(), m["y"].clone(), m["due"].clone()],
        )?;
    }
    for (i, r) in array(&v["pending"]).iter().enumerate() {
        known(r, &["kind", "targetId", "owner", "source", "amount"])?;
        let id = format!("reaction:{i}");
        e.add("reaction",json!({"id":id,"owner":r["owner"],"source":r["source"]["id"],"target":r["targetId"],"order":i}),vec![e.cat(&r["kind"],"reactions")?,r["amount"].clone()])?;
        e.unit(&r["source"], "snapshot", Some(&id), 0)?;
    }
    for pair in ["auras", "clockFrames"] {
        if let Some(v) = v.get(pair) {
            known(v, &["1", "2"])?;
        }
    }
    for (p, key) in sides.iter().zip(&sidekeys) {
        for (i, a) in array(&v["auras"][key]).iter().enumerate() {
            known(a, &["kind", "parity", "usedPly"])?;
            e.add(
                "aura",
                json!({"kind":a["kind"],"owner":p,"order":i}),
                vec![cat(&a["parity"], &["odd", "even"])?, a["usedPly"].clone()],
            )?;
        }
        let frames = &v["clockFrames"][key];
        if !frames.is_null() {
            known(frames, &["current", "previous"])?;
        }
        for (i, name) in ["current", "previous"].iter().enumerate() {
            if let Some(f) = frames.get(*name) {
                known(f, &["ply", "turns", "units"])?;
                known(&f["turns"], &["1", "2"])?;
                let id = format!("frame:{p}:{name}");
                e.add(
                    "frame",
                    json!({"id":id,"owner":p,"order":i}),
                    vec![
                        f["ply"].clone(),
                        f["turns"][&sidekeys[0]].clone(),
                        f["turns"][&sidekeys[1]].clone(),
                    ],
                )?;
                for (j, u) in array(&f["units"]).iter().enumerate() {
                    e.unit(u, "snapshot", Some(&id), j)?;
                }
            }
        }
    }
    let draft = &v["shrineDraft"];
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
    if let Some(offer) = v.get("summonOffer") {
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
    let mut globals = vec![viewer as f64 / 2.0, e.owner(&v["active"])?];
    globals.extend(
        e.vocab("phases")
            .iter()
            .map(|p| if *p == v["phase"] { 1.0 } else { 0.0 }),
    );
    globals.push(if v["mode"] == "shrine" { 1.0 } else { 0.0 });
    for value in [
        &v["ply"],
        &v["summonSlots"],
        &v["turns"][&sidekeys[0]],
        &v["turns"][&sidekeys[1]],
        &v["bases"][&sidekeys[0]],
        &v["bases"][&sidekeys[1]],
        &v["heads"][&sidekeys[0]],
        &v["heads"][&sidekeys[1]],
        &v["bonus"][&sidekeys[0]],
        &v["bonus"][&sidekeys[1]],
        v.get("regularSummons").unwrap_or(&json!(0)),
    ] {
        globals.push(num(value)?);
    }
    globals.extend([
        if v["winner"].is_null() || v["winner"] == "draw" {
            0.0
        } else {
            e.owner(&v["winner"])?
        },
        if v.get("winner").is_some() { 1.0 } else { 0.0 },
        num(&json!(array(&v["pending"]).len()))?,
        num(&json!(array(&v["hands"][&sidekeys[0]]).len()))?,
        num(&json!(array(&v["hands"][&sidekeys[1]]).len()))?,
    ]);
    for p in sides {
        globals.push(if array(&v["shrineSetupDone"]).contains(&json!(p)) {
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
        num(v["summonOffer"].get("count").unwrap_or(&json!(0)))?,
        e.owner(&v["summonOffer"]["owner"])?,
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
    let v = tree.observation;
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
    let recipes: Vec<Value> = catalog.recipes.iter().map(|r| r["id"].clone()).collect();
    if let Some(c) = &node.prefix {
        let optional_kind = |k: &str| -> Result<Value, String> {
            if c[k].is_null() {
                Ok(Value::Null)
            } else {
                Ok(json!(e.kind(&c[k])?))
            }
        };
        let fields = vec![
            e.cat(&c["type"], "commands")?,
            e.cat(&c["mode"], "modes")?,
            c["x"].clone(),
            c["y"].clone(),
            c["row"].clone(),
            c["column"].clone(),
            c["ultimate"].clone(),
            c["charge"].clone(),
            cat(&c["direction"], &["up", "down", "left", "right"])?,
            optional_kind("ability")?,
            optional_kind("chosenKind")?,
            optional_kind("shrineKind")?,
            cat(&c["parity"], &["odd", "even"])?,
            category(&c["recipeId"], &recipes)?,
            if c["player"].is_null() {
                Value::Null
            } else {
                json!(e.owner(&c["player"])?)
            },
        ];
        e.add("prefix",json!({"id":"prefix","owner":viewer,"source":c.get("unitId").unwrap_or(&c["cardId"]),"target":c["targetId"]}),fields)?;
        for (f, key) in [
            "secondId",
            "deathId",
            "materialIds",
            "cardIds",
            "sacrificeIds",
        ]
        .iter()
        .enumerate()
        {
            let ids = if f < 2 {
                c.get(*key).map(|v| vec![v.clone()]).unwrap_or_default()
            } else {
                array(&c[*key]).to_vec()
            };
            for (i, id) in ids.iter().enumerate() {
                e.add(
                    "argument",
                    json!({"parent":"prefix","target":id,"order":i}),
                    vec![json!(f + 1)],
                )?;
            }
        }
        for (i, p) in array(&c["path"]).iter().enumerate() {
            e.add(
                "path",
                json!({"parent":"prefix","order":i}),
                vec![p["x"].clone(), p["y"].clone()],
            )?;
        }
        for (i, n) in array(&c["offerIndices"]).iter().enumerate() {
            e.add(
                "argument",
                json!({"parent":"prefix","order":i}),
                vec![json!(6), n.clone()],
            )?;
        }
    }
    let mut globals = base.globals.clone();
    globals[31] = if node.prefix.is_some() { 1.0 } else { 0.0 };
    let mut candidates = vec![];
    let mut sources = vec![];
    let mut targets = vec![];
    for choice in &node.choices {
        let c = &choice.command;
        let mut row = vec![0.0; 64];
        let ci = e
            .cat(&c["type"], "commands")?
            .as_u64()
            .ok_or("missing command category")? as usize;
        row[ci - 1] = 1.0;
        if !c["mode"].is_null() {
            row[23 + e.cat(&c["mode"], "modes")?.as_u64().unwrap() as usize - 1] = 1.0;
        }
        for (i, key, div) in [
            (36, "x", 9.0),
            (37, "y", 13.0),
            (38, "row", 13.0),
            (39, "column", 9.0),
        ] {
            row[i] = c[key].as_f64().unwrap_or(0.0) / div;
        }
        for (i, key) in [(40, "ultimate"), (41, "charge")] {
            row[i] = if c[key].is_null() {
                -1.0
            } else if c[key] == true {
                1.0
            } else {
                0.0
            };
        }
        row[42] = cat(&c["direction"], &["up", "down", "left", "right"])?
            .as_f64()
            .unwrap_or(0.0)
            / 4.0;
        for (i, key) in [(43, "ability"), (44, "chosenKind"), (45, "shrineKind")] {
            row[i] = e.kind(&c[key])? as f64 / 128.0;
        }
        row[46] = category(&c["recipeId"], &recipes)?.as_f64().unwrap_or(0.0) / 8.0;
        row[47] = if c["parity"] == "odd" {
            1.0
        } else if c["parity"] == "even" {
            -1.0
        } else {
            0.0
        };
        row[48] = e
            .cat(&json!(node.stage), "decision_stages")?
            .as_f64()
            .unwrap()
            / 16.0;
        row[49] = if choice.status != "parameter" {
            1.0
        } else {
            0.0
        };
        row[50] = if choice.status == "uncertain" {
            1.0
        } else {
            0.0
        };
        for i in 0..2 {
            row[51 + i] = c["offerIndices"][i]
                .as_f64()
                .map(|n| n / 16.0)
                .unwrap_or(-1.0);
        }
        for (i, key, div) in [
            (53, "materialIds", 3.0),
            (54, "sacrificeIds", 2.0),
            (55, "path", 117.0),
        ] {
            row[i] = array(&c[key]).len() as f64 / div;
        }
        row[56] = e.reference(&c["secondId"])? as f64 / 256.0;
        row[57] = e.reference(&c["deathId"])? as f64 / 256.0;
        row[58] = if choice.key == "commit-path" {
            1.0
        } else {
            0.0
        };
        row[59] = e.owner(&c["player"])?;
        if let Some(p) = array(&c["path"]).last() {
            row[60] = p["x"].as_f64().unwrap_or(0.0) / 9.0;
            row[61] = p["y"].as_f64().unwrap_or(0.0) / 13.0;
        }
        let source = c
            .get("unitId")
            .or_else(|| c.get("cardId"))
            .unwrap_or_else(|| {
                if c["type"] == "react" {
                    &v["pending"][0]["source"]["id"]
                } else {
                    &Value::Null
                }
            });
        sources.push(e.index(source)?);
        let subject = choice.subject.as_ref().map(|s| json!(s));
        targets.push(
            e.index(
                subject
                    .as_ref()
                    .or_else(|| c.get("targetId"))
                    .unwrap_or(&c["deathId"]),
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
