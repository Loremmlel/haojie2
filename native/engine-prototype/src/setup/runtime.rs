//! 原生开局与公开观察边界。随机状态只留在权威局面，策略接口采用明确白名单。
use crate::model::{Catalog, State};
use crate::resolution::Resolution;
use serde_json::{Value, json};

pub fn create(seed: u32, shrine: bool, catalog: &Catalog) -> Result<State, String> {
    let seed = if seed == 0 { 2654435769 } else { seed };
    let mut s:State=serde_json::from_value(json!({"version":2,"seed":seed,"rng":seed,"serial":1,"ply":1,"active":1,"phase":"summon","summonSlots":2,"turns":{"1":0,"2":0},"bases":{"1":300,"2":300},"baseEffects":{"1":[],"2":[]},"heads":{"1":0,"2":0},"hands":{"1":[],"2":[]},"bonus":{"1":0,"2":0},"deployRows":{"1":[1,2,3,4,5,6,7,8],"2":[6,7,8,9,10,11,12,13]},"units":[],"pending":[],"deaths":[],"hazards":[],"siphons":[],"iceMarks":[],"log":[],"events":[]})).map_err(|e|e.to_string())?;
    let mut ctx = Resolution::default();
    if shrine {
        s.extra.insert("mode".into(), json!("shrine"));
        s.phase = "shrine-draft".into();
        s.ply = 0.0;
        s.summon_slots = 0.0;
        s.extra.insert("auras".into(), json!({"1":[],"2":[]}));
        s.landmarks = Some(vec![]);
        s.extra.insert("shrineSetupDone".into(), json!([]));
        let mut offers = json!({});
        for p in 1..=2 {
            let mut pool = catalog.pools["shrine"].clone();
            let mut choices = vec![];
            for _ in 0..3 {
                let index = (s.random(false).map_err(|e| format!("{e:?}"))? * pool.len() as f64)
                    .floor() as usize;
                choices.push(pool.remove(index));
            }
            offers[p.to_string()] = json!(choices);
        }
        s.extra.insert("shrineDraft".into(),json!({"offers":offers,"committed":{"1":false,"2":false},"choices":{},"revealed":false}));
        ctx.emit(
            &mut s,
            json!({"type":"turn","text":"第0回合 · 秘密选择神龛"}),
            Some("双方各抽3个神龛；双方锁定后同时公布选择".into()),
        );
    } else {
        crate::lifecycle::begin(&mut s, catalog, &mut ctx).map_err(|e| format!("{e:?}"))?;
    }
    Ok(s)
}
pub fn viewer(s: &State) -> usize {
    s.pending
        .first()
        .and_then(|r| r["owner"].as_u64())
        .map(|v| v as usize)
        .unwrap_or(s.active)
}
pub fn observe(s: &State, viewer: usize) -> Result<Value, String> {
    if ![1, 2].contains(&viewer) {
        return Err("viewer must be 1 or 2".into());
    }
    let mut output = json!({"version":s.version,"serial":s.serial,"ply":s.ply,"active":s.active,"phase":s.phase,"summonSlots":s.summon_slots,"turns":s.turns,"bases":s.bases,"units":s.units,"pending":s.pending,"siphons":s.siphons,"deployRows":{"1":crate::geometry::deployment_rows(s,1),"2":crate::geometry::deployment_rows(s,2)}});
    for key in [
        "baseEffects",
        "heads",
        "hands",
        "bonus",
        "deaths",
        "hazards",
        "iceMarks",
        "mode",
        "auras",
        "regularSummons",
        "summonOffer",
        "shrineSetupDone",
        "clockFrames",
        "winner",
    ] {
        if let Some(v) = s.extra.get(key) {
            output[key] = v.clone();
        }
    }
    if let Some(landmarks) = &s.landmarks {
        output["landmarks"] = json!(landmarks);
    }
    if let Some(d) = s.extra.get("shrineDraft") {
        let mut choices = json!({});
        for p in 1..=2 {
            if let Some(c) = d["choices"]
                .get(p.to_string())
                .filter(|_| d["revealed"] == true || p == viewer)
            {
                let mut visible = json!({"kind":c["kind"]});
                if let Some(parity) = c.get("parity") {
                    visible["parity"] = parity.clone();
                }
                choices[p.to_string()] = visible;
            }
        }
        output["shrineDraft"] = json!({"offers":d["offers"],"committed":d["committed"],"revealed":d["revealed"],"choices":choices});
    }
    Ok(output)
}
