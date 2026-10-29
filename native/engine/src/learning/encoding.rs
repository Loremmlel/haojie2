//! 普通局实体行与完整命令行；公开状态之外的数据不得进入网络缓冲。
use crate::{
    model::{Catalog, Command, State},
    options::{array, text},
};
use serde_json::Value;
use std::collections::HashMap;

pub const STATE_WIDTH: usize = 128;
pub const ACTION_WIDTH: usize = 576;
const UNIT_FIELDS: &[&str] = &[
    "x",
    "y",
    "hp",
    "maxHp",
    "size",
    "born",
    "offset",
    "deployedAt",
    "chargedOnDeploy",
    "mode",
    "operations",
    "shots",
    "moves",
    "bonusAttacks",
    "bonusSequence",
    "weaponFirstUsed",
    "charge",
    "readyCharge",
    "chargeType",
    "lastCharge",
    "upgrades",
    "kills",
    "attackBonus",
    "rangeBonus",
    "guardUsed",
    "rerollUsedPly",
    "silenced",
    "freeUsed",
    "onceUsed",
    "extraOperations",
    "bannerHp",
    "overMaxFromBanner",
    "bladeQualified",
    "expiresAt",
    "hookReadyAt",
    "hookExpiresAt",
];
const MODES: &[&str] = &[
    "none", "move", "attack", "skill", "charge", "damage", "heal", "summon", "single", "double",
    "row", "column", "pull", "normal",
];
const EFFECTS: &[&str] = &[
    "attack",
    "immune",
    "execute",
    "convert",
    "mark",
    "freeze",
    "burn",
    "stun",
    "inner-fire",
];
const REACTIONS: &[&str] = &["death-shot", "reflect", "bounce", "hut-spawn", "hit-pull"];
const COMMANDS: &[&str] = &[
    "end",
    "summon",
    "begin",
    "reroll",
    "deploy",
    "move",
    "attack",
    "charge",
    "skill",
    "cast",
    "equip",
    "craft",
    "synthesize",
    "skip-synthesis",
    "react",
    "finish-mode",
];
fn category(value: &str, values: &[&str]) -> f32 {
    values
        .iter()
        .position(|s| *s == value)
        .map_or(0., |i| (i + 1) as f32)
}
fn number(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_bool().map(|v| f64::from(u8::from(v))))
        .or_else(|| v.as_str().map(|v| category(v, MODES) as f64))
}
fn scalar(row: &mut [f32; STATE_WIDTH], index: usize, value: Option<f64>) {
    if let Some(value) = value {
        row[12 + index] = (value.signum() * value.abs().ln_1p() / 8.) as f32;
        row[64 + index] = 1.;
    }
}

struct Encoder<'a> {
    catalog: &'a Catalog,
    viewer: usize,
    rows: Vec<[f32; STATE_WIDTH]>,
    identities: HashMap<String, usize>,
    indices: HashMap<String, usize>,
}
impl Encoder<'_> {
    fn reference(&mut self, id: &str) -> f32 {
        if id.is_empty() {
            return 0.;
        }
        let next = self.identities.len() + 1;
        *self.identities.entry(id.to_owned()).or_insert(next) as f32
    }
    fn kind(&self, v: &Value) -> f32 {
        let key = if let Some(n) = v.as_u64() {
            n.to_string()
        } else {
            text(v).to_owned()
        };
        self.catalog
            .keys()
            .position(|k| *k == key)
            .map_or(0., |i| (i + 1) as f32)
    }
    fn add(&mut self, role: f32, v: &Value, parent: &str, order: usize, fields: &[&str]) -> usize {
        let mut row = [0.; STATE_WIDTH];
        row[0] = role;
        row[1] = self.kind(&v["kind"]);
        row[2] = v["owner"]
            .as_u64()
            .map_or(0., |p| if p as usize == self.viewer { 1. } else { -1. });
        row[3] = self.reference(text(&v["id"]));
        row[4] = self.reference(parent);
        row[5] = self.reference(text(&v["sourceId"]));
        row[6] = self.reference(text(&v["targetId"]));
        if let Some(group) = v["group"].as_str() {
            row[7] = self.reference(&format!("group:{group}"));
        }
        row[8] = order as f32;
        for (i, key) in fields.iter().enumerate() {
            scalar(&mut row, i, number(&v[*key]));
        }
        let index = self.rows.len();
        if let Some(id) = v["id"].as_str() {
            self.indices.entry(id.to_owned()).or_insert(index);
        }
        self.rows.push(row);
        index
    }
    fn effect(&mut self, v: &Value, parent: &str, order: usize) {
        let index = self.add(5., v, parent, order, &["from", "until", "amount", "global"]);
        scalar(
            &mut self.rows[index],
            4,
            Some(category(text(&v["type"]), EFFECTS) as f64),
        );
    }
    fn unit(&mut self, v: &Value, parent: &str, order: usize, state: &State) {
        let index = self.add(2., v, parent, order, UNIT_FIELDS);
        let id = text(&v["id"]);
        if let Some(u) = state.unit(id) {
            let stats = crate::stats::stats(state, u, self.catalog);
            for (i, n) in [
                stats.attack,
                stats.range,
                stats.actions,
                stats.remaining,
                stats.movement,
                stats.operation_limit,
                stats.operations_left,
                f64::from(u8::from(stats.sleeping)),
                f64::from(u8::from(stats.frozen)),
                f64::from(u8::from(stats.stunned)),
            ]
            .into_iter()
            .enumerate()
            {
                scalar(&mut self.rows[index], 36 + i, Some(n));
            }
        }
        for (i, e) in array(&v["effects"]).iter().enumerate() {
            self.effect(e, id, i);
        }
        for (key, role, reference_field) in [
            ("attacked", 7., "targetId"),
            ("guardSourceIds", 8., "sourceId"),
        ] {
            for (i, target) in array(&v[key]).iter().enumerate() {
                let record = serde_json::json!({reference_field:target,"owner":v["owner"]});
                self.add(role, &record, id, i, &[]);
            }
        }
        for (i, kind) in array(&v["equipment"]).iter().enumerate() {
            let key = if let Some(n) = kind.as_u64() {
                n.to_string()
            } else {
                text(kind).to_owned()
            };
            self.add(
                6.,
                &serde_json::json!({"kind":kind,"owner":v["owner"],"id":v["equipmentIds"][key]}),
                id,
                i,
                &[],
            );
        }
        for (i, hit) in array(&v["receivedDamage"]).iter().enumerate() {
            self.add(9., hit, id, i, &["ply", "amount"]);
        }
        // 普通局没有强夺或神龛装备；意外输入直接失败由入口统一检查。
    }
}

pub fn encode(
    state: &State,
    catalog: &Catalog,
    commands: &[Command],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let viewer = crate::runtime::viewer(state);
    let view = crate::runtime::public_view(state, viewer)?;
    let public = view.position();
    let s = serde_json::to_value(public).map_err(|e| e.to_string())?;
    if s.get("rng").is_some() || s.get("seed").is_some() {
        return Err("private RNG in observation".into());
    }
    let mut e = Encoder {
        catalog,
        viewer,
        rows: vec![[0.; STATE_WIDTH]],
        identities: HashMap::new(),
        indices: HashMap::new(),
    };
    let global = &mut e.rows[0];
    for (i, n) in [
        s["ply"].as_f64(),
        Some(if public.active == viewer { 1. } else { -1. }),
        Some(category(&public.phase, &["synthesis", "summon", "play"]) as f64),
        s["summonSlots"].as_f64(),
        s["turns"][viewer.to_string()].as_f64(),
        s["turns"][(3 - viewer).to_string()].as_f64(),
        s["serial"].as_f64(),
    ]
    .into_iter()
    .enumerate()
    {
        scalar(global, i, n);
    }
    for side in [viewer, 3 - viewer] {
        let id = format!("base-{side}");
        let record = serde_json::json!({"id":id,"owner":side,"hp":s["bases"][side.to_string()],
            "heads":s["heads"][side.to_string()],"bonus":s["bonus"][side.to_string()]});
        e.add(1., &record, "", side, &["hp", "heads", "bonus"]);
        for (i, effect) in array(&s["baseEffects"][side.to_string()])
            .iter()
            .enumerate()
        {
            e.effect(effect, &id, i);
        }
    }
    for (i, u) in array(&s["units"]).iter().enumerate() {
        if u.get("traits").is_some() || u.get("abilityCharges").is_some() {
            return Err("shrine-only unit in classic training".into());
        }
        e.unit(u, "", i, public);
    }
    for side in [viewer, 3 - viewer] {
        for (i, card) in array(&s["hands"][side.to_string()]).iter().enumerate() {
            let mut card = card.clone();
            card["owner"] = serde_json::json!(side);
            let index = e.add(
                3.,
                &card,
                "",
                i,
                &["drawnAt", "expiresAt", "rerolled", "summonedPly"],
            );
            scalar(
                &mut e.rows[index],
                4,
                Some(category(text(&card["summonPool"]), &["normal", "ultimate"]) as f64),
            );
        }
    }
    for (i, record) in array(&s["deaths"]).iter().enumerate() {
        e.add(4., record, "", i, &["ply", "revived"]);
    }
    for (i, record) in array(&s["hazards"]).iter().enumerate() {
        let index = e.add(10., record, "", i, &["line", "due"]);
        scalar(
            &mut e.rows[index],
            2,
            Some(category(text(&record["axis"]), &["row", "column"]) as f64),
        );
    }
    for (i, record) in array(&s["siphons"]).iter().enumerate() {
        let index = e.add(11., record, "", i, &[]);
        e.rows[index][9] = e.reference(text(&record["fromId"]));
        e.rows[index][10] = e.reference(text(&record["toId"]));
    }
    for (i, record) in array(&s["iceMarks"]).iter().enumerate() {
        e.add(12., record, "", i, &["x", "y", "due"]);
    }
    // 老千来自普通局合成，光环与本回合指定抽牌次数仍属普通局规则。
    for side in [viewer, 3 - viewer] {
        for (i, aura) in array(&s["auras"][side.to_string()]).iter().enumerate() {
            let mut aura = aura.clone();
            aura["owner"] = serde_json::json!(side);
            e.add(14., &aura, "", i, &["usedPly"]);
        }
    }
    for (i, record) in array(&s["pending"]).iter().enumerate() {
        let index = e.add(13., record, "", i, &["amount"]);
        scalar(
            &mut e.rows[index],
            1,
            Some(category(text(&record["kind"]), REACTIONS) as f64),
        );
        let parent = format!("reaction-{i}");
        e.rows[index][3] = e.reference(&parent);
        e.unit(&record["source"], &parent, i, public);
    }
    e.rows[0][9] = e.reference(text(&s["transitUnitId"]));
    let mut actions = Vec::with_capacity(commands.len() * ACTION_WIDTH);
    let index = |id: Option<&str>| id.and_then(|id| e.indices.get(id)).copied().unwrap_or(0) as f32;
    for c in commands {
        let mut row = [0.; ACTION_WIDTH];
        row[0] = category(&c.kind, COMMANDS);
        row[1] = index(c.unit_id.as_deref().or(c.card_id.as_deref()));
        row[2] = index(c.target_id.as_deref());
        row[3] = index(c.card_id.as_deref());
        row[4] = index(c.death_id.as_deref());
        row[5] = index(c.second_id.as_deref());
        row[6] = c
            .ability
            .as_ref()
            .map_or(0., |k| e.kind(&serde_json::json!(k)));
        row[7] = catalog
            .recipes
            .iter()
            .position(|r| r["id"].as_str() == c.recipe_id.as_deref())
            .map_or(0., |i| (i + 1) as f32);
        row[8] = category(c.mode.as_deref().unwrap_or(""), MODES);
        row[9] = category(
            c.direction.as_deref().unwrap_or(""),
            &["up", "down", "left", "right"],
        );
        row[10] = c.charge.map_or(-1., |v| f32::from(u8::from(v)));
        row[11] = c.ultimate.map_or(-1., |v| f32::from(u8::from(v)));
        row[12] = c
            .chosen_kind
            .as_ref()
            .map_or(0., |k| e.kind(&serde_json::json!(k)));
        for (field, ids) in [
            (16, c.material_ids.as_deref()),
            (20, c.card_ids.as_deref()),
            (24, c.sacrifice_ids.as_deref()),
        ] {
            if let Some(ids) = ids {
                if ids.len() > 3 {
                    return Err("classic command list exceeds three".into());
                }
                for (i, id) in ids.iter().enumerate() {
                    row[field + i] = index(Some(id)) / 256.;
                }
            }
        }
        for (i, n) in [c.x, c.y, c.row, c.column].into_iter().enumerate() {
            if let Some(n) = n {
                row[32 + i] = n as f32 / 13.;
                row[36 + i] = 1.;
            }
        }
        if let Some(path) = &c.path {
            if path.len() > 256 {
                return Err("path exceeds rule limit".into());
            }
            row[40] = path.len() as f32 / 256.;
            for (i, p) in path.iter().enumerate() {
                row[64 + i * 2] = p.x as f32 / 9.;
                row[65 + i * 2] = p.y as f32 / 13.;
            }
        }
        actions.extend_from_slice(&row);
    }
    Ok((e.rows.into_iter().flatten().collect(), actions))
}
