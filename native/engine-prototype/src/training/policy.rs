//! 与经济性实验 TinyPolicy 同形；双精度累加，每层和池化写回 Float32，不训练权重。
use crate::encoding::Input;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
/// 行身份限定一次决策；Weak 防止释放后的地址复用命中旧投影，不缓存池化或候选结果。
pub struct Decision<'a> {
    policy: &'a TinyPolicy,
    rows: HashMap<usize, EntityProjection>,
}
struct EntityProjection {
    row: Weak<Vec<f64>>,
    kind: usize,
    hidden: Rc<Vec<f32>>,
}
impl Decision<'_> {
    pub fn logits(&mut self, input: &Input) -> Vec<f64> {
        // 释放已失效前缀对应的计算结果；固定行仍由树编码持有。
        self.rows
            .retain(|_, projection| projection.row.strong_count() > 0);
        self.policy.evaluate(input, Some(&mut self.rows))
    }
}

pub struct Random(pub u32);
impl Random {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f64 + 0.5) / 4294967296.0
    }
}
pub struct TinyPolicy {
    embedding: Vec<f32>,
    entity: Vec<f32>,
    global: Vec<f32>,
    action: Vec<f32>,
    output: Vec<f32>,
}
fn layer(input: &[f64], weights: &[f32]) -> Vec<f32> {
    weights
        .chunks_exact(input.len() + 1)
        .map(|w| {
            let mut value = w[input.len()] as f64;
            for (x, w) in input.iter().zip(w) {
                value += x * (*w as f64);
            }
            value.tanh() as f32
        })
        .collect()
}
impl TinyPolicy {
    pub fn new(seed: u32) -> Self {
        let mut random = Random::new(seed);
        let mut matrix = |inputs: usize, outputs: usize| -> Vec<f32> {
            (0..(inputs + 1) * outputs)
                .map(|_| ((random.next() * 2.0 - 1.0) / (inputs as f64).sqrt()) as f32)
                .collect()
        };
        Self {
            embedding: matrix(255, 16),
            entity: matrix(80, 32),
            global: matrix(32, 32),
            action: matrix(160, 32),
            output: matrix(32, 1),
        }
    }
    pub fn logits(&self, input: &Input) -> Vec<f64> {
        self.evaluate(input, None)
    }
    pub fn decision(&self) -> Decision<'_> {
        Decision {
            policy: self,
            rows: HashMap::new(),
        }
    }
    fn evaluate(
        &self,
        input: &Input,
        mut cache: Option<&mut HashMap<usize, EntityProjection>>,
    ) -> Vec<f64> {
        let mut entity_input = [0.0; 80];
        let entities: Vec<_> = input
            .entities
            .iter()
            .zip(&input.kinds)
            .map(|(row, k)| {
                let identity = Rc::as_ptr(row) as usize;
                if let Some(known) = cache.as_ref().and_then(|c| c.get(&identity))
                    && known.kind == *k
                    && known
                        .row
                        .upgrade()
                        .is_some_and(|saved| Rc::ptr_eq(&saved, row))
                {
                    return known.hidden.clone();
                }
                entity_input[..64].copy_from_slice(row);
                for j in 0..16 {
                    entity_input[64 + j] = self.embedding[k * 16 + j] as f64;
                }
                let hidden = Rc::new(layer(&entity_input, &self.entity));
                if let Some(cache) = cache.as_mut() {
                    cache.insert(
                        identity,
                        EntityProjection {
                            row: Rc::downgrade(row),
                            kind: *k,
                            hidden: hidden.clone(),
                        },
                    );
                }
                hidden
            })
            .collect();
        let mut context = layer(&input.globals, &self.global);
        let count = input.entity_mask.iter().filter(|v| **v).count().max(1) as f64;
        for (entity, mask) in entities.iter().zip(&input.entity_mask) {
            if *mask {
                for (c, v) in context.iter_mut().zip(entity.iter()) {
                    *c = (*c as f64 + *v as f64 / count) as f32;
                }
            }
        }
        let empty = [0.0; 32];
        let mut action_input = [0.0; 160];
        for j in 0..32 {
            action_input[64 + j] = context[j] as f64;
        }
        input
            .candidates
            .iter()
            .enumerate()
            .map(|(i, row)| {
                action_input[..64].copy_from_slice(row);
                let source = entities
                    .get(input.sources[i] as usize)
                    .map(|v| v.as_slice())
                    .unwrap_or(&empty);
                let target = entities
                    .get(input.targets[i] as usize)
                    .map(|v| v.as_slice())
                    .unwrap_or(&empty);
                for j in 0..32 {
                    action_input[96 + j] = source[j] as f64;
                    action_input[128 + j] = target[j] as f64;
                }
                let hidden = layer(&action_input, &self.action);
                let mut out = self.output[32] as f64;
                for (h, w) in hidden.iter().zip(&self.output) {
                    out += (*h as f64) * (*w as f64);
                }
                out
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readonly_rows_reuse_exact_projection_without_reusing_masks_or_dead_prefixes() {
        let policy = TinyPolicy::new(73129);
        let mut random = Random::new(19);
        let mut row = || Rc::new((0..64).map(|_| random.next()).collect::<Vec<_>>());
        let fixed = row();
        let mut input = Input {
            entities: vec![fixed.clone(), row()],
            kinds: vec![1, 150],
            globals: vec![0.25; 32],
            candidates: vec![vec![0.5; 64], vec![0.75; 64]],
            entity_mask: vec![true, true],
            candidate_mask: vec![true, true],
            sources: vec![0, -1],
            targets: vec![1, 0],
        };
        let mut cache = policy.decision();
        for i in 0..100 {
            input.entities[1] = row();
            input.entity_mask[0] = i % 2 == 0;
            input.kinds[1] = 128 + i % 24;
            assert_eq!(cache.logits(&input), policy.logits(&input));
            assert!(cache.rows.len() <= 2, "失效前缀不能积累投影或复用地址");
        }
        Rc::make_mut(&mut input.entities[0])[0] = 9.0;
        assert_ne!(input.entities[0][0], fixed[0]);
        assert_eq!(cache.logits(&input), policy.logits(&input));
        assert_eq!(policy.decision().logits(&input), policy.logits(&input));
    }
}
