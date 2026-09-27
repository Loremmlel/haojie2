//! 与经济性实验 TinyPolicy 同形；双精度累加，每层和池化写回 Float32，不训练权重。
use crate::encoding::Input;
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
        let entities: Vec<_> = input
            .entities
            .iter()
            .zip(&input.kinds)
            .map(|(row, k)| {
                let mut v = row.clone();
                v.extend(
                    self.embedding[k * 16..(k + 1) * 16]
                        .iter()
                        .map(|v| *v as f64),
                );
                layer(&v, &self.entity)
            })
            .collect();
        let mut context = layer(&input.globals, &self.global);
        let count = input.entity_mask.iter().filter(|v| **v).count().max(1) as f64;
        for (entity, mask) in entities.iter().zip(&input.entity_mask) {
            if *mask {
                for (c, v) in context.iter_mut().zip(entity) {
                    *c = (*c as f64 + *v as f64 / count) as f32;
                }
            }
        }
        let empty = vec![0.0; 32];
        input
            .candidates
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut v = row.clone();
                for part in [
                    &context,
                    entities.get(input.sources[i] as usize).unwrap_or(&empty),
                    entities.get(input.targets[i] as usize).unwrap_or(&empty),
                ] {
                    v.extend(part.iter().map(|v| *v as f64));
                }
                let hidden = layer(&v, &self.action);
                let mut out = self.output[32] as f64;
                for (h, w) in hidden.iter().zip(&self.output) {
                    out += (*h as f64) * (*w as f64);
                }
                out
            })
            .collect()
    }
}
