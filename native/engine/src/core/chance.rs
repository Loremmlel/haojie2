//! 搜索专用随机源；作用域结束即移除，正式 PRNG 与预检保持原行为。
use crate::model::Failure;
use std::cell::RefCell;

#[derive(Default)]
struct Source {
    tape: Vec<f64>,
    offset: usize,
    fallback: Option<(u32, usize, usize)>,
    index: usize,
    cuts: Option<Vec<f64>>,
}
thread_local! {static SOURCE:RefCell<Option<Source>>=const{RefCell::new(None)};}
pub fn hash(text: &str) -> u32 {
    text.encode_utf16()
        .fold(2166136261, |n, v| (n ^ u32::from(v)).wrapping_mul(16777619))
}
pub fn supplied(boundaries: &[f64]) -> Option<Result<f64, Failure>> {
    SOURCE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let source = slot.as_mut()?;
        if source.offset < source.tape.len() {
            let n = source.tape[source.offset];
            source.offset += 1;
            return Some(Ok(n));
        }
        if let Some((seed, lane, samples)) = source.fallback {
            let jitter =
                (f64::from(hash(&format!("{seed}:{}:{lane}", source.index))) + 0.5) / 4294967296.;
            source.index += 1;
            return Some(Ok(if source.index == 1 {
                (lane as f64 + jitter) / samples as f64
            } else {
                jitter
            }));
        }
        let mut cuts = boundaries.to_vec();
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        if cuts.len() == 2 {
            return Some(Ok((cuts[0] + cuts[1]) / 2.));
        }
        source.cuts = Some(cuts);
        Some(Err(Failure::Uncertain))
    })
}
pub fn using<T>(
    tape: &[f64],
    fallback: Option<(u32, usize, usize)>,
    run: impl FnOnce() -> T,
) -> (T, Option<Vec<f64>>) {
    struct Restore(Option<Source>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SOURCE.with(|s| {
                *s.borrow_mut() = self.0.take();
            });
        }
    }
    let old = SOURCE.with(|slot| {
        slot.replace(Some(Source {
            tape: tape.to_vec(),
            fallback,
            ..Default::default()
        }))
    });
    let _restore = Restore(old);
    let result = run();
    let cuts = SOURCE.with(|slot| slot.borrow_mut().as_mut().unwrap().cuts.take());
    (result, cuts)
}
