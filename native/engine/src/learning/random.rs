//! 与 NumPy 对应的随机状态和数学函数；复用无需 GIL 的原生分布与指数循环。
use serde_json::{Value, json};
use std::ffi::c_void;

#[repr(C)]
pub struct BitGenerator {
    state: *mut c_void,
    next_uint64: unsafe extern "C" fn(*mut c_void) -> u64,
    next_uint32: unsafe extern "C" fn(*mut c_void) -> u32,
    next_double: unsafe extern "C" fn(*mut c_void) -> f64,
    next_raw: unsafe extern "C" fn(*mut c_void) -> u64,
}

#[cfg(test)]
pub(crate) fn testing() -> Random {
    unsafe extern "C" fn gamma(generator: *mut BitGenerator, _: f64) -> f64 {
        unsafe { ((*generator).next_double)((*generator).state) }
    }
    unsafe extern "C" fn bounded(
        generator: *mut BitGenerator,
        offset: u64,
        range: u64,
        _: u64,
        _: bool,
    ) -> u64 {
        offset + unsafe { ((*generator).next_uint64)((*generator).state) } % (range + 1)
    }
    Random::from_value(
        &json!({"bit_generator":"PCG64", "state": {
        "state":"160078363690744033601390112987726904141",
        "inc":"17686443629577124697969402389330893883"
    }, "has_uint32":0, "uinteger":0}),
        Functions {
            gamma,
            bounded,
            exp: testing_exp,
            exp_context: 0,
            exp_auxdata: 0,
        },
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg64_matches_numpy_raw_words_and_uint32_cache() {
        let expected = [
            12587170189557361101_u64,
            992822559630912803,
            4064922177151560342,
            3401059606364785669,
            3244891138370822358,
        ];
        let mut rng = testing();
        for word in expected {
            assert_eq!(rng.next_u64(), word);
        }
        let mut rng = testing();
        assert_eq!(rng.next_u32(), expected[0] as u32);
        assert_eq!(rng.next_u64(), expected[1]);
        assert_eq!(rng.next_u32(), (expected[0] >> 32) as u32);
        assert_eq!(rng.next_u64(), expected[2]);
    }

    #[test]
    fn pcg64_snapshot_restores_full_state_and_choice_consumption() {
        let mut rng = testing();
        rng.next_u32();
        let snapshot = rng.snapshot();
        let mut restored = Random::from_value(&snapshot, rng.functions).unwrap();
        assert_eq!(restored.next_u32(), rng.next_u32());
        assert_eq!(restored.random(), rng.random());
        let mut single = rng.clone();
        assert_eq!(rng.choice(&[0.0, 1.0, 0.0]).unwrap(), 1);
        single.random();
        assert_eq!(single.snapshot(), rng.snapshot());
    }
}

pub type Gamma = unsafe extern "C" fn(*mut BitGenerator, f64) -> f64;
pub type Bounded = unsafe extern "C" fn(*mut BitGenerator, u64, u64, u64, bool) -> u64;
pub type Exp = unsafe extern "C" fn(
    *mut c_void,
    *const *mut u8,
    *const isize,
    *const isize,
    *mut c_void,
) -> i32;

#[cfg(test)]
pub(crate) unsafe extern "C" fn testing_exp(
    _: *mut c_void,
    data: *const *mut u8,
    dimensions: *const isize,
    _: *const isize,
    _: *mut c_void,
) -> i32 {
    let length = unsafe { *dimensions } as usize;
    let input = unsafe { std::slice::from_raw_parts((*data).cast::<f32>(), length) };
    let output = unsafe { std::slice::from_raw_parts_mut((*data.add(1)).cast::<f32>(), length) };
    for (output, input) in output.iter_mut().zip(input) {
        *output = input.exp();
    }
    0
}

#[derive(Clone, Copy)]
pub struct Functions {
    pub gamma: Gamma,
    pub bounded: Bounded,
    pub exp: Exp,
    pub exp_context: usize,
    pub exp_auxdata: usize,
}

impl Functions {
    /// # Safety
    /// 地址必须来自存活的 NumPy 动态库及 f32 exp capsule；循环无需 GIL 且只读上下文。
    pub unsafe fn from_addresses(
        gamma: usize,
        bounded: usize,
        exp: usize,
        exp_context: usize,
        exp_auxdata: usize,
    ) -> Result<Self, String> {
        if gamma == 0 || bounded == 0 || exp == 0 {
            return Err("NumPy 原生函数地址为空".into());
        }
        Ok(Self {
            gamma: unsafe { std::mem::transmute::<usize, Gamma>(gamma) },
            bounded: unsafe { std::mem::transmute::<usize, Bounded>(bounded) },
            exp: unsafe { std::mem::transmute::<usize, Exp>(exp) },
            exp_context,
            exp_auxdata,
        })
    }
}

#[derive(Clone)]
pub struct Random {
    state: u128,
    increment: u128,
    has_uint32: bool,
    uinteger: u32,
    functions: Functions,
}

impl Random {
    pub fn from_value(value: &Value, functions: Functions) -> Result<Self, String> {
        fn integer(value: &Value) -> Result<u128, String> {
            if let Some(text) = value.as_str() {
                text.parse().map_err(|_| "PCG64 十进制状态无效".into())
            } else if let Some(number) = value.as_u64() {
                Ok(number.into())
            } else {
                Err("PCG64 的 128 位状态必须通过字符串传入".into())
            }
        }
        if value["bit_generator"] != "PCG64" {
            return Err("仅支持 NumPy PCG64 状态".into());
        }
        Ok(Self {
            state: integer(&value["state"]["state"])?,
            increment: integer(&value["state"]["inc"])?,
            has_uint32: value["has_uint32"].as_u64().unwrap_or(0) != 0,
            uinteger: value["uinteger"].as_u64().unwrap_or(0) as u32,
            functions,
        })
    }

    pub fn snapshot(&self) -> Value {
        json!({"bit_generator":"PCG64", "state": {
            "state":self.state.to_string(), "inc":self.increment.to_string()
        }, "has_uint32":u8::from(self.has_uint32), "uinteger":self.uinteger})
    }

    pub fn next_u64(&mut self) -> u64 {
        // NumPy 的 XSL-RR 先更新状态；不能换成 DXSM 或输出旧状态。
        const MULTIPLIER: u128 = ((2549297995355413924_u128) << 64) | 4865540595714422341;
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
        ((self.state >> 64) as u64 ^ self.state as u64).rotate_right((self.state >> 122) as u32)
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let value = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (value >> 32) as u32;
        value as u32
    }

    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    pub fn exp(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0; input.len()];
        let arrays = [input.as_ptr() as *mut u8, output.as_mut_ptr().cast()];
        let length = input.len() as isize;
        let strides = [4_isize, 4];
        let status = unsafe {
            (self.functions.exp)(
                self.functions.exp_context as *mut c_void,
                arrays.as_ptr(),
                &length,
                strides.as_ptr(),
                self.functions.exp_auxdata as *mut c_void,
            )
        };
        if status != 0 {
            return Err("NumPy f32 exp 循环失败".into());
        }
        Ok(output)
    }

    fn bit_generator(&mut self) -> BitGenerator {
        unsafe extern "C" fn next64(state: *mut c_void) -> u64 {
            unsafe { (&mut *state.cast::<Random>()).next_u64() }
        }
        unsafe extern "C" fn next32(state: *mut c_void) -> u32 {
            unsafe { (&mut *state.cast::<Random>()).next_u32() }
        }
        unsafe extern "C" fn double(state: *mut c_void) -> f64 {
            unsafe { (&mut *state.cast::<Random>()).random() }
        }
        BitGenerator {
            state: (self as *mut Self).cast(),
            next_uint64: next64,
            next_uint32: next32,
            next_double: double,
            next_raw: next64,
        }
    }

    pub fn simulation_seed(&mut self) -> u32 {
        let function = self.functions.bounded;
        let mut generator = self.bit_generator();
        unsafe {
            function(
                &mut generator,
                1,
                u32::MAX as u64 - 1,
                u32::MAX as u64,
                false,
            ) as u32
        }
    }

    pub fn dirichlet(&mut self, length: usize) -> Vec<f64> {
        let function = self.functions.gamma;
        let mut generator = self.bit_generator();
        let mut result: Vec<_> = (0..length)
            .map(|_| unsafe { function(&mut generator, 0.3) })
            .collect();
        let inverse = 1.0 / result.iter().copied().fold(0.0, |a, b| a + b);
        for value in &mut result {
            *value *= inverse;
        }
        result
    }

    pub fn choice(&mut self, probabilities: &[f64]) -> Result<usize, String> {
        if probabilities.is_empty() {
            return Err("不能从空概率分布抽样".into());
        }
        let mut cdf = Vec::with_capacity(probabilities.len());
        let mut total = 0.0;
        for value in probabilities {
            if !value.is_finite() || *value < 0.0 {
                return Err("概率分布无效".into());
            }
            total += value;
            cdf.push(total);
        }
        if total <= 0.0 {
            return Err("概率分布总质量为零".into());
        }
        for value in &mut cdf {
            *value /= total;
        }
        let uniform = self.random();
        Ok(cdf.partition_point(|value| *value <= uniform))
    }
}
