//! C ABI 仅交接批次描述和连续数组；控制与完整命令记录允许低频 JSON。
use super::{Array, Control, Pool, RecordBatch};
use crate::learning::search::Reply;
use serde_json::{Value, json};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CArray {
    data: *const f32,
    len: usize,
    ndim: usize,
    shape: [usize; 3],
}
impl Default for CArray {
    fn default() -> Self {
        Self {
            data: ptr::null(),
            len: 0,
            ndim: 0,
            shape: [0; 3],
        }
    }
}
impl CArray {
    fn from_array(array: &Array) -> Self {
        let mut shape = [0; 3];
        shape[..array.shape.len()].copy_from_slice(&array.shape);
        Self {
            data: array.values.as_ptr(),
            len: array.values.len(),
            ndim: array.shape.len(),
            shape,
        }
    }
}

#[repr(C)]
pub struct RequestView {
    id: u64,
    lane: u64,
    game_id: u64,
    step: u64,
    context: u64,
    model: i64,
    kind: u32,
    array_count: u32,
    key: *const u8,
    key_len: usize,
    arrays: [CArray; 4],
}

#[repr(C)]
pub struct ReplyView {
    id: u64,
    data: *const u16,
    len: usize,
    version: i64,
}

fn checked<T>(pool: &Pool, fallback: T, action: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            pool.shared.fail(error);
            fallback
        }
        Err(_) => {
            pool.shared.fail("原生池接口 panic".into());
            fallback
        }
    }
}

/// # Safety
/// 输入是 UTF-8 JSON，随机函数地址指向池存活期间保持加载的 NumPy 动态库。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_new(data: *const u8, length: usize) -> *mut Pool {
    match catch_unwind(AssertUnwindSafe(|| -> Result<Pool, String> {
        let config = serde_json::from_slice(unsafe { std::slice::from_raw_parts(data, length) })
            .map_err(|e| e.to_string())?;
        Pool::new(config)
    })) {
        Ok(Ok(pool)) => Box::into_raw(Box::new(pool)),
        _ => ptr::null_mut(),
    }
}

/// # Safety
/// pool 存活；控制调用仅由一个协调线程执行，返回字节在下一次 control 前有效。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_control(
    pool: *const Pool,
    data: *const u8,
    length: usize,
) -> *const u8 {
    let pool = unsafe { &*pool };
    let value = match catch_unwind(AssertUnwindSafe(|| -> Result<Value, String> {
        let input = serde_json::from_slice(unsafe { std::slice::from_raw_parts(data, length) })
            .map_err(|e| e.to_string())?;
        pool.control(input)
    })) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => json!({"error":error}),
        Err(_) => json!({"error":"原生池控制 panic"}),
    };
    let mut response = pool.response.lock().unwrap_or_else(|e| e.into_inner());
    *response = serde_json::to_vec(&value).unwrap();
    response.as_ptr()
}

/// # Safety
/// pool 存活；与 control 使用同一协调线程。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_response_len(pool: *const Pool) -> usize {
    unsafe { &*pool }
        .response
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .len()
}

/// # Safety
/// 请求批次只有一个 GPU 所有者；前一批次先 reply 并 release。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_take_requests(
    pool: *const Pool,
    maximum: usize,
    wait_ms: u64,
) -> usize {
    let pool = unsafe { &*pool };
    checked(pool, usize::MAX, || pool.take_requests(maximum, wait_ms))
}

/// # Safety
/// pool 存活，GPU 所有者未并发释放请求批次。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_request_count(pool: *const Pool) -> usize {
    unsafe { &*pool }
        .requests
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .len()
}

/// # Safety
/// out 可写；所有返回指针只在当前请求批次 release 前有效。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_request_view(
    pool: *const Pool,
    index: usize,
    out: *mut RequestView,
) -> i32 {
    let pool = unsafe { &*pool };
    checked(pool, 0, || {
        let held = pool.requests.lock().unwrap();
        let item = held.get(index).ok_or("请求批次索引越界")?;
        let values = item.arrays();
        let mut arrays = [CArray::default(); 4];
        for (view, value) in arrays.iter_mut().zip(&values) {
            *view = CArray::from_array(value);
        }
        unsafe {
            out.write(RequestView {
                id: item.id,
                lane: item.lane as u64,
                game_id: item.game_id,
                step: item.step,
                context: item.request.context,
                model: item.model,
                kind: item.request.kind,
                array_count: values.len() as u32,
                key: item.request.key.as_ptr(),
                key_len: item.request.key.len(),
                arrays,
            })
        };
        Ok(1)
    })
}

/// # Safety
/// 输入 BF16 数组在调用期间有效；每个 id 必须来自尚未回应的当前批次。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_reply(
    pool: *const Pool,
    data: *const ReplyView,
    count: usize,
) -> i32 {
    let pool = unsafe { &*pool };
    checked(pool, 0, || {
        let replies = unsafe { std::slice::from_raw_parts(data, count) };
        let mut held = pool.requests.lock().unwrap();
        for response in replies {
            let item = held
                .iter_mut()
                .find(|r| r.id == response.id && !r.replied)
                .ok_or("重复或未知神经回应")?;
            let expected = if item.request.kind == 0 {
                5
            } else {
                item.request.options.len() / 704
            };
            if response.len != expected {
                return Err("神经输出长度不匹配".into());
            }
            let values = unsafe { std::slice::from_raw_parts(response.data, response.len) }
                .iter()
                .map(|v| f32::from_bits((*v as u32) << 16))
                .collect();
            let (worker, local) = pool.destination(item.lane)?;
            pool.senders[worker]
                .send(Control::Reply(
                    local,
                    item.id,
                    Reply {
                        values,
                        version: response.version,
                    },
                ))
                .map_err(|_| "原生线程已经退出")?;
            item.replied = true;
        }
        Ok(1)
    })
}

/// # Safety
/// 所有数组视图已经停止使用，且当前批次的每个请求都已经回应。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_release_requests(pool: *const Pool) -> i32 {
    let pool = unsafe { &*pool };
    checked(pool, 0, || {
        let mut held = pool.requests.lock().unwrap();
        if held.iter().any(|r| !r.replied) {
            return Err("未回应的神经请求不可释放".into());
        }
        held.clear();
        Ok(1)
    })
}

/// # Safety
/// 正式记录批次由一个协调线程所有，前一批次须先释放。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_take_records(
    pool: *const Pool,
    maximum: usize,
    wait_ms: u64,
) -> usize {
    let pool = unsafe { &*pool };
    checked(pool, usize::MAX, || pool.take_records(maximum, wait_ms))
}

/// # Safety
/// 返回 JSON 字节只在当前记录批次 release 前有效。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_records_meta(pool: *const Pool) -> *const u8 {
    unsafe { &*pool }
        .records
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .metadata
        .as_ptr()
}

/// # Safety
/// pool 存活且记录批次未被其他线程释放。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_records_len(pool: *const Pool) -> usize {
    unsafe { &*pool }
        .records
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .metadata
        .len()
}

/// # Safety
/// out 可写；数组 view 必须在 release_records 前完成复制或消费。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_record_array(
    pool: *const Pool,
    index: usize,
    out: *mut CArray,
) -> i32 {
    let pool = unsafe { &*pool };
    checked(pool, 0, || {
        let held = pool.records.lock().unwrap();
        let array = held.arrays.get(index).ok_or("记录数组索引越界")?;
        unsafe { out.write(CArray::from_array(array)) };
        Ok(1)
    })
}

/// # Safety
/// pool 存活；Python 已取得所需数组的所有权，随后才可让lane继续。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_ack_records(
    pool: *const Pool,
    lanes: *const u64,
    count: usize,
) -> i32 {
    let pool = unsafe { &*pool };
    checked(pool, 0, || {
        for lane in unsafe { std::slice::from_raw_parts(lanes, count) } {
            let (worker, local) = pool.destination(*lane as usize)?;
            pool.senders[worker]
                .send(Control::Ack(local))
                .map_err(|_| "原生线程已经退出")?;
        }
        Ok(1)
    })
}

/// # Safety
/// 当前记录批次的所有数组 view 已停止使用。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_release_records(pool: *const Pool) {
    *unsafe { &*pool }
        .records
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = RecordBatch::default();
}

/// # Safety
/// pool 存活，唤醒操作可与取批次和控制线程并行。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_wake(pool: *const Pool) {
    unsafe { &*pool }.wake();
}

/// # Safety
/// 所有 Python 调用和数组 view 已结束；仅由句柄所有者释放一次。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc_pool_close(pool: *mut Pool) {
    if !pool.is_null() {
        drop(unsafe { Box::from_raw(pool) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::random::BitGenerator;
    use std::time::{Duration, Instant};

    unsafe extern "C" fn gamma(_: *mut BitGenerator, _: f64) -> f64 {
        1.
    }
    unsafe extern "C" fn bounded(
        _: *mut BitGenerator,
        offset: u64,
        _: u64,
        _: u64,
        _: bool,
    ) -> u64 {
        offset
    }

    fn respond(pool: &Pool, version: i64) {
        let count = unsafe { hc_pool_request_count(pool) };
        let mut views = vec![];
        for i in 0..count {
            let mut view = std::mem::MaybeUninit::uninit();
            assert_eq!(
                unsafe { hc_pool_request_view(pool, i, view.as_mut_ptr()) },
                1
            );
            views.push(unsafe { view.assume_init() });
        }
        let outputs: Vec<Vec<u16>> = views
            .iter()
            .map(|v| vec![0; if v.kind == 0 { 5 } else { v.arrays[0].shape[0] }])
            .collect();
        let replies: Vec<_> = views
            .iter()
            .zip(&outputs)
            .map(|(v, output)| ReplyView {
                id: v.id,
                data: output.as_ptr(),
                len: output.len(),
                version,
            })
            .collect();
        assert_eq!(
            unsafe { hc_pool_reply(pool, replies.as_ptr(), replies.len()) },
            1
        );
        assert_eq!(unsafe { hc_pool_release_requests(pool) }, 1);
    }

    fn acknowledge(pool: &Pool) -> Vec<Value> {
        let count = unsafe { hc_pool_take_records(pool, 8, 0) };
        assert_ne!(count, usize::MAX);
        if count == 0 {
            unsafe { hc_pool_release_records(pool) };
            return vec![];
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(hc_pool_records_meta(pool), hc_pool_records_len(pool))
        };
        let records: Vec<Value> = serde_json::from_slice(bytes).unwrap();
        let lanes: Vec<_> = records
            .iter()
            .map(|r| r["lane"].as_u64().unwrap())
            .collect();
        unsafe { hc_pool_release_records(pool) };
        assert_eq!(
            unsafe { hc_pool_ack_records(pool, lanes.as_ptr(), lanes.len()) },
            1
        );
        records
    }

    #[test]
    fn independent_lane_pause_reply_barrier_and_wake() {
        let pool = Pool::new(json!({"workers":2,"games_per_worker":1,"simulations":0,
            "fast_simulations":0,"full_fraction":1.,"temperature":0.,"audit":true,
            "gamma_address":gamma as *const () as usize,"bounded_address":bounded as *const () as usize,
            "exp_address":crate::learning::random::testing_exp as *const () as usize,
            "exp_context":0,"exp_auxdata":0})).unwrap();
        pool.control(json!({"op":"assign","lane":0,"game":{"id":0,"seed":713,"learner":1,"opponent":"self","ply_limit":0},
            "rng":{"bit_generator":"PCG64","state":{"state":"17","inc":"31"},"has_uint32":0,"uinteger":0},"restored":null})).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(Instant::now() < deadline, "正常初始局面未进入神经决策");
            let requests = unsafe { hc_pool_take_requests(&pool, 8, 2) };
            assert_ne!(requests, usize::MAX);
            if requests > 0 {
                break;
            }
            unsafe { hc_pool_release_requests(&pool) };
            acknowledge(&pool);
        }
        pool.control(json!({"op":"pause"})).unwrap();
        assert_eq!(
            pool.control(json!({"op":"barrier"})).unwrap()["ready"],
            false
        );
        respond(&pool, 7);
        assert_eq!(
            pool.control(json!({"op":"barrier"})).unwrap()["ready"],
            true
        );
        let snapshot = pool.control(json!({"op":"snapshot"})).unwrap();
        assert_eq!(snapshot["lanes"].as_array().unwrap().len(), 1);
        assert_eq!(snapshot["lanes"][0]["root_version"], 7);
        let status = pool.control(json!({"op":"status"})).unwrap();
        assert_eq!(status["threads"].as_array().unwrap().len(), 2);
        assert!(status["threads"][0].as_u64().unwrap() > 0);
        assert_ne!(status["threads"][0], status["threads"][1]);
        assert!(status["requests_taken"].as_u64().unwrap() > 0);
        // 控制唤醒在没有神经请求时也能让长等待立即返回。
        pool.wake();
        assert_eq!(unsafe { hc_pool_take_requests(&pool, 8, 1000) }, 0);
        unsafe { hc_pool_release_requests(&pool) };
        pool.control(json!({"op":"resume"})).unwrap();
        loop {
            assert!(Instant::now() < deadline);
            let requests = unsafe { hc_pool_take_requests(&pool, 8, 2) };
            assert_ne!(requests, usize::MAX);
            if requests > 0 {
                respond(&pool, 7);
            } else {
                unsafe { hc_pool_release_requests(&pool) };
            }
            let records = acknowledge(&pool);
            if !records.is_empty() {
                assert_eq!(records[0]["sample"]["version"], 7);
                assert!(records[0]["audit"]["rng_after"]["state"]["state"].is_string());
                break;
            }
        }
    }
}
