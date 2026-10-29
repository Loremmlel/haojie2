//! 池工作线程的可选分项计时；独占时间扣除子项，Windows CPU 周期不换算秒。
use serde_json::Value;
use std::{cell::RefCell, time::Instant};

#[derive(Clone, Copy)]
pub enum Phase {
    WorkerActive,
    WorkerControl,
    WorkerWait,
    WorkerPaused,
    Search,
    HostEncode,
    HostDecode,
    RuleApply,
    Transition,
    StateFork,
    PublicView,
    InfoMetrics,
    StateKey,
    Decoder,
    StateEncoding,
    Geometry,
    OptionEncoding,
    Teacher,
    CommitRecord,
    PublicSerialization,
}
const NAMES: [&str; 20] = [
    "worker_active",
    "worker_control",
    "worker_wait",
    "worker_paused",
    "search",
    "host_encode",
    "host_decode",
    "rule_apply",
    "transition",
    "state_fork",
    "public_view",
    "info_metrics",
    "state_key",
    "decoder",
    "state_encoding",
    "geometry",
    "option_encoding",
    "teacher",
    "commit_record",
    "public_serialization",
];

#[derive(Clone, Copy, Default)]
struct Row {
    calls: u64,
    inclusive_ns: u64,
    exclusive_ns: u64,
    cpu_cycles: Option<(u64, u64)>,
}
struct Frame {
    phase: usize,
    start: Instant,
    children_ns: u64,
    start_cycles: Option<u64>,
    children_cycles: Option<u64>,
}
#[derive(Default)]
struct Timing {
    enabled: bool,
    rows: [Row; NAMES.len()],
    stack: Vec<Frame>,
}
thread_local! {
    static TIMING: RefCell<Timing> = RefCell::new(Timing::default());
}

/// 只由池工作线程在启动时调用；不会统计池外的轨迹准备或兼容环境。
pub fn initialize(enabled: bool) {
    TIMING.with_borrow_mut(|timing| {
        assert!(timing.stack.is_empty());
        timing.enabled = enabled;
        timing.rows = [Row::default(); NAMES.len()];
        if enabled {
            timing.stack.reserve(32);
        }
    });
}

pub struct Guard;

fn cpu_cycles() -> Option<u64> {
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn QueryThreadCycleTime(thread: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
        }
        let mut cycles = 0;
        (unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) } != 0).then_some(cycles)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[inline]
pub fn scope(phase: Phase) -> Option<Guard> {
    TIMING.with_borrow_mut(|timing| {
        if !timing.enabled {
            return None;
        }
        timing.stack.push(Frame {
            phase: phase as usize,
            start: Instant::now(),
            children_ns: 0,
            start_cycles: cpu_cycles(),
            children_cycles: Some(0),
        });
        Some(Guard)
    })
}

#[inline]
pub fn measure<T>(phase: Phase, action: impl FnOnce() -> T) -> T {
    let _scope = scope(phase);
    action()
}

impl Drop for Guard {
    fn drop(&mut self) {
        TIMING.with_borrow_mut(|timing| {
            let frame = timing.stack.pop().expect("计时作用域须成对结束");
            let elapsed = frame.start.elapsed().as_nanos() as u64;
            let cycles = frame
                .start_cycles
                .zip(cpu_cycles())
                .map(|(start, end)| end.saturating_sub(start));
            if let Some(parent) = timing.stack.last_mut() {
                parent.children_ns += elapsed;
                parent.children_cycles = parent.children_cycles.zip(cycles).map(|(a, b)| a + b);
            }
            let row = &mut timing.rows[frame.phase];
            if row.calls == 0 {
                row.cpu_cycles = Some((0, 0));
            }
            row.cpu_cycles = row.cpu_cycles.zip(cycles.zip(frame.children_cycles)).map(
                |((total, own), (elapsed, children))| {
                    (total + elapsed, own + elapsed.saturating_sub(children))
                },
            );
            row.calls += 1;
            row.inclusive_ns += elapsed;
            row.exclusive_ns += elapsed.saturating_sub(frame.children_ns);
        });
    }
}

/// 低频状态查询才生成 JSON；仅返回已结束作用域，当前 Inspect 不包含在内。
pub fn snapshot() -> Value {
    TIMING.with_borrow(|timing| {
        if !timing.enabled {
            return Value::Null;
        }
        Value::Object(
            NAMES
                .iter()
                .zip(timing.rows)
                .map(|(name, row)| {
                    (
                        (*name).into(),
                        serde_json::json!({"calls":row.calls,
                            "inclusive_ns":row.inclusive_ns,"exclusive_ns":row.exclusive_ns,
                            "inclusive_cpu_cycles":row.cpu_cycles.map(|c|c.0),
                            "exclusive_cpu_cycles":row.cpu_cycles.map(|c|c.1)}),
                    )
                })
                .collect(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_scopes_partition_elapsed_without_counting_open_parent() {
        initialize(true);
        let outer = scope(Phase::Search);
        measure(Phase::PublicView, || std::hint::black_box(37));
        assert_eq!(snapshot()["search"]["calls"], 0);
        assert_eq!(snapshot()["public_view"]["calls"], 1);
        drop(outer);
        let result = snapshot();
        let parent = &result["search"];
        let child = &result["public_view"];
        assert_eq!(parent["calls"], 1);
        assert_eq!(
            parent["inclusive_ns"].as_u64().unwrap(),
            parent["exclusive_ns"].as_u64().unwrap() + child["inclusive_ns"].as_u64().unwrap()
        );
        assert_eq!(child["inclusive_ns"], child["exclusive_ns"]);
        if let Some(cycles) = parent["inclusive_cpu_cycles"].as_u64() {
            assert_eq!(
                cycles,
                parent["exclusive_cpu_cycles"].as_u64().unwrap()
                    + child["inclusive_cpu_cycles"].as_u64().unwrap()
            );
        }
        initialize(false);
    }

    #[test]
    fn disabled_scopes_have_no_frames_or_rows() {
        initialize(false);
        assert!(scope(Phase::Search).is_none());
        assert_eq!(measure(Phase::RuleApply, || 91), 91);
        TIMING.with_borrow(|timing| {
            assert!(timing.stack.is_empty());
            assert!(timing.rows.iter().all(|row| row.calls == 0));
        });
        assert!(snapshot().is_null());
    }
}
