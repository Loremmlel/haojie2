//! 独立诊断构建：分项独占时间与分配累计量，禁止用该构建报告正式倍率。
use serde_json::{Value, json};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::RefCell,
    sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
    time::Instant,
};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
struct Counting;
#[global_allocator]
static ALLOCATOR: Counting = Counting;
// 只透传系统分配器；不改变对齐、布局、失败值或内存所有权。
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && ACTIVE.load(Relaxed) {
            COUNTS[0].fetch_add(1, Relaxed);
            COUNTS[1].fetch_add(layout.size() as u64, Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if ACTIVE.load(Relaxed) {
            COUNTS[2].fetch_add(1, Relaxed);
            COUNTS[3].fetch_add(layout.size() as u64, Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(ptr, layout, size) };
        if !next.is_null() && ACTIVE.load(Relaxed) {
            COUNTS[0].fetch_add(1, Relaxed);
            COUNTS[1].fetch_add(size as u64, Relaxed);
            COUNTS[2].fetch_add(1, Relaxed);
            COUNTS[3].fetch_add(layout.size() as u64, Relaxed);
        }
        next
    }
}
#[derive(Clone, Copy)]
pub enum Phase {
    Kernel,
    Observe,
    Candidates,
    Preparation,
    Attributes,
    Paths,
    Branch,
    Settlement,
    Preview,
    Encoding,
    Hash,
    Protocol,
    UnitCopy,
    Identity,
    Compatibility,
    StateImport,
    StateExport,
    Policy,
    Events,
}
const NAMES: [&str; 19] = [
    "kernel",
    "observe",
    "candidates",
    "preparation",
    "attributes",
    "paths",
    "branch",
    "settlement",
    "preview",
    "encoding",
    "hash",
    "protocol",
    "unit_copy",
    "identity",
    "compatibility",
    "state_import",
    "state_export",
    "policy",
    "events",
];
const COUNT: usize = NAMES.len();
#[derive(Clone, Copy, Default)]
struct Row {
    calls: u64,
    inclusive: f64,
    own: f64,
    inclusive_allocations: [u64; 4],
    own_allocations: [u64; 4],
}
struct Frame {
    phase: usize,
    started: Instant,
    counts: [u64; 4],
    children: f64,
    child_counts: [u64; 4],
}
thread_local! {
    static ROWS: RefCell<[Row;COUNT]> = const { RefCell::new([Row { calls:0,inclusive:0.0,own:0.0,inclusive_allocations:[0;4],own_allocations:[0;4] };COUNT]) };
    static STACK: RefCell<Vec<Frame>> = RefCell::new(Vec::with_capacity(128));
}
fn counts() -> [u64; 4] {
    std::array::from_fn(|i| COUNTS[i].load(Relaxed))
}
pub struct Guard;
pub fn scope(phase: Phase) -> Option<Guard> {
    if !ACTIVE.load(Relaxed) {
        return None;
    }
    STACK.with_borrow_mut(|s| {
        s.push(Frame {
            phase: phase as usize,
            started: Instant::now(),
            counts: counts(),
            children: 0.0,
            child_counts: [0; 4],
        })
    });
    Some(Guard)
}
impl Drop for Guard {
    fn drop(&mut self) {
        STACK.with_borrow_mut(|s| {
            let f = s.pop().expect("成对的诊断作用域");
            let elapsed = f.started.elapsed().as_secs_f64() * 1000.0;
            let now = counts();
            let allocated: [u64; 4] = std::array::from_fn(|i| now[i] - f.counts[i]);
            if let Some(parent) = s.last_mut() {
                parent.children += elapsed;
                for (i, n) in allocated.iter().enumerate() {
                    parent.child_counts[i] += n;
                }
            }
            ROWS.with_borrow_mut(|rows| {
                let row = &mut rows[f.phase];
                row.calls += 1;
                row.inclusive += elapsed;
                row.own += elapsed - f.children;
                for (i, n) in allocated.iter().enumerate() {
                    row.inclusive_allocations[i] += n;
                    row.own_allocations[i] += n - f.child_counts[i];
                }
            });
        });
    }
}
pub fn start() {
    ACTIVE.store(false, Relaxed);
    STACK.with_borrow_mut(|s| {
        assert!(s.is_empty());
        s.reserve(128);
    });
    ROWS.with_borrow_mut(|rows| *rows = [Row::default(); COUNT]);
    for c in &COUNTS {
        c.store(0, Relaxed);
    }
    ACTIVE.store(true, Relaxed);
}
pub fn finish() -> Value {
    ACTIVE.store(false, Relaxed);
    STACK.with_borrow(|s| assert!(s.is_empty()));
    ROWS.with_borrow(|rows|json!({
        "rows":NAMES.iter().zip(rows).map(|(name,r)|json!({"name":name,"calls":r.calls,"inclusiveMs":r.inclusive,"selfMs":r.own,"inclusiveAllocations":r.inclusive_allocations,"selfAllocations":r.own_allocations})).collect::<Vec<_>>(),
        "allocations":counts(),
        "note":"分配数组依次为分配次数/字节、释放次数/字节，包含 realloc；累计量不是驻留峰值。时间含诊断开销，不用于倍率。"
    }))
}
