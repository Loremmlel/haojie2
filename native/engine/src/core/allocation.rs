//! 分配后端只改变语言底层内存管理；规则、采样与外部协议共用原实现。
//! 诊断构建由 Counting 唯一注册全局分配器，再透传这里同一份后端。

#[cfg(feature = "allocator-mimalloc")]
pub static RAW: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(not(feature = "allocator-mimalloc"))]
pub static RAW: std::alloc::System = std::alloc::System;

#[cfg(not(feature = "kernel-profile"))]
mod live {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    pub static BYTES: AtomicUsize = AtomicUsize::new(0);
    struct Allocator;
    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;
    // 请求容量随所有权跨线程转移仍准确；不统计分配器自身的驻留碎片。
    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { super::RAW.alloc(layout) };
            if !p.is_null() {
                BYTES.fetch_add(layout.size(), Relaxed);
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
            unsafe { super::RAW.dealloc(p, layout) };
            BYTES.fetch_sub(layout.size(), Relaxed);
        }
        unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let next = unsafe { super::RAW.realloc(p, layout, size) };
            if !next.is_null() {
                if size >= layout.size() {
                    BYTES.fetch_add(size - layout.size(), Relaxed);
                } else {
                    BYTES.fetch_sub(layout.size() - size, Relaxed);
                }
            }
            next
        }
    }
}

#[cfg(not(feature = "kernel-profile"))]
pub fn live_bytes() -> usize {
    live::BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

pub const NAME: &str = if cfg!(feature = "allocator-mimalloc") {
    "mimalloc"
} else {
    "system"
};
