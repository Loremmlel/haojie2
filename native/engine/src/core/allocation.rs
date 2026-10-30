//! 分配后端只改变语言底层内存管理；规则、采样与外部协议共用原实现。

#[cfg(feature = "allocator-mimalloc")]
#[global_allocator]
pub static RAW: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(not(feature = "allocator-mimalloc"))]
#[global_allocator]
pub static RAW: std::alloc::System = std::alloc::System;

pub const NAME: &str = if cfg!(feature = "allocator-mimalloc") {
    "mimalloc"
} else {
    "system"
};
