//! 同一规则内核的进程内入口；环境对象只在创建它们的原生线程上使用。
include!("modules.rs");
#[path = "online/runtime.rs"]
mod online;
