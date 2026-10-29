//! 普通局分层解码与 Python 进程内桥接；规则仍由原内核执行。
pub(crate) mod decoder;
pub(crate) mod encoding;
mod ffi;
mod parameters;
pub(crate) mod pool;
pub(crate) mod random;
pub(crate) mod search;
mod timing;
