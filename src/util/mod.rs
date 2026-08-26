//! Small crate-internal utilities with no home of their own.

pub(crate) mod io;
pub(crate) mod small_buf;
pub(crate) mod small_map;

// `io_error` is exported for direct use at call sites (dynamic messages); adoption
// beyond `io_errorfn` is incremental, so it may be momentarily unused.
#[allow(unused_imports)]
pub(crate) use io::{io_error, io_errorfn};
