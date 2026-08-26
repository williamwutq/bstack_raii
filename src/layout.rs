//! Overflow-checked block-offset arithmetic.
//!
//! The single helper every on-disk-derived offset computation funnels through, so
//! a corrupt or forged base pointer can never silently wrap to an unrelated
//! in-bounds address. The on-disk header shape ([`BlockHeader`](crate::types::compiled::block::BlockHeader))
//! and the injected refcount / control field offsets live under
//! [`crate::types::compiled`].

use std::io;

use crate::util::io_errorfn;

io_errorfn!(block_offset_overflow, InvalidData, "block offset overflow");

/// Add a small field-offset constant (`RC_REFCOUNT_OFFSET`/`CTRL_*_OFFSET`, a
/// stdlib collection's own `N*_OFF` node-field constants, …) to a base offset,
/// rejecting overflow. The base routinely originates from an on-disk pointer (a
/// `ctrl` back-pointer, a `Foreign` target, a linked structure's stored
/// next/prev/child offset) that can be corrupted or forged, so plain `+` would
/// either panic under `overflow-checks` or silently wrap to an unrelated
/// in-bounds offset that a later read/write would then corrupt.
#[inline(always)]
pub fn checked_off(base: u64, delta: u64) -> io::Result<u64> {
    base.checked_add(delta).ok_or_else(block_offset_overflow)
}
