//! Atomic operations on on-disk `u64` counters, built on [`bstack::BStack`].
//!
//! Every counter is stored **little-endian** (fixed by the `bstack` ABI). Each
//! function takes the absolute payload offset of the counter within the stack.
//!
//! The read-modify-write helpers use [`BStack::process`], which reads the
//! counter, runs a closure to mutate it in place, and writes it back — all under
//! one held write lock, crash-atomically. That does the whole RMW in a single
//! lock acquisition with **no compare-and-swap spin loop**. (`BStack::cas` would
//! also be correct here, since each counter's dangerous value — zero — is a sink
//! state and so immune to ABA; `process` is preferred purely to avoid retrying.)
//!
//! Because `process`'s closure returns `()` and always writes the buffer back,
//! the error paths (overflow / underflow) signal out through a captured flag and
//! leave the buffer *unchanged*, so the write-back is a no-op on those paths.

use std::io;

use bstack::BStack;

use crate::primitives::NonNullOffset;
use crate::util::bytes::get_u64;
use crate::util::io_errorfn;

io_errorfn!(overflow_err, InvalidData, "refcount overflow");
io_errorfn!(underflow_err, InvalidData, "refcount underflow");

// A counter offset near `u64::MAX` (so the fixed 8-byte counter range can't be
// formed) can only come from a corrupted/forged on-disk pointer — every caller
// derives `offset` from a stored back-pointer, `Foreign` target, or field value.
io_errorfn!(corrupt_offset_err, InvalidData, "refcount offset overflow");

/// Compare-and-swap the counter at `offset`: set it to `new` iff it currently
/// equals `expected`. Returns whether the swap happened. The atomic "try-unwrap"
/// primitive behind [`crate::BStackRc::try_move`].
#[inline(always)]
pub fn cas(stack: &BStack, offset: NonNullOffset, expected: u64, new: u64) -> io::Result<bool> {
    stack.cas(offset.as_u64(), expected.to_le_bytes(), new.to_le_bytes())
}

/// Load the current value of the counter at `offset` (little-endian). Read-only,
/// so it takes only `get_into` (no lock upgrade, no write-back).
#[inline(always)]
pub fn load(stack: &BStack, offset: NonNullOffset) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    stack.get_into(offset.as_u64(), &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Atomically add `delta`, returning the previous value. Errors on overflow
/// rather than wrapping (leaving the counter unchanged in that case).
pub fn fetch_add(stack: &BStack, offset: NonNullOffset, delta: u64) -> io::Result<u64> {
    let offset = offset.as_u64();
    let end = offset.checked_add(8).ok_or_else(corrupt_offset_err)?;
    let mut prev = 0u64;
    let mut overflow = false;
    stack.process(offset, end, |buf| {
        let cur = get_u64(buf);
        prev = cur;
        match cur.checked_add(delta) {
            Some(new) => buf.copy_from_slice(&new.to_le_bytes()),
            None => overflow = true, // leave buf unchanged; report below
        }
    })?;
    if overflow {
        return Err(overflow_err());
    }
    Ok(prev)
}

/// Atomically subtract `delta`, returning the previous value. Errors on
/// underflow rather than wrapping (leaving the counter unchanged in that case).
pub fn fetch_sub(stack: &BStack, offset: NonNullOffset, delta: u64) -> io::Result<u64> {
    let offset = offset.as_u64();
    let end = offset.checked_add(8).ok_or_else(corrupt_offset_err)?;
    let mut prev = 0u64;
    let mut underflow = false;
    stack.process(offset, end, |buf| {
        let cur = get_u64(buf);
        prev = cur;
        match cur.checked_sub(delta) {
            Some(new) => buf.copy_from_slice(&new.to_le_bytes()),
            None => underflow = true, // leave buf unchanged; report below
        }
    })?;
    if underflow {
        return Err(underflow_err());
    }
    Ok(prev)
}

/// Increment the counter only if it is currently non-zero, returning the new
/// value on success or `None` if it was zero. The primitive behind
/// [`crate::BStackWeak::upgrade`]: it must never resurrect a counter that a
/// concurrent drop has already driven to zero.
///
/// A read-only fast path returns `None` without any write when the counter is
/// already zero (the common "the object is long dead" case); zero is terminal,
/// so that observation is authoritative. When the fast path sees non-zero, the
/// `process` closure re-checks under the lock — the value may have raced to zero
/// in between — before committing the increment.
pub fn increment_if_nonzero(stack: &BStack, offset: NonNullOffset) -> io::Result<Option<u64>> {
    if load(stack, offset)? == 0 {
        return Ok(None);
    }
    let offset = offset.as_u64();
    let end = offset.checked_add(8).ok_or_else(corrupt_offset_err)?;
    let mut result = None;
    let mut overflow = false;
    stack.process(offset, end, |buf| {
        let cur = get_u64(buf);
        if cur == 0 {
            return; // raced to zero after the fast-path read; leave unchanged
        }
        match cur.checked_add(1) {
            Some(new) => {
                buf.copy_from_slice(&new.to_le_bytes());
                result = Some(new);
            }
            None => overflow = true,
        }
    })?;
    if overflow {
        return Err(overflow_err());
    }
    Ok(result)
}
