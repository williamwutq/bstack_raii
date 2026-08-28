//! Fixed-width little-endian on-disk `u64` codec — the crate's one place for
//! reading and writing 8-byte integer fields, by hand into a buffer or straight
//! from a [`BStack`].
//!
//! Every on-disk multi-byte integer (`bstack`'s own format, refcounts, offsets,
//! `ctrl` back-pointers, `Foreign` targets, linked-structure next/prev/child
//! slots) is little-endian; these helpers centralize that convention instead of
//! repeating `to_le_bytes` / `from_le_bytes` at every image builder and reader.

use std::io;

use bstack::BStack;

/// Write `val` as a little-endian `u64` at byte offset `off` in `buf`.
///
/// The one place the crate builds on-disk integer fields by hand, instead of
/// repeating `copy_from_slice(&x.to_le_bytes())` at every image builder.
#[inline(always)]
pub fn put_u64(buf: &mut [u8], off: u64, val: u64) {
    let o = off as usize;
    buf[o..o + 8].copy_from_slice(&val.to_le_bytes());
}

/// Read a little-endian `u64` from the first 8 bytes of `buf`.
///
/// This centralizes the crate's fixed-width on-disk `u64` decode pattern.
#[inline(always)]
pub fn get_u64(buf: &[u8]) -> u64 {
    u64::from_le_bytes(buf[..8].try_into().unwrap())
}

/// Read a little-endian `u64` directly from the block at `off` — the crate's
/// fixed-width on-disk `u64` load (an 8-byte `get_into` fed through [`get_u64`]).
/// Every on-disk pointer/count field (`ctrl` back-pointers, `Foreign` targets,
/// linked-structure offsets, refcounts) is decoded through this.
#[inline(always)]
pub fn read_u64(stack: &BStack, off: u64) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    stack.get_into(off, &mut buf)?;
    Ok(get_u64(&buf))
}
