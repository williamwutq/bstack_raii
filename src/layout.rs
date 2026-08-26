//! The on-disk block header and the shared offset-arithmetic / integer helpers.
//!
//! Both are [`bytemuck::Pod`] so they can be embedded directly in a generated
//! `XOnDisk` struct and read back with `bytemuck::from_bytes`.

use std::io;

use bytemuck::{Pod, Zeroable};

use crate::primitives::EightCC;
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

/// The header prefixing every on-disk block. 16 bytes.
///
/// `size` is the payload length in bytes; `tag` is the [`EightCC`] discriminant
/// written by the allocator at block creation. Declared `#[repr(C)]` rather than
/// `#[repr(C, packed)]`: a `u64` followed by an 8-byte tag is already densely
/// packed with no padding, and avoiding `packed` keeps field access sound. The
/// *generated* `XOnDisk` structs that embed this and then mix in smaller POD
/// fields are the ones that need `packed`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlockHeader {
    pub size: u64,
    pub tag: EightCC,
}

/// Byte length of a [`BlockHeader`] — the offset at which a block's payload
/// begins.
pub const HEADER_SIZE: u64 = core::mem::size_of::<BlockHeader>() as u64;

// -- Injected-field offsets ------------------------------------------------
//
// The macros inject the refcount / control back-pointer / control counters
// immediately after the header, ahead of any user fields and in a fixed order.
// Their offsets are therefore the same for *every* block, so they live here as
// constants rather than as per-type trait members.

/// `#[bstack_block(rc)]` data block: offset of the inline `refcount: AtomicU64`,
/// injected right after the header.
///
/// ```text
/// struct XOnDisk { header, refcount: AtomicU64, <user fields...> }
/// ```
pub const RC_REFCOUNT_OFFSET: u64 = HEADER_SIZE;

/// `#[bstack_block(rc, weak)]` data block: offset of the `ctrl` back-pointer to
/// the control block, injected right after the header.
///
/// ```text
/// struct XOnDisk { header, ctrl: BStackRef<XOnDiskRef>, <user fields...> }
/// ```
pub const CTRL_BACKPTR_OFFSET: u64 = HEADER_SIZE;

/// `#[bstack_block(rc, weak)]` control block (`XOnDiskRef`): offset of `strong`.
///
/// ```text
/// struct XOnDiskRef { header, strong: AtomicU64, weak: AtomicU64, x: BStackRef<X> }
/// ```
pub const CTRL_STRONG_OFFSET: u64 = HEADER_SIZE;

/// Control block: offset of `weak` (starts at 1 — the phantom weak held
/// collectively by all live strong owners).
pub const CTRL_WEAK_OFFSET: u64 = HEADER_SIZE + 8;

/// Control block: offset of `x`, the forward pointer back to the data block.
/// Read by [`crate::BStackWeak::upgrade`] once it wins the strong CAS.
pub const CTRL_DATA_OFFSET: u64 = HEADER_SIZE + 16;

// Guard the hand-derived offsets against a header size change.
const _: () = assert!(HEADER_SIZE == 16);
