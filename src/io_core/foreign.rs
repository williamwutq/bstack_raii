//! Cross-file teardown & deep-clone mechanism for `Foreign<T>` fields — the
//! cross-file counterparts of [`teardown`](crate::io_core::teardown) /
//! [`clone`](crate::io_core::clone).
//!
//! Each helper runs a `Foreign<T>` field's per-kind teardown or clone in *whichever
//! file the target lives in*, selected entirely by the `alloc` handed in: the local
//! allocator for a `SELF` target, or a
//! [`ForeignHostAllocator`](crate::registry::ForeignHostAllocator) for a cross-file
//! one. Because the whole teardown/clone machinery (`OwnedRef` /
//! `BStackShared::drop_strong_ref` / `WeakRef`, `try_clone_in`, and the recursive
//! `__bstack_drop_children`) is generic over `A: BStackRaiiAllocator`, the same code
//! frees or copies the target in its own file with no duplication — reads/writes go
//! through `alloc.stack()`, and frees are tagged with `alloc.wal_file_id()`.
//!
//! The `BStackBlock` / `BStackShared` / `BStackWeakable` `foreign_drop_*` /
//! `foreign_clone_*` trait methods delegate straight here; the generated field code
//! picks the helper by the field's annotation (`#[bstack_ref]` has none — a foreign
//! ref owns nothing).

use std::io;

use bstack::BStackRange;

use crate::BStackRaiiAllocator;
use crate::io_core::{TryCloneIn, refcount};
use crate::primitives::NonNullOffset;
use crate::types::compiled::rc::{CTRL_WEAK_OFFSET, strong_counter_off};
use crate::types::compiled::{OwnedRef, WeakRef};
use crate::types::traits::{BStackBlock, BStackDrop, BStackRef, BStackShared, BStackWeakable};

/// Tear down an `#[bstack_owned] Foreign<T>` target: free the block at `offset`
/// (and, recursively, its own children) in the file `alloc` addresses.
///
/// # Safety
/// `offset` names a live `T` block, in the file `alloc` addresses, exclusively owned
/// by this foreign pointer (freed exactly once).
pub(crate) unsafe fn foreign_drop_owned<T: BStackBlock, A: BStackRaiiAllocator>(
    alloc: &A,
    offset: NonNullOffset,
) -> io::Result<()> {
    let range = BStackRange::new(offset.as_u64(), core::mem::size_of::<T::OnDisk>() as u64);
    // SAFETY: `range` is the caller-asserted live block of `T`.
    let child = unsafe { BStackRef::<T>::from_range(range) };
    // SAFETY: the caller (an owning foreign slot's teardown) owns `child`.
    unsafe { OwnedRef::new(child) }.bstack_drop(alloc)
}

/// Tear down an `#[bstack_strong] Foreign<T>` target: decrement the strong count at
/// `offset` (the target's *data* block) and, at zero, free it in the file `alloc`
/// addresses. `T` must be a shared block.
///
/// # Safety
/// `offset` names a live shared `T` data block in the file `alloc` addresses, holding
/// one strong reference on behalf of this foreign pointer.
pub(crate) unsafe fn foreign_drop_strong<T: BStackShared, A: BStackRaiiAllocator>(
    alloc: &A,
    offset: NonNullOffset,
) -> io::Result<()> {
    let range = BStackRange::new(offset.as_u64(), core::mem::size_of::<T::OnDisk>() as u64);
    // SAFETY: `range` is the caller-asserted live data block of a shared `T`.
    let data = unsafe { BStackRef::<T>::from_range(range) };
    <T as BStackShared>::drop_strong_ref(data, alloc)
}

/// Tear down a `#[bstack_weak] Foreign<T>` target: decrement the weak count in the
/// *control* block at `ctrl_offset` and, at zero, free the control block in the file
/// `alloc` addresses. The data block is never touched. `T` must be weakable, and the
/// foreign pointer stores the **control** offset (as an in-file weak field does).
///
/// # Safety
/// `ctrl_offset` names a live `T::Control` block in the file `alloc` addresses,
/// holding one weak reference on behalf of this foreign pointer.
pub(crate) unsafe fn foreign_drop_weak<T: BStackWeakable, A: BStackRaiiAllocator>(
    alloc: &A,
    ctrl_offset: NonNullOffset,
) -> io::Result<()> {
    let range = BStackRange::new(
        ctrl_offset.as_u64(),
        core::mem::size_of::<T::Control>() as u64,
    );
    // SAFETY: `range` is the caller-asserted live control block of a weakable `T`.
    let ctrl = unsafe { BStackRef::<T::Control>::from_range(range) };
    // SAFETY: the weak foreign slot being torn down held this weak count.
    unsafe { WeakRef::<T>::new(ctrl) }.bstack_drop(alloc)
}

// ---------------------------------------------------------------------------
// Cross-file deep-clone helpers (the mirror of the teardown helpers above).
//
// A `Foreign<T>` field's clone acts on the target *in the target's own file* per its
// annotation: `#[bstack_owned]` deep-copies the target (a fresh block on that file,
// the pointer repointed); `#[bstack_strong]` / `#[bstack_weak]` share the same target
// and bump its count; `#[bstack_ref]` aliases (byte-copied — no helper). A `SELF`
// pointer instead folds into the *home* clone plan (atomic with the home commit); the
// generated code picks that path. These helpers cover the cross-file case.
//
// **Atomicity across files is best-effort (option-1):** the foreign side is touched
// eagerly (its own commit for `owned`, an atomic increment for `strong`/`weak`) BEFORE
// the home clone commits, and a home failure afterwards does not undo it. That always
// errs toward **over-provisioning** — an orphaned fresh block, or an over-count — which
// leaks, never toward an under-count (which would be a premature free / double-free).
// A target file that is not currently attached makes the clone **error** (aliasing an
// owning pointer would create a second owner ⇒ later double-free); the generated code
// enforces that before calling these.
// ---------------------------------------------------------------------------

/// Deep-clone an `#[bstack_owned] Foreign<T>` target at `offset` in the file `alloc`
/// addresses; returns the new copy's offset. Self-contained atomic commit on that
/// file; eager (a later home-clone failure leaks the new block).
///
/// # Safety
/// `offset` names a live `T` block, in the file `alloc` addresses, owned by this
/// foreign pointer.
pub(crate) unsafe fn foreign_clone_owned<T: TryCloneIn + BStackBlock, A: BStackRaiiAllocator>(
    alloc: &A,
    offset: NonNullOffset,
) -> io::Result<u64> {
    let range = BStackRange::new(offset.as_u64(), core::mem::size_of::<T::OnDisk>() as u64);
    let src = unsafe { T::from_range(range) };
    let new = src.try_clone_in(alloc)?;
    Ok(new.handle().range().start())
}

/// Bump the strong count of an `#[bstack_strong] Foreign<T>` target at `offset` (its
/// data block) in the file `alloc` addresses — the strong reference the clone
/// acquires. Eager atomic increment.
///
/// # Safety
/// `offset` names a live shared `T` data block in the file `alloc` addresses.
pub(crate) unsafe fn foreign_clone_strong<T: BStackShared, A: BStackRaiiAllocator>(
    alloc: &A,
    offset: NonNullOffset,
) -> io::Result<()> {
    let range = BStackRange::new(offset.as_u64(), core::mem::size_of::<T::OnDisk>() as u64);
    // SAFETY: `range` is the caller-asserted live data block of a shared `T`.
    let data = unsafe { BStackRef::<T>::from_range(range) };
    let (data_ref, ctrl) = <T as BStackShared>::strong_parts(data, alloc)?;
    let off = strong_counter_off(data_ref.into_range().start(), ctrl)?;
    refcount::fetch_add(alloc.stack(), NonNullOffset::from_field(off)?, 1)?;
    Ok(())
}

/// Bump the weak count of a `#[bstack_weak] Foreign<T>` target's control block at
/// `ctrl_offset` in the file `alloc` addresses — the weak reference the clone
/// acquires. Eager atomic increment. (`T` documents the intended weakable target; the
/// increment needs only the offset.)
///
/// # Safety
/// `ctrl_offset` names a live `T::Control` block in the file `alloc` addresses.
// `T` documents the target and keeps the signature parallel to the other three
// helpers (and to `BStackWeakable::foreign_clone_weak`); the increment is offset-only.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) unsafe fn foreign_clone_weak<T: BStackWeakable, A: BStackRaiiAllocator>(
    alloc: &A,
    ctrl_offset: NonNullOffset,
) -> io::Result<()> {
    refcount::fetch_add(alloc.stack(), ctrl_offset.checked_add(CTRL_WEAK_OFFSET)?, 1)?;
    Ok(())
}
