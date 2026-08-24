//! Block creation: allocate, stamp the header, and initialize refcounts /
//! control blocks. The teardown side lives in [`crate::handle`]; this is the
//! matching build side.
//!
//! These are the low-level, type-agnostic primitives the `#[bstack_block]`
//! macro's generated constructors (and the tests) build on. Writing the
//! type-specific payload — child refs and POD fields after the header — is the
//! caller's job; these helpers only lay down the header and the injected
//! refcount / control machinery at the fixed offsets from [`crate::layout`].

use core::mem::size_of;
use std::error::Error;
use std::fmt;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::{BStack, BStackRange};

use crate::block::BStackWeakable;
use crate::handle::WeakRef;
use crate::layout::{self, BlockHeader, EightCC, put_u64};
use crate::reference::BStackRef;
use crate::replace::ReplaceError;
use crate::shared::{BStackRc, BStackWeak};
use crate::teardown::BStackDrop;

/// The error a generated `new` constructor returns when a fallible construction
/// step fails after it has already consumed the caller's owned/strong/embedded
/// children.
///
/// A generated `new` **consumes** the child handles you hand it (`#[bstack_owned]`,
/// `#[bstack_strong]`, `#[embed]`). A bare `io::Result` would then *lose* those
/// children on an allocation or write failure: their on-disk blocks would be
/// neither linked into the new block, freed, nor returned — an unreachable orphan
/// whose contents (an arbitrarily large subtree the caller just built) are
/// **unrecoverable**. So a failed `new` returns this instead,
/// handing the still-valid children back in [`fields`](Self::fields) — the same
/// region-hand-back contract as bstack's `BStackAllocError`,
/// [`ForeignAllocError`](crate::registry::ForeignAllocError), and
/// [`ReplaceError`](crate::ReplaceError).
///
/// `F` is the block's [`Fields`](crate::block::BStackMove::Fields) tuple — exactly
/// what `bstack_move!` hands back — so a recovered construction returns the
/// children in the same shape a later move would.
///
/// Only constructors that actually consume an owning child return this; a
/// constructor whose fields are all POD / `#[bstack_ref]` (nothing to orphan)
/// keeps a plain `io::Result`. There is deliberately **no**
/// `From<ConstructError> for io::Error`: propagating one with `?` would drop the
/// recovered children and silently re-orphan them, reintroducing the very defect
/// this type exists to prevent. Handle it explicitly, or `.unwrap()` it.
///
/// Implements [`std::error::Error`] (delegating [`Display`](fmt::Display) to
/// [`source`](Self::source)).
pub struct ConstructError<F> {
    /// The underlying I/O error that caused the construction to fail.
    pub source: io::Error,
    /// The children the constructor consumed, handed back if they survived.
    ///
    /// * `Some` — recovered: the children are intact and yours again, in the
    ///   block's `bstack_move!` [`Fields`](crate::block::BStackMove::Fields)
    ///   shape. Retry `new`, re-home them, or free each — dropping them as-is may
    ///   leak, since a bare handle is unrooted (the crate's
    ///   *moved-out-is-unrooted* rule). Every allocation/write failure path takes
    ///   this branch: the children were never touched, only their offsets read.
    /// * `None` — the children could not be handed back here. Generated
    ///   constructors never produce this today (they keep the original handles, so
    ///   recovery is infallible); it exists for parity with
    ///   [`ReplaceError`](crate::ReplaceError) and future fallible-recovery paths.
    pub fields: Option<F>,
}

impl<F> ConstructError<F> {
    /// An error that hands the still-valid children back to the caller.
    #[inline]
    pub fn recovered(source: io::Error, fields: F) -> Self {
        Self {
            source,
            fields: Some(fields),
        }
    }

    /// An error whose children could not be recovered here (see
    /// [`fields`](Self::fields)).
    #[inline]
    pub fn lost(source: io::Error) -> Self {
        Self {
            source,
            fields: None,
        }
    }

    /// Discard the recovered children (if any) and take just the underlying
    /// `io::Error`. Explicit, because dropping recovered children may leak — call
    /// this only when you have decided not to reclaim them.
    #[inline]
    pub fn into_source(self) -> io::Error {
        self.source
    }
}

/// Propagate an inner `io::Error` as a `lost` construction (`fields: None`).
///
/// This is for the generated constructor's own `?` on the fallible *preparation*
/// steps — allocating a child vector's data block, re-encoding a `Foreign`, an
/// `#[embed]` copy — that run before (or between) the block's own allocation and
/// write. Those faults are rare mid-construction I/O errors; degrading them to
/// `lost` matches the pre-existing behaviour (the children a partial prep already
/// consumed were orphaned there too) while the *primary* allocation / write /
/// commit failures still hand the children back through
/// [`recovered`](ConstructError::recovered).
///
/// This affects only `?` **inside** a constructor (an `io::Error` becoming a
/// `ConstructError`); it deliberately does **not** provide the reverse
/// (`From<ConstructError> for io::Error`), so a caller cannot silently `?` a
/// failed `new` and re-orphan the returned children.
impl<F> From<io::Error> for ConstructError<F> {
    #[inline]
    fn from(source: io::Error) -> Self {
        Self::lost(source)
    }
}

// Manual, so `F` need not be `Debug` (the handed-back handles generally aren't).
impl<F> fmt::Debug for ConstructError<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstructError")
            .field("source", &self.source)
            .field("fields", &self.fields.as_ref().map(|_| "..."))
            .finish()
    }
}

impl<F> fmt::Display for ConstructError<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.source, f)
    }
}

impl<F> Error for ConstructError<F> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

#[inline(always)]
fn read_u64_at(stack: &BStack, off: u64) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    stack.get_into(off, &mut buf)?;
    Ok(layout::get_u64(&buf))
}

/// Allocate a `size`-byte block and stamp its `BlockHeader { size, tag }`.
///
/// Returns the block's range. The bytes after the header are left as the
/// allocator provided them; the caller fills in the payload. On a write failure
/// the freshly allocated block is released so nothing leaks.
/// Crate-internal, test-only: a public block-minting primitive would let safe code
/// stamp any type's tag over any size — the exact credential every header-trusting
/// gate (`bstack_cast!`, `AnyRef::from_block`, `verify_data_block`) validates. No
/// non-test code path needs it; generated constructors stamp their header inline.
#[cfg(test)]
pub(crate) fn alloc_block<A: BStackRaiiAllocator>(
    allocator: &A,
    tag: EightCC,
    size: u64,
) -> io::Result<BStackRange> {
    let mut slice = allocator.alloc(size)?;
    let header = BlockHeader { size, tag };
    if let Err(e) = slice.write_range(0, bytemuck::bytes_of(&header)) {
        let _ = allocator.dealloc(slice);
        return Err(e);
    }
    Ok(slice.as_range())
}

/// Build a `(rc, weak)` control-block payload image in memory (no allocation, no
/// write): header, `strong = 1`, `weak = 1` (the phantom weak the strong owners
/// hold), and the `x` forward pointer to the data block at `data_start`.
///
/// The building block for a **batched** constructor: the caller allocates the
/// data and control blocks up front, bakes the control offset into the data
/// block's `ctrl` back-pointer, and commits both block images in one
/// [`bstack::BStack::set_batched`] — so a `(rc, weak)` block is created atomically,
/// with no separate back-pointer write and no transient half-wired state.
pub fn build_control_payload(ctrl_tag: EightCC, data_start: u64, control_size: u64) -> Vec<u8> {
    let mut payload = vec![0u8; control_size as usize];
    let header = BlockHeader {
        size: control_size,
        tag: ctrl_tag,
    };
    payload[..layout::HEADER_SIZE as usize].copy_from_slice(bytemuck::bytes_of(&header));
    put_u64(&mut payload, layout::CTRL_STRONG_OFFSET, 1);
    put_u64(&mut payload, layout::CTRL_WEAK_OFFSET, 1);
    put_u64(&mut payload, layout::CTRL_DATA_OFFSET, data_start);
    payload
}

/// Set a `#[bstack_weak]` field, located at absolute on-disk offset `field_off`,
/// to point at `new_weak` — releasing any weak reference the field previously
/// held.
///
/// The field stores the child's **control-block** offset, not its data offset:
/// the control block outlives the data block (it lives while `weak > 0`), so
/// resolving it at teardown is sound even after the target's data has been
/// freed. `new_weak` is consumed and the weak count it holds becomes the field's;
/// a previous non-null target has its weak count decremented. 0 means "unset".
///
/// # Safety
///
/// `field_off` must be the absolute offset of a live `#[bstack_weak]` field of
/// declared target type `T`, owned by a block in `allocator`'s file. The old
/// value read from it is released as a control-block reference: a wrong offset
/// decrements (and can free) a control block at whatever offset that location
/// happens to hold.
pub unsafe fn set_weak_field<'w, T: BStackWeakable, A: BStackRaiiAllocator>(
    allocator: &'w A,
    field_off: u64,
    new_weak: BStackWeak<'w, T, A>,
) -> Result<(), ReplaceError<BStackWeak<'w, T, A>>> {
    // Serialize against a concurrent `upgrade_weak_field` on the same field: the
    // old control block is released (and possibly freed) below, and a racing
    // upgrade — which holds no weak count to pin it — would otherwise increment a
    // counter in freed storage. Both take this per-file lock.
    let lock = crate::wal::wal_lock_for(allocator);
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let stack = allocator.stack();

    // Exchange the new pointer for the old one in a single atomic `swap`: the read
    // of the old control offset and the write of the new happen together under one
    // lock, so two concurrent setters each take (and release) the distinct old
    // control block they displaced — never the same one twice.
    // `new_weak` is consumed without decrementing — its weak count becomes the
    // field's.
    let ctrl = new_weak.into_raw();
    let ctrl_off = ctrl.into_range().start();
    let old_bytes = match stack.swap(field_off, ctrl_off.to_le_bytes()) {
        Ok(b) => b,
        Err(e) => {
            // Commit failed: the field still points at the old target, and
            // `new_weak` was already consumed (`into_raw` defused its decrement).
            // Hand it back rather than release its just-transferred weak count, so
            // the caller can retry or release at their discretion.
            // SAFETY: `ctrl_off` carries the weak count `new_weak` transferred in.
            let weak = unsafe {
                BStackWeak::from_raw(
                    BStackRef::<T::Control>::from_range(BStackRange::new(
                        ctrl_off,
                        size_of::<T::Control>() as u64,
                    )),
                    allocator,
                )
            };
            return Err(ReplaceError::recovered(e, weak));
        }
    };
    let old = u64::from_le_bytes(old_bytes[..8].try_into().unwrap());

    // Only now release the old target — pure reclamation, since the field no
    // longer refers to it. A crash before this leaks at most the old control
    // block (its weak count stays one too high), never a dangling field.
    if old != 0 {
        let old_ctrl = unsafe {
            BStackRef::<T::Control>::from_range(BStackRange::new(
                old,
                size_of::<T::Control>() as u64,
            ))
        };
        // SAFETY: `old_ctrl` carries the weak count the field held until the
        // commit above displaced it.
        if let Err(e) = unsafe { WeakRef::<T>::new(old_ctrl) }.bstack_drop(allocator) {
            // The new weak is already installed (the swap committed); only the old
            // target's weak-count release failed, leaving it one-too-high — the
            // leak teardown always tolerates. Nothing is handed back (`lost`): the
            // new value is in the field, and the caller cannot re-drive this.
            return Err(ReplaceError::lost(e));
        }
    }
    Ok(())
}

/// Attempt to upgrade a `#[bstack_weak]` field (holding a control-block offset at
/// `field_off`) to a strong handle. Returns `None` if the field is unset (0) or
/// the target's strong count has already reached zero. What a generated weak
/// field accessor calls.
///
/// # Safety
///
/// `field_off` must be the absolute offset of a live `#[bstack_weak]` field of
/// declared target type `T` in `allocator`'s file: the u64 read there is
/// treated as a control-block offset and its counters are read and written —
/// a wrong offset manufactures an owning `BStackRc` from arbitrary bytes.
pub unsafe fn upgrade_weak_field<'a, T: BStackWeakable, A: BStackRaiiAllocator>(
    allocator: &'a A,
    field_off: u64,
) -> io::Result<Option<BStackRc<'a, T, A>>> {
    // Hold the per-file lock across the read of the control offset and the pin
    // (`increment_if_nonzero`), so a concurrent `set_weak_field` can't free the old
    // control block between the two steps. The field slot is not
    // owned here, so nothing else keeps that block alive.
    let lock = crate::wal::wal_lock_for(allocator);
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let off = read_u64_at(allocator.stack(), field_off)?;
    if off == 0 {
        return Ok(None);
    }
    let ctrl = unsafe {
        BStackRef::<T::Control>::from_range(BStackRange::new(off, size_of::<T::Control>() as u64))
    };
    // Borrow a weak over the field's control ref just long enough to upgrade;
    // consume it via `into_raw` so the field's own weak count is untouched.
    let weak = unsafe { BStackWeak::from_raw(ctrl, allocator) };
    let result = weak.upgrade();
    let _ = weak.into_raw();
    result
}
