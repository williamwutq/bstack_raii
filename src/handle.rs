//! Without-allocator inner handles: small `Copy` types constructed transiently
//! during teardown, each encapsulating one field annotation's destruction logic.
//!
//! Keeping the per-annotation logic here (rather than in generated block code)
//! means `#[bstack_block]` can emit a flat, uniform sequence of
//! `.bstack_drop(allocator)?` calls. The with-allocator wrappers
//! ([`crate::BStackOwned`], [`crate::BStackRc`], [`crate::BStackWeak`]) hold one
//! of these plus an allocator reference.
//!
//! All the layout facts these teardowns need are constants in [`crate::layout`]
//! (the injected refcount / control fields sit at fixed offsets after the
//! header) plus the `OnDisk` / `Control` sizes from
//! [`BStackBlock`] / [`BStackWeakable`]. No per-type layout members are
//! required.

use core::mem::size_of;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::BStackRange;

use crate::block::{BStackBlock, BStackWeakable};
use crate::layout;
use crate::refcount;
use crate::reference::BStackRef;
use crate::teardown::{BStackDrop, dealloc_range};

/// `#[bstack_owned]`: an exclusively-owned child.
///
/// Not `Copy`/`Clone`, and its field is private: this is an *ownership* token
/// whose [`BStackDrop`] frees unconditionally, so it must not be freely
/// mintable or duplicable from a non-owning [`BStackRef`] — construct it with
/// the `unsafe` [`new`](Self::new).
pub struct OwnedRef<T>(BStackRef<T>);

impl<T> OwnedRef<T> {
    /// # Safety
    ///
    /// `inner` must reference a live block the caller exclusively owns (and
    /// gives up by constructing this): the wrapper's `bstack_drop` frees the
    /// block outright, ignoring any refcount.
    pub unsafe fn new(inner: BStackRef<T>) -> Self {
        Self(inner)
    }
}

/// `#[bstack_strong]` on a plain `(rc)` `T`: holds just the data ref; teardown
/// decrements the inline refcount and frees at zero.
///
/// The macro only emits this for children whose type is `#[bstack_block(rc)]`,
/// so the inline `refcount` at [`layout::RC_REFCOUNT_OFFSET`] is guaranteed
/// present; the type system does not otherwise enforce it. Not `Copy`: it
/// embodies exactly one strong-count debt, paid once by `bstack_drop`.
pub struct StrongRef<T>(BStackRef<T>);

impl<T> StrongRef<T> {
    /// # Safety
    ///
    /// `inner` must reference a live `(rc)` block on which the caller holds one
    /// strong reference that this wrapper now embodies (and pays exactly once).
    pub unsafe fn new(inner: BStackRef<T>) -> Self {
        Self(inner)
    }
}

/// `#[bstack_strong]` on an `(rc, weak)` `T`: holds the data ref and the control
/// ref; teardown runs the two-phase strong-then-weak release. Not `Copy` — one
/// strong-count debt, paid once.
pub struct StrongWeakRef<T: BStackWeakable>(BStackRef<T>, BStackRef<T::Control>);

/// `#[bstack_weak]` on an `(rc, weak)` `T`: holds only the control ref; teardown
/// decrements `ctrl.weak` and frees the control block at zero. The data block is
/// never touched. Not `Copy` — one weak-count debt, paid once.
pub struct WeakRef<T: BStackWeakable>(BStackRef<T::Control>);

impl<T: BStackWeakable> WeakRef<T> {
    /// # Safety
    ///
    /// `ctrl` must reference a live control block on which the caller holds one
    /// weak count that this wrapper now embodies (and pays exactly once).
    pub unsafe fn new(ctrl: BStackRef<T::Control>) -> Self {
        Self(ctrl)
    }

    /// The (non-owning) control reference.
    pub fn ctrl_ref(&self) -> BStackRef<T::Control> {
        self.0
    }
}

/// Read a data block's `ctrl` back-pointer (a `u64` offset at
/// [`layout::CTRL_BACKPTR_OFFSET`]) and resolve it to a typed control ref,
/// recovering the control block's length from `size_of::<T::Control>()`.
fn read_ctrl_ref<T: BStackWeakable, A: BStackRaiiAllocator>(
    data_ref: BStackRef<T>,
    allocator: &A,
) -> io::Result<BStackRef<T::Control>> {
    let pos = layout::checked_off(data_ref.into_range().start(), layout::CTRL_BACKPTR_OFFSET)?;
    let mut bytes = [0u8; 8];
    allocator.stack().get_into(pos, &mut bytes)?;
    let ctrl_offset = u64::from_le_bytes(bytes);
    let ctrl_range = BStackRange::new(ctrl_offset, size_of::<T::Control>() as u64);
    Ok(unsafe { BStackRef::from_range(ctrl_range) })
}

impl<T: BStackBlock> BStackDrop for OwnedRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        // Bound the in-file owned recursion (this is the chokepoint every
        // generated `__bstack_drop_children` re-enters through): an owned cycle
        // errors here instead of overflowing the native stack.
        let _depth = crate::teardown::TeardownDepthGuard::enter()?;
        // An owned child is freed by running the block's own recursive teardown,
        // which frees its children (post-order) and then deallocs the block.
        // SAFETY: an `OwnedRef` is an ownership token minted (via the `unsafe`
        // `new`) over a live block this token exclusively owns.
        unsafe { crate::teardown::drop_block::<T, A>(self.0.into_range(), allocator) }
    }
}

impl<T: BStackBlock> BStackDrop for StrongRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        // Bound the recursion a last-owner free re-enters (see `OwnedRef`).
        let _depth = crate::teardown::TeardownDepthGuard::enter()?;
        let data_range = self.0.into_range();
        let off = layout::checked_off(data_range.start(), layout::RC_REFCOUNT_OFFSET)?;
        // Decrement the inline refcount; only the last owner frees the block.
        if refcount::fetch_sub(allocator.stack(), off, 1)? == 1 {
            // SAFETY: last strong owner (the fetch_sub hit 1) of a live block.
            unsafe { crate::teardown::drop_block::<T, A>(data_range, allocator)? };
        }
        Ok(())
    }
}

impl<T: BStackWeakable> StrongWeakRef<T> {
    /// Resolve the control ref from the data block's `ctrl` back-pointer with a
    /// single read, then pair it with the data ref.
    ///
    /// # Safety
    ///
    /// `data_ref` must reference a live `(rc, weak)` block on which the caller
    /// holds one strong reference that the result now embodies (and pays
    /// exactly once).
    pub unsafe fn from_disk<A: BStackRaiiAllocator>(
        data_ref: BStackRef<T>,
        allocator: &A,
    ) -> io::Result<Self> {
        let ctrl = read_ctrl_ref(data_ref, allocator)?;
        Ok(StrongWeakRef(data_ref, ctrl))
    }

    /// The (non-owning) data reference.
    pub fn data_ref(&self) -> BStackRef<T> {
        self.0
    }

    /// The (non-owning) control reference.
    pub fn ctrl_ref(&self) -> BStackRef<T::Control> {
        self.1
    }
}

/// The two-phase strong release for an `(rc, weak)` block, given raw data and
/// control ranges. Requires only `T: BStackBlock` (for the data block's own
/// recursive teardown), so it is shared by both [`StrongWeakRef::bstack_drop`]
/// and [`crate::BStackRc`]'s `Drop` — the latter carries `T: BStackBlock` and so
/// cannot construct a `StrongWeakRef<T>` (which needs `BStackWeakable`) itself.
pub(crate) fn strong_release_ctrl<T: BStackBlock, A: BStackRaiiAllocator>(
    allocator: &A,
    data_range: BStackRange,
    ctrl_range: BStackRange,
) -> io::Result<()> {
    // Bound the recursion a last-owner free re-enters (see `OwnedRef`).
    let _depth = crate::teardown::TeardownDepthGuard::enter()?;
    let stack = allocator.stack();
    let strong_off = layout::checked_off(ctrl_range.start(), layout::CTRL_STRONG_OFFSET)?;
    // Phase 1: last strong owner frees the data block (children + shell), then
    // releases the phantom weak the strong owners collectively held.
    if refcount::fetch_sub(stack, strong_off, 1)? == 1 {
        // SAFETY: last strong owner of a live `(rc, weak)` data block.
        unsafe { crate::teardown::drop_block::<T, A>(data_range, allocator)? };
        let weak_off = layout::checked_off(ctrl_range.start(), layout::CTRL_WEAK_OFFSET)?;
        // Phase 2 (early): if no real weak handles remain, the phantom release
        // drives weak to zero and the control block is freed here.
        if refcount::fetch_sub(stack, weak_off, 1)? == 1 {
            unsafe { dealloc_range(allocator, ctrl_range)? };
        }
    }
    Ok(())
}

impl<T: BStackWeakable> BStackDrop for StrongWeakRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        strong_release_ctrl::<T, A>(allocator, self.0.into_range(), self.1.into_range())
    }
}

impl<T: BStackWeakable> WeakRef<T> {
    /// Resolve the control ref from the data block's `ctrl` back-pointer.
    ///
    /// # Safety
    ///
    /// As [`new`](Self::new): the caller must hold one weak count on the
    /// resolved control block, transferred into the result.
    pub unsafe fn from_disk<A: BStackRaiiAllocator>(
        data_ref: BStackRef<T>,
        allocator: &A,
    ) -> io::Result<Self> {
        Ok(WeakRef(read_ctrl_ref(data_ref, allocator)?))
    }
}

impl<T: BStackWeakable> BStackDrop for WeakRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        let ctrl_range = self.0.into_range();
        let weak_off = layout::checked_off(ctrl_range.start(), layout::CTRL_WEAK_OFFSET)?;
        // Decrement ctrl.weak; free the control block when the last weak handle
        // (or the phantom) drops it to zero. The data block is never touched.
        if refcount::fetch_sub(allocator.stack(), weak_off, 1)? == 1 {
            unsafe { dealloc_range(allocator, ctrl_range)? };
        }
        Ok(())
    }
}
