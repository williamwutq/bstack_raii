//! The crate's **RAII drop protocol**: the [`BStackDrop`] trait every affine owner
//! implements, the [`AutoDrop`] guard that bridges it to Rust's `Drop`, and the
//! internal [`BlockShell`] teardown token.
//!
//! These are semantic types — the ownership vocabulary — so they live under
//! [`crate::types`]. The teardown *mechanism* they drive (`drop_block`,
//! `dealloc_range`, `wal_teardown`) is in [`crate::io_core::teardown`].

use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Deref;
use std::io;

use bstack::BStackRange;

use crate::BStackRaiiAllocator;
use crate::types::block::BStackBlock;
use crate::io_core::teardown::drop_block;

/// The safe teardown protocol for an affine (non-`Copy`) owning handle: consume
/// `self` and free the on-disk block(s) it owns.
///
/// The allocator is bound to the crate-wide [`BStackRaiiAllocator`], whose
/// [`BStackOwnedSliceAllocator`](bstack::BStackOwnedSliceAllocator) supertrait pins
/// `Allocated<'a> = BStackOwnedSlice<'a, A>` (so a reconstructed owned slice is the
/// accepted `dealloc` handle) and `Error = io::Error` (so the layer speaks
/// [`io::Result`]).
pub trait BStackDrop: Sized {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()>;
}

/// An affine (non-`Copy`) teardown token for a `T` block, so
/// [`wal_teardown`](crate::io_core::teardown::wal_teardown) can drive the block
/// teardown of a [`BStackOwned<T>`](crate::BStackOwned) without the block handle
/// itself implementing [`BStackDrop`]. Not public: minted only inside `bstack_drop`.
pub(crate) struct BlockShell<T: BStackBlock> {
    range: BStackRange,
    _marker: PhantomData<fn() -> T>,
}

impl<T: BStackBlock> BlockShell<T> {
    pub(crate) fn new(range: BStackRange) -> Self {
        Self {
            range,
            _marker: PhantomData,
        }
    }
}

impl<T: BStackBlock> BStackDrop for BlockShell<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        // SAFETY: a `BlockShell` is minted only by `BStackOwned::bstack_drop` from
        // a handle that asserted sole ownership at construction.
        unsafe { drop_block::<T, A>(self.range, allocator) }
    }
}

/// A guard that runs [`BStackDrop::bstack_drop`] on its inner handle when it goes
/// out of scope, bridging fallible on-disk teardown to Rust's `Drop`.
///
/// It is the *one* place that calls `bstack_drop` from a `Drop` impl: every
/// allocator-bound handle that wants automatic cleanup is (or embeds) an
/// `AutoDrop`, rather than hand-writing its own `Drop`. A bare [`BStackDrop`]
/// handle that is not wrapped frees nothing on its own — its `bstack_drop` is
/// invoked explicitly, or runs as a child of a parent block's recursive
/// teardown.
///
/// It is a newtype over `(ManuallyDrop<T>, &'a A)`; `bstack_move!` and the raw
/// accessors defuse it via [`into_raw_parts`](Self::into_raw_parts) so no
/// parallel destruction path exists.
pub struct AutoDrop<'a, T: BStackDrop, A: BStackRaiiAllocator> {
    inner: ManuallyDrop<T>,
    allocator: &'a A,
}

impl<'a, T: BStackDrop, A: BStackRaiiAllocator> AutoDrop<'a, T, A> {
    /// Pair an inner handle with its allocator into an auto-dropping guard.
    ///
    /// # Safety
    /// The caller asserts `inner` describes a live allocation owned by
    /// `allocator` and that no other handle will also free it.
    pub unsafe fn from_raw(inner: T, allocator: &'a A) -> Self {
        Self {
            inner: ManuallyDrop::new(inner),
            allocator,
        }
    }

    /// Split into the raw inner handle and allocator **without** running the
    /// disk-level `Drop`. The caller takes over responsibility for the
    /// allocation (e.g. `bstack_move!`, which frees only the parent shell).
    pub fn into_raw_parts(self) -> (T, &'a A) {
        // Wrapping `self` in ManuallyDrop defuses our own `Drop`, so
        // `bstack_drop` is not called; then move the inner `T` out.
        let mut me = ManuallyDrop::new(self);
        let inner = unsafe { ManuallyDrop::take(&mut me.inner) };
        (inner, me.allocator)
    }

    /// The allocator this handle is bound to.
    pub fn allocator(&self) -> &'a A {
        self.allocator
    }

    /// Borrow the underlying handle, e.g. to call generated field accessors:
    /// `owned.handle().get_field(stack)`.
    pub fn handle(&self) -> &T {
        &self.inner
    }
}

impl<'a, T: BStackDrop, A: BStackRaiiAllocator> Deref for AutoDrop<'a, T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<'a, T: BStackDrop, A: BStackRaiiAllocator> Drop for AutoDrop<'a, T, A> {
    fn drop(&mut self) {
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        // Errors are swallowed, matching the contract of Rust's `Drop`.
        let _ = inner.bstack_drop(self.allocator);
    }
}
