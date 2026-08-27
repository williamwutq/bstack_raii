//! [`BStackOwned`]: a without-allocator, uniquely-owned block handle.
//!
//! `BStackOwned<T>` is an *ownership marker* over an inner [`BStackDrop`] handle
//! (typically a `#[bstack_block]` type). Its own [`BStackDrop`] recursively frees
//! the block — but, being a bare handle, it frees **nothing on scope exit**:
//! teardown is explicit ([`bstack_drop`](BStackDrop::bstack_drop)) or automatic
//! only once wrapped in an [`AutoDrop`] (`owned.auto(alloc)`), whose Rust `Drop`
//! runs it. This keeps a persistent root from being silently deleted when its
//! handle drops.
//!
//! `X::new(..)` and `bstack_move!`'d owned children hand back a bare
//! `BStackOwned<X>`; the caller decides when (and whether) it dies.

use core::ops::Deref;
use std::io;

use super::super::traits::block::BStackBlock;
use super::super::traits::drop::{AutoDrop, BStackDrop, BlockShell, drop_block};
use super::super::traits::r#move::{BStackMove, BStackMoveExpr};
use super::super::traits::reference::BStackRef;
use crate::BStackRaiiAllocator;
use crate::io_core::teardown::{TeardownDepthGuard, wal_teardown};

/// A uniquely-owned handle to a block: an ownership marker over an inner
/// [`BStackDrop`] handle whose teardown recursively frees the block on disk.
///
/// Carries no allocator (unlike an [`AutoDrop`]-wrapped handle), so it never
/// frees itself on `Drop`. Wrap it via [`auto`](Self::auto) for RAII, or free it
/// explicitly with [`BStackDrop::bstack_drop`].
pub struct BStackOwned<T: BStackBlock>(T);

impl<T: BStackBlock> BStackOwned<T> {
    /// Mark `inner` as uniquely owned.
    ///
    /// # Safety
    /// The caller asserts `inner` describes a live allocation that no other
    /// handle will also free.
    pub unsafe fn from_raw(inner: T) -> Self {
        BStackOwned(inner)
    }

    /// Unwrap to the inner handle, dropping the ownership marker without freeing
    /// anything (the caller takes over responsibility). Used to read a child's
    /// offset when transferring it into a parent field.
    ///
    /// The returned handle is a non-owning **view** (`Copy`, no [`BStackDrop`]),
    /// so this cannot mint a second owner — the whole point of the affine design:
    /// only a `BStackOwned` / `*Ref` token can free, and this consumes the one
    /// that existed.
    pub fn into_inner(self) -> T {
        self.0
    }

    /// Borrow the inner handle, e.g. to call generated field accessors:
    /// `owned.handle().get_field(stack)`.
    pub fn handle(&self) -> &T {
        &self.0
    }

    /// Attach an allocator to make an auto-freeing [`AutoDrop`] guard: dropping
    /// the returned value runs this handle's recursive teardown.
    pub fn auto<A: BStackRaiiAllocator>(self, allocator: &A) -> AutoDrop<'_, Self, A> {
        // SAFETY: a `BStackOwned` asserts sole ownership of a live block at
        // construction, exactly the invariant `AutoDrop::from_raw` requires.
        unsafe { AutoDrop::from_raw(self, allocator) }
    }
}

impl<T: BStackBlock> Deref for BStackOwned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: BStackBlock> BStackDrop for BStackOwned<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        // Recursively free the owned block (and its children) as one crash-atomic,
        // leak-reclaiming batch — automatically, whenever the allocator names a WAL
        // anchor. `wal_teardown` collects the whole subtree's frees and commits
        // them as one transaction (and is a plain teardown when there is no anchor,
        // or when this runs nested inside an outer teardown that owns the sink).
        //
        // The affine `BlockShell` carries the block-teardown that used to live in
        // `impl BStackDrop for <handle>`; the bare handle is now a pure view.
        wal_teardown(BlockShell::<T>::new(self.0.range()), allocator)
    }
}

/// `bstack_move!` on an `AutoDrop`-wrapped owned handle: defuse the guard and
/// destructure via the block's [`BStackMove`], passing the recovered allocator.
///
/// (A *bare* `BStackOwned<X>` carries no allocator, so it is moved with the
/// explicit two-argument form `bstack_move!(owned, allocator)` instead.)
impl<'a, X: BStackMove, A: BStackRaiiAllocator> BStackMoveExpr for AutoDrop<'a, BStackOwned<X>, A> {
    type Output = io::Result<X::Fields<'a, A>>;
    fn bstack_move(self) -> Self::Output {
        let (owned, allocator) = self.into_raw_parts();
        X::bstack_move(owned, allocator)
    }
}

/// `#[bstack_owned]`: an exclusively-owned child — the without-allocator drop core
/// a generated `__bstack_drop_children` mints for an owned field.
///
/// Not `Copy`/`Clone`, and its field is private: this is an *ownership* token whose
/// [`BStackDrop`] frees unconditionally, so it must not be freely mintable or
/// duplicable from a non-owning [`BStackRef`] — construct it with the `unsafe`
/// [`new`](Self::new).
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

impl<T: BStackBlock> BStackDrop for OwnedRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        // Bound the in-file owned recursion (this is the chokepoint every
        // generated `__bstack_drop_children` re-enters through): an owned cycle
        // errors here instead of overflowing the native stack.
        let _depth = TeardownDepthGuard::enter()?;
        // An owned child is freed by running the block's own recursive teardown,
        // which frees its children (post-order) and then deallocs the block.
        // SAFETY: an `OwnedRef` is an ownership token minted (via the `unsafe`
        // `new`) over a live block this token exclusively owns.
        unsafe { drop_block::<T, A>(allocator, self.0.into_range()) }
    }
}
