//! [`BStackRc`] + [`BStackWeak`]: the with-allocator shared handles.
//!
//! Neither hand-writes a `Drop`. Each embeds an [`AutoDrop`] over a
//! without-allocator *drop core* ([`StrongCore`] / [`WeakRef`]) whose
//! [`BStackDrop`] performs the refcount release; the embedded guard runs it on
//! scope exit.

use core::mem::size_of;
use core::ops::Deref;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::BStackRange;

use super::super::traits::drop::drop_block;
use super::super::traits::{
    AutoDrop, BStackBlock, BStackDrop, BStackMove, BStackMoveExpr, BStackRef, BStackWeakable,
};
use super::BStackOwned;
use crate::handback::ReplaceError;
use crate::io_core::{TeardownDepthGuard, dealloc_range, refcount};
use crate::primitives::{EightCC, NonNullOffset, TryClone, checked_off};
use crate::util::{put_u64, read_u64};

use super::block::{BlockHeader, HEADER_SIZE};

// The macros inject the refcount / control back-pointer / control counters
// immediately after the header, ahead of any user fields and in a fixed order.
// Their offsets are therefore the same for *every* block, so they live here as
// constants rather than as per-type trait members — alongside the shared handles
// ([`BStackRc`] / [`BStackWeak`]) that read and write them.

/// `#[bstack_block(rc)]` data block: offset of the inline `refcount: AtomicU64`,
/// injected right after the header.
///
/// ```text
/// struct XOnDisk { header, refcount: AtomicU64, <user fields...> }
/// ```
pub(crate) const RC_REFCOUNT_OFFSET: u64 = HEADER_SIZE;

/// `#[bstack_block(rc, weak)]` data block: offset of the `ctrl` back-pointer to
/// the control block, injected right after the header.
///
/// ```text
/// struct XOnDisk { header, ctrl: BStackRef<XOnDiskRef>, <user fields...> }
/// ```
pub(crate) const CTRL_BACKPTR_OFFSET: u64 = HEADER_SIZE;

/// `#[bstack_block(rc, weak)]` control block (`XOnDiskRef`): offset of `strong`.
///
/// ```text
/// struct XOnDiskRef { header, strong: AtomicU64, weak: AtomicU64, x: BStackRef<X> }
/// ```
pub(crate) const CTRL_STRONG_OFFSET: u64 = HEADER_SIZE;

/// Control block: offset of `weak` (starts at 1 — the phantom weak held
/// collectively by all live strong owners).
pub(crate) const CTRL_WEAK_OFFSET: u64 = HEADER_SIZE + 8;

/// Control block: offset of `x`, the forward pointer back to the data block.
/// Read by [`BStackWeak::upgrade`] once it wins the strong CAS.
pub(crate) const CTRL_DATA_OFFSET: u64 = HEADER_SIZE + 16;

/// Total bytes of an `(rc, weak)` control block: [`BlockHeader`] + `strong` + `weak` +
/// the data forward-pointer `u64`. Fixed for every weakable type — the control layout
/// does not depend on `T` — so a control payload is a fixed-size stack buffer.
pub(crate) const CONTROL_SIZE: u64 = CTRL_DATA_OFFSET + 8;

// Guard the hand-derived offsets against a header size change.
const _: () = assert!(HEADER_SIZE == 16);

// ---------------------------------------------------------------------------
// Without-allocator drop cores for the `rc` / `weak` field annotations.
//
// Each is a concrete, non-`Copy` ownership token a generated
// `__bstack_drop_children` mints for one `#[bstack_strong]` / `#[bstack_weak]`
// field; its `BStackDrop` carries that annotation's refcount-release logic. They
// live here beside [`StrongCore`] (the `BStackRc` drop core that dispatches to
// them) and the control-block offset constants they read.
// ---------------------------------------------------------------------------

/// `#[bstack_strong]` on a plain `(rc)` `T`: holds just the data ref; teardown
/// decrements the inline refcount and frees at zero.
///
/// The macro only emits this for children whose type is `#[bstack_block(rc)]`,
/// so the inline `refcount` at [`RC_REFCOUNT_OFFSET`] is guaranteed
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

impl<T: BStackBlock> BStackDrop for StrongRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        // Bound the recursion a last-owner free re-enters (see `OwnedRef`).
        let _depth = TeardownDepthGuard::enter()?;
        let data_range = self.0.into_range();
        let off = NonNullOffset::from_field(checked_off(data_range.start(), RC_REFCOUNT_OFFSET)?)?;
        // Decrement the inline refcount; only the last owner frees the block.
        if refcount::fetch_sub(allocator.stack(), off, 1)? == 1 {
            // SAFETY: last strong owner (the fetch_sub hit 1) of a live block.
            unsafe { drop_block::<T, A>(allocator, data_range)? };
        }
        Ok(())
    }
}

/// `#[bstack_strong]` on an `(rc, weak)` `T`: holds the data ref and the control
/// ref; teardown runs the two-phase strong-then-weak release. Not `Copy` — one
/// strong-count debt, paid once.
pub struct StrongWeakRef<T: BStackWeakable>(BStackRef<T>, BStackRef<T::Control>);

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
        let ctrl = WeakRef::<T>::resolve_ctrl(data_ref, allocator)?;
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

impl<T: BStackWeakable> BStackDrop for StrongWeakRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        strong_release_ctrl::<T, A>(allocator, self.0.into_range(), self.1.into_range())
    }
}

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

    /// Resolve a data block's `ctrl` back-pointer (a `u64` offset at
    /// [`CTRL_BACKPTR_OFFSET`]) to its typed control ref, recovering the control
    /// block's length from `size_of::<T::Control>()`.
    ///
    /// The shared control-block resolver behind both [`from_disk`](Self::from_disk)
    /// and [`StrongWeakRef::from_disk`]. It returns the *bare* ref (it takes no
    /// weak debt), so the strong-weak path can store it under the strong owners'
    /// phantom weak rather than as a genuine weak handle.
    fn resolve_ctrl<A: BStackRaiiAllocator>(
        data_ref: BStackRef<T>,
        allocator: &A,
    ) -> io::Result<BStackRef<T::Control>> {
        let pos = checked_off(data_ref.into_range().start(), CTRL_BACKPTR_OFFSET)?;
        let mut bytes = [0u8; 8];
        allocator.stack().get_into(pos, &mut bytes)?;
        let ctrl_offset = u64::from_le_bytes(bytes);
        let ctrl_range = BStackRange::new(ctrl_offset, size_of::<T::Control>() as u64);
        Ok(unsafe { BStackRef::from_range(ctrl_range) })
    }

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
        Ok(Self(Self::resolve_ctrl(data_ref, allocator)?))
    }
}

impl<T: BStackWeakable> BStackDrop for WeakRef<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        // Pay this handle's one weak debt on the control block; the data block is
        // never touched.
        release_weak(allocator, self.0.into_range())
    }
}

/// Pay one weak-count debt on a control block by range: decrement `ctrl.weak`
/// (at [`CTRL_WEAK_OFFSET`]) and free the control block iff this was the last
/// weak holder — a real [`WeakRef`], or the phantom weak the strong owners hold
/// collectively (released by [`strong_release_ctrl`] once the last strong owner
/// leaves). The data block is never touched here.
///
/// This is the untyped core [`WeakRef::bstack_drop`] and `strong_release_ctrl`
/// share: the latter runs under only `T: BStackBlock` and so cannot name a
/// `WeakRef<T>` (which needs `BStackWeakable`), so the common logic takes a raw
/// `ctrl_range` rather than living on `WeakRef`. The caller must hold the weak
/// count being paid — both call sites do (a consumed `WeakRef`, or the last
/// strong owner's phantom).
fn release_weak<A: BStackRaiiAllocator>(allocator: &A, ctrl_range: BStackRange) -> io::Result<()> {
    let weak_off = NonNullOffset::from_field(checked_off(ctrl_range.start(), CTRL_WEAK_OFFSET)?)?;
    if refcount::fetch_sub(allocator.stack(), weak_off, 1)? == 1 {
        // SAFETY: last weak holder of a live control block; nothing else frees it.
        unsafe { dealloc_range(allocator, ctrl_range)? };
    }
    Ok(())
}

/// The two-phase strong release for an `(rc, weak)` block, given raw data and
/// control ranges. Requires only `T: BStackBlock` (for the data block's own
/// recursive teardown), so it is shared by both [`StrongWeakRef::bstack_drop`]
/// and [`BStackRc`]'s `Drop` (via [`StrongCore`]) — the latter carries
/// `T: BStackBlock` and so cannot construct a `StrongWeakRef<T>` (which needs
/// `BStackWeakable`) itself.
pub(crate) fn strong_release_ctrl<T: BStackBlock, A: BStackRaiiAllocator>(
    allocator: &A,
    data_range: BStackRange,
    ctrl_range: BStackRange,
) -> io::Result<()> {
    // Bound the recursion a last-owner free re-enters (see `OwnedRef`).
    let _depth = TeardownDepthGuard::enter()?;
    let strong_off =
        NonNullOffset::from_field(checked_off(ctrl_range.start(), CTRL_STRONG_OFFSET)?)?;
    // Phase 1: last strong owner frees the data block (children + shell), then
    // releases the phantom weak the strong owners collectively held — which is
    // exactly a weak-handle drop on the control block ([`release_weak`], the same
    // path [`WeakRef::bstack_drop`] takes), so a lingering real weak keeps the
    // control block alive and the last weak holder frees it.
    if refcount::fetch_sub(allocator.stack(), strong_off, 1)? == 1 {
        // SAFETY: last strong owner of a live `(rc, weak)` data block.
        unsafe { drop_block::<T, A>(allocator, data_range)? };
        release_weak(allocator, ctrl_range)?;
    }
    Ok(())
}

/// The without-allocator drop core of a [`BStackRc`]: the data ref plus the
/// optional control-block range.
///
/// `ctrl` distinguishes the two block kinds at runtime — `None` for a plain
/// `(rc)` block (inline refcount), `Some(range)` for an `(rc, weak)` block
/// (control block). Its [`BStackDrop`] is the strong release, so a `BStackRc`'s
/// embedded [`AutoDrop`] runs it automatically and the handle needs no
/// hand-written `Drop`.
pub(crate) struct StrongCore<T: BStackBlock> {
    data: BStackRef<T>,
    ctrl: Option<BStackRange>,
}

impl<T: BStackBlock> BStackDrop for StrongCore<T> {
    fn bstack_drop<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Result<()> {
        match self.ctrl {
            // SAFETY: this StrongCore held one strong reference; it is paid here.
            None => unsafe { StrongRef::new(self.data) }.bstack_drop(allocator),
            Some(ctrl) => strong_release_ctrl::<T, A>(allocator, self.data.into_range(), ctrl),
        }
    }
}

/// A shared, refcounted, allocator-bound handle.
///
/// Serves **both** block kinds via its [`StrongCore`]'s runtime `ctrl`:
/// * `None` — a plain `#[bstack_block(rc)]` block, whose refcount lives inline
///   in the data block at [`RC_REFCOUNT_OFFSET`].
/// * `Some(range)` — an `#[bstack_block(rc, weak)]` block, whose `strong`/`weak`
///   counters live in a separate control block at `range`.
///
/// Carrying this as a runtime `Option` (rather than a type-level split via an
/// associated `Strong` handle) keeps `BStackRc<'a, T, A>`'s public signature
/// fixed at three parameters; a zero-cost representation can replace it later
/// without breaking callers.
///
/// **Invariant:** for a `T: BStackWeakable` block, `ctrl` is always `Some` — such
/// blocks are only ever constructed through the control-block paths
/// ([`BStackWeak::upgrade`], `bstack_move!`). `downgrade` relies on this.
pub struct BStackRc<'a, T: BStackBlock, A: BStackRaiiAllocator> {
    inner: AutoDrop<'a, StrongCore<T>, A>,
    /// A standing copy of the typed handle, purely so [`Deref`] can hand back
    /// `&T` — [`Deref::deref`] can't construct a temporary and return a
    /// reference to it. Reconstructed once in [`from_raw`](Self::from_raw), same
    /// as [`handle`](Self::handle) computes on demand; doesn't touch the refcount.
    handle: T,
}

impl<'a, T: BStackBlock, A: BStackRaiiAllocator> BStackRc<'a, T, A> {
    /// Reconstruct a shared handle from its raw parts.
    ///
    /// # Safety
    /// The refs must describe a live `(rc)` / `(rc, weak)` block owned by
    /// `allocator`, and this handle must account for a strong count the caller
    /// has already established (e.g. the allocation's initial `strong = 1`, or a
    /// count bumped by `upgrade`). `ctrl` must be `Some` iff the block is
    /// `(rc, weak)`.
    pub unsafe fn from_raw(
        data: BStackRef<T>,
        ctrl: Option<BStackRange>,
        allocator: &'a A,
    ) -> Self {
        let handle = unsafe { <T as BStackBlock>::from_range(data.into_range()) };
        Self {
            inner: unsafe { AutoDrop::from_raw(StrongCore { data, ctrl }, allocator) },
            handle,
        }
    }

    fn data(&self) -> BStackRef<T> {
        self.inner.handle().data
    }

    fn ctrl(&self) -> Option<BStackRange> {
        self.inner.handle().ctrl
    }

    fn allocator(&self) -> &'a A {
        self.inner.allocator()
    }

    /// The underlying typed handle, e.g. to call generated field accessors:
    /// `rc.handle().get_field(stack)` (or just `rc.get_field(stack)` via
    /// [`Deref`]). Cheap: it just re-wraps the cached handle's range and does
    /// not touch the refcount.
    pub fn handle(&self) -> T {
        unsafe { <T as BStackBlock>::from_range(self.handle.range()) }
    }

    /// Consume the handle into its raw parts **without** decrementing the strong
    /// count — the count is transferred to the caller (e.g. into a parent's
    /// `#[bstack_strong]` field). `ctrl` is `Some` for `(rc, weak)` blocks.
    pub fn into_raw(self) -> (BStackRef<T>, Option<BStackRange>) {
        let (core, _) = self.inner.into_raw_parts();
        (core.data, core.ctrl)
    }

    /// Byte offset of the strong counter for this handle's block kind.
    fn strong_offset(&self) -> io::Result<NonNullOffset> {
        let off = match self.ctrl() {
            None => checked_off(self.data().into_range().start(), RC_REFCOUNT_OFFSET)?,
            Some(ctrl) => checked_off(ctrl.start(), CTRL_STRONG_OFFSET)?,
        };
        NonNullOffset::from_field(off)
    }
}

/// Field access without the `.handle()` indirection: `rc.get_field(stack)`
/// instead of `rc.handle().get_field(stack)`, matching [`BStackOwned`]'s
/// `Deref`. Same handle [`handle`](Self::handle) returns, just borrowed rather
/// than re-wrapped fresh each call.
impl<'a, T: BStackBlock, A: BStackRaiiAllocator> Deref for BStackRc<'a, T, A> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.handle
    }
}

/// Cloning a strong handle bumps the block's strong count and returns another
/// handle to the **same** block — sharing, not copying (like `Rc::clone`). This
/// is the clone semantics for a shared block; there is deliberately no
/// deep-copy-to-owned (`TryCloneIn`) for one.
impl<'a, T: BStackBlock, A: BStackRaiiAllocator> TryClone for BStackRc<'a, T, A> {
    fn try_clone(&self) -> io::Result<Self> {
        refcount::fetch_add(self.allocator().stack(), self.strong_offset()?, 1)?;
        // SAFETY: the fetch_add above established the strong count this clone
        // accounts for.
        Ok(unsafe { Self::from_raw(self.data(), self.ctrl(), self.allocator()) })
    }
}

impl<'a, T: BStackWeakable, A: BStackRaiiAllocator> BStackRc<'a, T, A> {
    /// Create a weak handle to the same block by incrementing `ctrl.weak`.
    ///
    /// Available only for `(rc, weak)` blocks (`T: BStackWeakable`), so a plain
    /// `(rc)` block's `BStackRc` has no `downgrade` at all — a compile error, not
    /// a runtime hazard.
    pub fn downgrade(&self) -> io::Result<BStackWeak<'a, T, A>> {
        // Invariant: a weakable block's `BStackRc` always carries a control ref.
        let ctrl_range = self
            .ctrl()
            .expect("BStackRc<T: BStackWeakable> always has a control block");
        let weak_off =
            NonNullOffset::from_field(checked_off(ctrl_range.start(), CTRL_WEAK_OFFSET)?)?;
        refcount::fetch_add(self.allocator().stack(), weak_off, 1)?;
        let ctrl = unsafe { BStackRef::<T::Control>::from_range(ctrl_range) };
        // SAFETY: the fetch_add above established the weak count this handle holds.
        Ok(unsafe { BStackWeak::from_raw(ctrl, self.allocator()) })
    }
}

impl<'a, T: BStackMove, A: BStackRaiiAllocator> BStackRc<'a, T, A> {
    /// `Rc::try_unwrap` + destructure: if this handle is the **sole strong
    /// owner**, move every field out (freeing only the data shell) and return
    /// them; otherwise hand the handle back in `Err`.
    ///
    /// The check-and-take is an atomic CAS `strong: 1 -> 0`, so a concurrent
    /// clone or `upgrade` makes it fail cleanly rather than tearing a shared
    /// block apart. Works for both `(rc)` (inline count) and `(rc, weak)` (the
    /// control block's phantom weak is released, freeing it if no weak handles
    /// remain). This is what `bstack_move!` calls on a `BStackRc`.
    pub fn try_move(self) -> io::Result<Result<T::Fields<'a, A>, Self>> {
        let strong_off = self.strong_offset()?;
        let stack = self.allocator().stack();

        // Atomic try-unwrap: succeed only if the strong count is exactly 1.
        if !refcount::cas(stack, strong_off, 1, 0)? {
            return Ok(Err(self));
        }

        // Strong is now 0 — no concurrent upgrade can revive the data block, so
        // it is safe to move the fields out and free the data shell. Defuse the
        // embedded guard so it does not double-free.
        let (core, allocator) = self.inner.into_raw_parts();
        let StrongCore { data, ctrl } = core;
        let owned =
            unsafe { BStackOwned::from_raw(<T as BStackBlock>::from_range(data.into_range())) };
        let fields = T::bstack_move(owned, allocator)?;

        // `(rc, weak)`: release the phantom weak; free the control block at zero.
        if let Some(ctrl) = ctrl {
            let weak_off = NonNullOffset::from_field(checked_off(ctrl.start(), CTRL_WEAK_OFFSET)?)?;
            if refcount::fetch_sub(allocator.stack(), weak_off, 1)? == 1 {
                unsafe { dealloc_range(allocator, ctrl)? };
            }
        }
        Ok(Ok(fields))
    }
}

impl<'a, T: BStackMove, A: BStackRaiiAllocator> BStackMoveExpr for BStackRc<'a, T, A> {
    type Output = io::Result<Result<T::Fields<'a, A>, Self>>;
    fn bstack_move(self) -> Self::Output {
        self.try_move()
    }
}

/// A non-owning weak handle to an `(rc, weak)` block's control block.
///
/// Obtained from [`BStackRc::downgrade`] or [`TryClone::try_clone`]. It keeps the
/// control block alive (so [`upgrade`](BStackWeak::upgrade) can check liveness)
/// but never pins the data block. Its drop core is a [`WeakRef`], whose
/// [`BStackDrop`] decrements `ctrl.weak` and frees the control block at zero.
pub struct BStackWeak<'a, T: BStackWeakable, A: BStackRaiiAllocator> {
    inner: AutoDrop<'a, WeakRef<T>, A>,
}

impl<'a, T: BStackWeakable, A: BStackRaiiAllocator> BStackWeak<'a, T, A> {
    /// Reconstruct a weak handle from its raw control ref.
    ///
    /// # Safety
    /// `ctrl` must describe a live control block owned by `allocator`, and this
    /// handle must account for a weak count the caller has already established.
    pub unsafe fn from_raw(ctrl: BStackRef<T::Control>, allocator: &'a A) -> Self {
        Self {
            inner: unsafe { AutoDrop::from_raw(WeakRef::new(ctrl), allocator) },
        }
    }

    fn ctrl(&self) -> BStackRef<T::Control> {
        self.inner.handle().ctrl_ref()
    }

    fn allocator(&self) -> &'a A {
        self.inner.allocator()
    }

    /// Consume the handle into its raw control ref **without** decrementing the
    /// weak count — the count is transferred to the caller.
    pub fn into_raw(self) -> BStackRef<T::Control> {
        let (weak, _) = self.inner.into_raw_parts();
        weak.ctrl_ref()
    }

    /// Attempt to promote to a strong handle. Succeeds iff `ctrl.strong` is
    /// currently non-zero (CAS-increment-if-nonzero), reading `ctrl.x` to recover
    /// the data ref. Returns `None` if the data block is already gone.
    pub fn upgrade(&self) -> io::Result<Option<BStackRc<'a, T, A>>> {
        let allocator = self.allocator();
        let stack = allocator.stack();
        let ctrl_range = self.ctrl().into_range();
        // Both offsets are computed (and can fail) up front, before any mutation —
        // so an overflowing `ctrl_range.start()` never leaves a claimed strong
        // count with nothing to release it.
        let strong_off =
            NonNullOffset::from_field(checked_off(ctrl_range.start(), CTRL_STRONG_OFFSET)?)?;
        let data_pos = checked_off(ctrl_range.start(), CTRL_DATA_OFFSET)?;
        if refcount::increment_if_nonzero(stack, strong_off)?.is_none() {
            return Ok(None);
        }
        // Strong is now claimed; recover the data ref from the forward pointer.
        let mut bytes = [0u8; 8];
        if let Err(e) = stack.get_into(data_pos, &mut bytes) {
            // The claim above already landed; release it here rather than
            // orphan it (same release-on-failure idea as the weak-setter fix) —
            // otherwise the strong count is permanently one too high and the
            // block can never reach zero. `strong_release_ctrl` needs the data
            // range only on the last-owner path; re-read it just for that case,
            // tolerating a second failure there (a bounded, already-permitted
            // leak, unlike the unbounded over-count this guards against).
            if refcount::fetch_sub(stack, strong_off, 1)? == 1 {
                let mut retry = [0u8; 8];
                if stack.get_into(data_pos, &mut retry).is_ok() {
                    let data_range =
                        BStackRange::new(u64::from_le_bytes(retry), size_of::<T::OnDisk>() as u64);
                    let _ = strong_release_ctrl::<T, A>(allocator, data_range, ctrl_range);
                }
            }
            return Err(e);
        }
        let data_range = BStackRange::new(u64::from_le_bytes(bytes), size_of::<T::OnDisk>() as u64);
        let data = unsafe { BStackRef::<T>::from_range(data_range) };
        // SAFETY: the increment above claimed the strong count this handle holds.
        Ok(Some(unsafe {
            BStackRc::from_raw(data, Some(ctrl_range), allocator)
        }))
    }
}

/// Cloning a weak handle bumps the control block's weak count and returns
/// another weak handle to the **same** control block. This is the *only* sound
/// meaning of a weak clone: a weak reference observes a live object's control
/// block, and a copy that observed anything else would not be observing what the
/// original does. So a weak clone shares the observation (a count bump) rather
/// than deep-copying — there is no `TryCloneIn` for a weak reference.
impl<'a, T: BStackWeakable, A: BStackRaiiAllocator> TryClone for BStackWeak<'a, T, A> {
    fn try_clone(&self) -> io::Result<Self> {
        let weak_off = NonNullOffset::from_field(checked_off(
            self.ctrl().into_range().start(),
            CTRL_WEAK_OFFSET,
        )?)?;
        refcount::fetch_add(self.allocator().stack(), weak_off, 1)?;
        // SAFETY: the fetch_add above established the weak count this clone holds.
        Ok(unsafe { Self::from_raw(self.ctrl(), self.allocator()) })
    }
}

// ---------------------------------------------------------------------------
// `(rc, weak)` build & field-mutation machinery.
//
// The *build* side of the control block (the drop-core `*Ref` tokens above are
// the teardown side), plus the runtime behind a generated `#[bstack_weak]`
// field's setter/upgrade. All of it speaks only rc/weak vocabulary — the
// control-block offsets, `WeakRef`, `BStackWeak`, `BStackRc` — so it lives here.
// ---------------------------------------------------------------------------

/// Build a `(rc, weak)` control-block payload image in memory (no allocation, no
/// write): header, `strong = 1`, `weak = 1` (the phantom weak the strong owners
/// hold), and the `x` forward pointer to the data block at `data_start`.
///
/// The building block for a **batched** constructor: the caller allocates the
/// data and control blocks up front, bakes the control offset into the data
/// block's `ctrl` back-pointer, and commits both block images in one
/// [`bstack::BStack::set_batched`] — so a `(rc, weak)` block is created atomically,
/// with no separate back-pointer write and no transient half-wired state.
pub fn build_control_payload(ctrl_tag: EightCC, data_start: u64) -> [u8; CONTROL_SIZE as usize] {
    // The control block is a fixed [`CONTROL_SIZE`]-byte layout, so the image is a
    // stack buffer, no heap allocation. The caller writes it (borrowed as `&[u8]`)
    // straight into its batched commit.
    let mut payload = [0u8; CONTROL_SIZE as usize];
    let header = BlockHeader {
        size: CONTROL_SIZE,
        tag: ctrl_tag,
    };
    payload[..HEADER_SIZE as usize].copy_from_slice(bytemuck::bytes_of(&header));
    put_u64(&mut payload, CTRL_STRONG_OFFSET, 1);
    put_u64(&mut payload, CTRL_WEAK_OFFSET, 1);
    put_u64(&mut payload, CTRL_DATA_OFFSET, data_start);
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
pub(crate) unsafe fn set_weak_field<'w, T: BStackWeakable, A: BStackRaiiAllocator>(
    allocator: &'w A,
    field_off: NonNullOffset,
    new_weak: BStackWeak<'w, T, A>,
) -> Result<(), ReplaceError<BStackWeak<'w, T, A>>> {
    // Serialize against a concurrent `upgrade_weak_field` on the same field: the
    // old control block is released (and possibly freed) below, and a racing
    // upgrade — which holds no weak count to pin it — would otherwise increment a
    // counter in freed storage. Both take this per-file lock.
    let lock = crate::io_core::wal_lock_for(allocator);
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
    let old_bytes = match stack.swap(field_off.as_u64(), ctrl_off.to_le_bytes()) {
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
pub(crate) unsafe fn upgrade_weak_field<'a, T: BStackWeakable, A: BStackRaiiAllocator>(
    allocator: &'a A,
    field_off: NonNullOffset,
) -> io::Result<Option<BStackRc<'a, T, A>>> {
    // Hold the per-file lock across the read of the control offset and the pin
    // (`increment_if_nonzero`), so a concurrent `set_weak_field` can't free the old
    // control block between the two steps. The field slot is not
    // owned here, so nothing else keeps that block alive.
    let lock = crate::io_core::wal_lock_for(allocator);
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    let off = read_u64(allocator.stack(), field_off.as_u64())?;
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
