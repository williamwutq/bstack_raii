//! The reference-counting capability contracts: [`BStackShared`] (any block that
//! can be a `#[bstack_strong]` target) and [`BStackWeakable`] (the `(rc, weak)`
//! blocks that additionally carry a control block and admit weak references).

use std::io;

use bstack::BStackRange;
use bytemuck::Pod;

use super::super::compiled::rc::{BStackRc, BStackWeak};
use super::block::BStackBlock;
use super::reference::BStackRef;
use crate::BStackRaiiAllocator;
use crate::primitives::{EightCC, NonNullOffset};
use crate::replace::ReplaceError;

/// Implemented by refcounted blocks (`#[bstack_block(rc)]` and
/// `#[bstack_block(rc, weak)]`), i.e. any block that can be the target of a
/// `#[bstack_strong]` field.
///
/// It abstracts "drop one strong reference to a child of this type" so a
/// parent's generated teardown does not need to know whether the child is a
/// plain `(rc)` block (inline refcount, [`crate::StrongRef`]) or an
/// `(rc, weak)` block (control block, [`crate::StrongWeakRef`]). The child's own
/// `#[bstack_block]` expansion picks the right implementation.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a reference-counted block (`#[bstack_block(rc)]` / `(rc, weak)`)",
    label = "not a shared block",
    note = "`#[bstack_strong]` fields (and generic parameters used as them) require an \
            `#[bstack_block(rc)]` or `#[bstack_block(rc, weak)]` type."
)]
pub trait BStackShared: BStackBlock {
    /// Drop one strong reference to a block of this type located at `data`,
    /// freeing it (and, for `(rc, weak)`, releasing the control block) when the
    /// strong count reaches zero.
    fn drop_strong_ref<A: BStackRaiiAllocator>(
        data: BStackRef<Self>,
        allocator: &A,
    ) -> io::Result<()>;

    /// Resolve the raw parts of a strong handle to a child of this type at
    /// `data`: the data ref, plus the control-block range for `(rc, weak)`
    /// blocks (`None` for plain `(rc)`). Used by `bstack_move!` to rebuild a
    /// `BStackRc` for a `#[bstack_strong]` field.
    fn strong_parts<A: BStackRaiiAllocator>(
        data: BStackRef<Self>,
        allocator: &A,
    ) -> io::Result<(BStackRef<Self>, Option<BStackRange>)>;

    /// Cross-file **teardown** of an `#[bstack_strong] Foreign<Self>` target:
    /// decrement the strong count at `offset` (the target's data block), freeing at
    /// zero, in the file `alloc` addresses.
    ///
    /// # Safety
    /// `offset` names a live shared `Self` data block in the file `alloc` addresses,
    /// holding one strong reference on behalf of this foreign pointer.
    #[doc(hidden)]
    unsafe fn foreign_drop_strong<A: BStackRaiiAllocator>(
        alloc: &A,
        offset: NonNullOffset,
    ) -> io::Result<()> {
        // SAFETY: forwarded to the caller's contract above.
        unsafe { crate::foreign::foreign_drop_strong::<Self, A>(alloc, offset) }
    }

    /// Cross-file **clone** of an `#[bstack_strong] Foreign<Self>` reference: bump the
    /// target's strong count at `offset` in the file `alloc` addresses.
    ///
    /// # Safety
    /// `offset` names a live shared `Self` data block in the file `alloc` addresses.
    #[doc(hidden)]
    unsafe fn foreign_clone_strong<A: BStackRaiiAllocator>(
        alloc: &A,
        offset: NonNullOffset,
    ) -> io::Result<()> {
        // SAFETY: forwarded to the caller's contract above.
        unsafe { crate::foreign::foreign_clone_strong::<Self, A>(alloc, offset) }
    }
}

/// Implemented only for blocks declared `#[bstack_block(rc, weak)]`.
///
/// Its presence is what lets [`crate::BStackRc`] expose `downgrade` and
/// [`crate::BStackWeak`] exist for the type. `Control` is the generated
/// `XOnDiskRef` control-block payload holding the `strong`/`weak` counters.
/// Plain `#[bstack_block(rc)]` blocks do not implement it, so weak references to
/// them are a compile error rather than a runtime hazard.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a weak-observable block (`#[bstack_block(rc, weak)]`)",
    label = "not a weakable block",
    note = "`#[bstack_weak]` fields (and generic parameters used as them) require an \
            `#[bstack_block(rc, weak)]` type; a plain `#[bstack_block(rc)]` is not weak-observable."
)]
pub trait BStackWeakable: BStackBlock {
    /// The on-disk control-block payload (the generated `XOnDiskRef`).
    type Control: Pod;
    /// The control block's [`EightCC`] tag — the tag stamped into the control
    /// (`XOnDiskRef`) header, distinct from the data tag ([`BStackCast::eightcc`]).
    /// Lets a validator confirm a region *is* this type's control block directly by
    /// its header, not only indirectly via its forward data pointer.
    fn control_eightcc() -> EightCC;

    /// Cross-file **teardown** of a `#[bstack_weak] Foreign<Self>` target: decrement
    /// the weak count in the *control* block at `ctrl_offset`, freeing it at zero, in
    /// the file `alloc` addresses. The data block is never touched.
    ///
    /// # Safety
    /// `ctrl_offset` names a live `Self::Control` block in the file `alloc` addresses,
    /// holding one weak reference on behalf of this foreign pointer.
    #[doc(hidden)]
    unsafe fn foreign_drop_weak<A: BStackRaiiAllocator>(
        alloc: &A,
        ctrl_offset: NonNullOffset,
    ) -> io::Result<()> {
        // SAFETY: forwarded to the caller's contract above.
        unsafe { crate::foreign::foreign_drop_weak::<Self, A>(alloc, ctrl_offset) }
    }

    /// Cross-file **clone** of a `#[bstack_weak] Foreign<Self>` reference: bump the
    /// target's weak count in the control block at `ctrl_offset` in the file `alloc`
    /// addresses.
    ///
    /// # Safety
    /// `ctrl_offset` names a live `Self::Control` block in the file `alloc` addresses.
    #[doc(hidden)]
    unsafe fn foreign_clone_weak<A: BStackRaiiAllocator>(
        alloc: &A,
        ctrl_offset: NonNullOffset,
    ) -> io::Result<()> {
        // SAFETY: forwarded to the caller's contract above.
        unsafe { crate::foreign::foreign_clone_weak::<Self, A>(alloc, ctrl_offset) }
    }

    /// Install `new_weak` into the `#[bstack_weak]` field at `field_off`, releasing
    /// the previous target's weak reference. What a generated weak-field setter calls.
    ///
    /// # Safety
    /// `field_off` names a live `#[bstack_weak]` field of target type `Self`, owned by
    /// a block in `allocator`'s file (see the crate's field-offset contract).
    #[doc(hidden)]
    unsafe fn set_weak_field<'w, A: BStackRaiiAllocator>(
        allocator: &'w A,
        field_off: NonNullOffset,
        new_weak: BStackWeak<'w, Self, A>,
    ) -> Result<(), ReplaceError<BStackWeak<'w, Self, A>>> {
        // SAFETY: forwarded to the caller's contract above.
        unsafe { crate::construct::set_weak_field::<Self, A>(allocator, field_off, new_weak) }
    }

    /// Attempt to upgrade the `#[bstack_weak]` field at `field_off` to a strong
    /// [`BStackRc`]. `None` if the field is unset or the target's strong count is
    /// already zero. What a generated weak-field accessor calls.
    ///
    /// # Safety
    /// `field_off` names a live `#[bstack_weak]` field of target type `Self` in
    /// `allocator`'s file.
    #[doc(hidden)]
    unsafe fn upgrade_weak_field<'a, A: BStackRaiiAllocator>(
        allocator: &'a A,
        field_off: NonNullOffset,
    ) -> io::Result<Option<BStackRc<'a, Self, A>>> {
        // SAFETY: forwarded to the caller's contract above.
        unsafe { crate::construct::upgrade_weak_field::<Self, A>(allocator, field_off) }
    }
}
