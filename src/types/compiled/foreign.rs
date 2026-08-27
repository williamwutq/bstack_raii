//! [`ForeignOwned`] / [`ForeignRc`] / [`ForeignWeak`]: the cross-file RAII duals of
//! [`BStackOwned`](crate::BStackOwned) / [`BStackRc`](crate::BStackRc) /
//! [`BStackWeak`](crate::BStackWeak), for a target reached through a
//! [`Foreign`](crate::Foreign) pointer.
//!
//! `bstack_move!` of a `#[bstack_owned/strong/weak] Foreign<T>` field hands back one
//! of these; a `#[bstack_ref]` field hands back a plain `Foreign` (which owns
//! nothing). Like `BStackOwned`, they do **not** free on `Drop` (freeing needs an
//! allocator): call `bstack_drop(&home)` to release the target in its own file, or
//! `into_foreign()` to relinquish ownership and re-store the raw pointer. Not `Copy`
//! — dropping one without `bstack_drop` leaks the target. The `bstack_drop` dispatch
//! resolves the target's file (`SELF` ⇒ `home`, else the registry's
//! [`ForeignHostAllocator`](crate::registry::ForeignHostAllocator)) and runs the
//! per-kind teardown in [`crate::io_core::foreign`].

use std::io;

use bstack::{BStack, BStackAllocator, BStackRange};

use super::super::traits::{BStackBlock, BStackRef, BStackShared, BStackWeakable, Foreign};
use super::{BStackOwned, BStackRc, BStackWeak};
use crate::BStackRaiiAllocator;
use crate::io_core::foreign::{foreign_drop_owned, foreign_drop_strong, foreign_drop_weak};
use crate::registry::{self, FileId};

/// The RAII dual of [`BStackOwned`](crate::BStackOwned) for a target owned through a
/// [`Foreign`] pointer. [`bstack_drop`](Self::bstack_drop) deep-frees the target in its
/// own file; [`into_foreign`](Self::into_foreign) relinquishes ownership to re-store it.
pub struct ForeignOwned<'a, T: 'static> {
    ptr: Foreign<'a, T>,
}

impl<'a, T: 'static> ForeignOwned<'a, T> {
    /// Wrap a pointer as a uniquely-owning handle (ownership transfers in).
    ///
    /// # Safety
    /// `ptr` must be the sole owning pointer to a live `T` — freed exactly once.
    pub unsafe fn from_foreign(ptr: Foreign<'a, T>) -> Self {
        Self { ptr }
    }
    /// Borrow the underlying pointer (e.g. to read the target via [`Foreign::with`]).
    pub fn as_foreign(&self) -> Foreign<'a, T> {
        self.ptr
    }
    /// Relinquish ownership, returning the raw pointer to re-store into another owning
    /// field (ownership transfers to wherever it is next stored).
    pub fn into_foreign(self) -> Foreign<'a, T> {
        self.ptr
    }
    /// Whether the target is in the current file ([`SELF`](FileId::SELF)).
    pub fn is_self(&self) -> bool {
        self.ptr.is_self()
    }
}

/// Run a `Foreign` reference's per-kind teardown against **the file its target
/// lives in**, selected by the pointer's `file_id`: `home` for a `SELF` pointer
/// (`fid == 0`), else the [`ForeignHostAllocator`](registry::ForeignHostAllocator)
/// of the registered file `fid` names. A null pointer (`offset == 0`), a detached
/// target file, or a malformed id are each a permitted leak (`Ok(())`), never a
/// panic. `$drop` is the kind's `foreign_drop_{owned,strong,weak}` helper.
///
/// # Safety
/// `$repr` must be the reference held by `self` (transferred in by `from_foreign`),
/// consumed exactly once — so `$drop` accounts for exactly the one reference the
/// caller held, in the target's own file.
macro_rules! foreign_dispatch_drop {
    ($repr:expr, $home:expr, $drop:path) => {{
        let repr = $repr;
        let off = repr.offset().get();
        if off == 0 {
            Ok(())
        } else if repr.file_id() == 0 {
            // `SELF` ⇒ the home file.
            unsafe { $drop($home, $crate::primitives::NonNullOffset::from_field(off)?) }
        } else if let Some(id) = FileId::from_u64(repr.file_id()) {
            if let Some(host) = registry::host_arc(id) {
                let adapter = registry::ForeignHostAllocator::new(host, id);
                unsafe {
                    $drop(
                        &adapter,
                        $crate::primitives::NonNullOffset::from_field(off)?,
                    )
                }
            } else {
                Ok(()) // target file detached ⇒ leak (permitted)
            }
        } else {
            Ok(()) // malformed id ⇒ unreachable target ⇒ leak
        }
    }};
}

impl<'a, T: BStackBlock + 'static> ForeignOwned<'a, T> {
    /// Deep-free the owned target in its own file (`SELF` ⇒ `home`, else registry).
    pub fn bstack_drop<A: BStackRaiiAllocator>(self, home: &A) -> io::Result<()> {
        // SAFETY: sole owner (`from_foreign` contract), consumed here exactly once.
        foreign_dispatch_drop!(self.ptr.repr(), home, foreign_drop_owned::<T, _>)
    }

    /// Read the target — convenience for `self.as_foreign().with(local, f)`.
    pub fn with<A, R>(&self, local: &A, f: impl FnOnce(T, &BStack) -> R) -> io::Result<Option<R>>
    where
        A: BStackAllocator,
    {
        self.ptr.with(local, f)
    }

    /// Resolve to a plain [`BStackOwned<T>`](crate::BStackOwned) — the in-file owning
    /// handle — **valid in the target's own file**. Consumes `self`, transferring the
    /// sole ownership to the returned handle (so there is never a second owner). The
    /// owning analogue of [`Foreign::into_local`], with its siblings'
    /// ([`ForeignRc::into_local`] / [`ForeignWeak::into_local`]) target-binding
    /// signature: `target` names the file the returned offset-only handle will be
    /// used against, and an **explicit**-`FileId` pointer is checked against it —
    /// a mismatch is rejected (`InvalidInput`) instead of handing back a handle
    /// whose safe `bstack_drop` would free that offset in the wrong file.
    ///
    /// A [`SELF`](FileId::SELF) pointer carries no file identity to check (it is
    /// only meaningful in the file it was read from — pass that file's allocator);
    /// the caller keeps that obligation, exactly as for a cast-produced
    /// [`BStackRef`].
    pub fn into_local<A: BStackRaiiAllocator>(self, target: &A) -> io::Result<BStackOwned<T>> {
        let fid = self.ptr.repr().file_id();
        // NOTE: see previous note
        if fid != 0 {
            // Resolve `target`'s identity: its adapter-declared id (a
            // `ForeignHostAllocator`), or the registry's reverse map for a plain
            // allocator over an attached file.
            let target_id = match target.wal_file_id() {
                FileId::SELF => registry::id_of_host(target.stack()),
                id => Some(id),
            };
            if target_id.map(FileId::as_u64) != Some(fid) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "ForeignOwned::into_local: the pointer's home file is not the \
                     given target allocator's file",
                ));
            }
        }
        // SAFETY: `self` was the sole owner (its `from_foreign` contract) and is
        // consumed here, so the returned `BStackOwned` becomes the sole owner of the
        // same live block, in the target's own file.
        Ok(unsafe { BStackOwned::from_raw(T::from_range(self.ptr.range())) })
    }
}

/// The RAII dual of [`BStackRc`](crate::BStackRc) for a **strong** reference held
/// through a [`Foreign`] pointer. [`bstack_drop`](Self::bstack_drop) decrements the
/// target's strong count in its own file (freeing at zero).
pub struct ForeignRc<'a, T: 'static> {
    ptr: Foreign<'a, T>,
}

impl<'a, T: 'static> ForeignRc<'a, T> {
    /// Wrap a pointer as a strong-reference handle (one strong ref transfers in).
    ///
    /// # Safety
    /// `ptr` must hold exactly one strong reference on a live shared `T` — released
    /// exactly once.
    pub unsafe fn from_foreign(ptr: Foreign<'a, T>) -> Self {
        Self { ptr }
    }
    /// Borrow the underlying pointer.
    pub fn as_foreign(&self) -> Foreign<'a, T> {
        self.ptr
    }
    /// Relinquish the strong reference, returning the raw pointer to re-store.
    pub fn into_foreign(self) -> Foreign<'a, T> {
        self.ptr
    }
    /// Whether the target is in the current file ([`SELF`](FileId::SELF)).
    pub fn is_self(&self) -> bool {
        self.ptr.is_self()
    }
}

impl<'a, T: BStackShared + 'static> ForeignRc<'a, T> {
    /// Decrement the target's strong count in its own file (`SELF` ⇒ `home`, else
    /// registry); the target is freed when the count reaches zero.
    pub fn bstack_drop<A: BStackRaiiAllocator>(self, home: &A) -> io::Result<()> {
        // SAFETY: one strong ref (`from_foreign` contract), consumed here exactly once.
        foreign_dispatch_drop!(self.ptr.repr(), home, foreign_drop_strong::<T, _>)
    }

    /// Read the target — convenience for `self.as_foreign().with(local, f)`.
    pub fn with<A, R>(&self, local: &A, f: impl FnOnce(T, &BStack) -> R) -> io::Result<Option<R>>
    where
        A: BStackAllocator,
    {
        self.ptr.with(local, f)
    }

    /// Resolve to a live [`BStackRc<T>`](crate::BStackRc) bound to `target` — the in-file
    /// **strong** handle for the target's own file. Consumes `self`, transferring the
    /// single strong reference. `target` must address the target's file (the home
    /// allocator for a [`SELF`](FileId::SELF) target, or the target file's host — e.g. a
    /// [`ForeignHostAllocator`](crate::registry::ForeignHostAllocator) — for a cross-file
    /// one). It is read to recover the `(rc, weak)` control block, so this is fallible.
    pub fn into_local<'t, A: BStackRaiiAllocator>(
        self,
        target: &'t A,
    ) -> io::Result<BStackRc<'t, T, A>> {
        // SAFETY: the stored offset is the target's live shared data block.
        let data = unsafe { BStackRef::<T>::from_range(self.ptr.range()) };
        let (data, ctrl) = <T as BStackShared>::strong_parts(data, target)?;
        // SAFETY: `self` held one strong ref (its `from_foreign` contract) and is
        // consumed, so the returned handle accounts for exactly that count.
        Ok(unsafe { BStackRc::from_raw(data, ctrl, target) })
    }
}

/// The RAII dual of [`BStackWeak`](crate::BStackWeak) for a **weak** reference held
/// through a [`Foreign`] pointer (it stores the target's *control* offset).
/// [`bstack_drop`](Self::bstack_drop) decrements the weak count in its own file.
pub struct ForeignWeak<'a, T: 'static> {
    ptr: Foreign<'a, T>,
}

impl<'a, T: 'static> ForeignWeak<'a, T> {
    /// Wrap a pointer as a weak-reference handle (one weak ref transfers in).
    ///
    /// # Safety
    /// `ptr` must hold exactly one weak reference on a live weakable `T`'s control
    /// block — released exactly once.
    pub unsafe fn from_foreign(ptr: Foreign<'a, T>) -> Self {
        Self { ptr }
    }
    /// Borrow the underlying pointer.
    pub fn as_foreign(&self) -> Foreign<'a, T> {
        self.ptr
    }
    /// Relinquish the weak reference, returning the raw pointer to re-store.
    pub fn into_foreign(self) -> Foreign<'a, T> {
        self.ptr
    }
    /// Whether the target is in the current file ([`SELF`](FileId::SELF)).
    pub fn is_self(&self) -> bool {
        self.ptr.is_self()
    }
}

impl<'a, T: BStackWeakable + 'static> ForeignWeak<'a, T> {
    /// Decrement the target's weak count in its own file (`SELF` ⇒ `home`, else
    /// registry).
    pub fn bstack_drop<A: BStackRaiiAllocator>(self, home: &A) -> io::Result<()> {
        // SAFETY: one weak ref (`from_foreign` contract), consumed here exactly once.
        foreign_dispatch_drop!(self.ptr.repr(), home, foreign_drop_weak::<T, _>)
    }

    /// Resolve to a live [`BStackWeak<T>`](crate::BStackWeak) bound to `target` — the
    /// in-file **weak** handle for the target's own file. Consumes `self`, transferring
    /// the single weak reference. `target` must address the target's file (the home
    /// allocator for a [`SELF`](FileId::SELF) target, else the target file's host).
    /// Infallible: a weak handle only names the control block.
    pub fn into_local<'t, A: BStackRaiiAllocator>(self, target: &'t A) -> BStackWeak<'t, T, A> {
        let ctrl_off = self.ptr.offset();
        // SAFETY: a weak `Foreign` stores the target's control-block offset.
        let ctrl = unsafe {
            BStackRef::<T::Control>::from_range(BStackRange::new(
                ctrl_off,
                core::mem::size_of::<T::Control>() as u64,
            ))
        };
        // SAFETY: `self` held one weak ref (its `from_foreign` contract) and is
        // consumed, so the returned handle accounts for exactly that count.
        unsafe { BStackWeak::from_raw(ctrl, target) }
    }
}
