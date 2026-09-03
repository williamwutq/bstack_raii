//! [`Foreign<T>`]: a cross-file pointer — "a slice with a file identity attached".
//!
//! An in-file reference stores just a `u64` offset (length recovered from
//! `size_of::<T::OnDisk>()`). A `Foreign<T>` widens that with the target file's
//! identity: on disk it is an inert wire record `{ file_id: u64, address: u64 }` (16
//! bytes, `Pod`) — length is still recovered from the type, so it is *not* stored.
//! Dereferencing resolves `file_id` through the process-wide
//! [registry](crate::registry) to the file's live allocator, then reads/writes at
//! `offset` in that file.
//!
//! It is the cross-file sibling of [`BStackRef`](super::reference::BStackRef) (the
//! in-file typed pointer), so it lives here among the reference vocabulary — the
//! *type only*. As a `#[bstack_block]` field it carries the same ownership
//! annotations as an in-file field — `#[bstack_owned/strong/weak/ref]` (or none) —
//! applied to the target `T` **in its own file**. The with-allocator handles a
//! `bstack_move!` of an owning foreign field hands back — [`ForeignOwned`](crate::ForeignOwned)
//! / [`ForeignRc`](crate::ForeignRc) / [`ForeignWeak`](crate::ForeignWeak) — live in
//! [`crate::types::compiled::foreign`]; the cross-file teardown / deep-clone
//! *mechanism* they and the generated field code drive is in
//! [`crate::io_core::foreign`]. `#[bstack_ref]` aliases (byte-copied, owns nothing).
//! Construction, nullability (`Option<Foreign<T>>`, `offset == 0` niche), and
//! resolution are here.
//!
//! **Cross-file atomicity is best-effort**, and inherently so: two independent
//! bstack files have no shared commit, so a deep clone / teardown that spans the home
//! file and a foreign file cannot be one atomic unit. Each *file's own* commit is
//! atomic (the foreign side runs its own crash-safe `try_clone_in` / teardown through
//! the adapter), and the ordering always errs toward **over-provisioning** — an
//! orphaned fresh block or an over-count, which leaks — never toward an under-count
//! (a premature free / double-free). A target file not attached at the time makes an
//! owning clone *error* rather than silently alias.

use core::marker::PhantomData;
use std::io;

use bstack::{BStack, BStackAllocator, BStackRange, BStackSlice};

use super::{BStackBlock, BStackRef};
use crate::primitives::{BrandedWidePtr, Offset, WidePtr};
#[cfg(test)]
use crate::registry::FileRegistry;
use crate::registry::{self, FileId};
use crate::util::io_error;

/// A **typed cross-file pointer** to a `T` block in another `bstack` file — the
/// self-qualifying counterpart of [`BStackRef`](super::BStackRef) (an in-file offset with
/// no file identity); see the [module docs](self) for its on-disk wire form.
///
/// It is one of two kinds: an **explicit** pointer (carries the target's
/// [`FileId`](crate::registry::FileId), so it resolves through the
/// process-wide [registry](crate::registry), borrow-free, deref fallible) or
/// [`SELF`](FileId::SELF) (an address in the file it was read from, bound to that file's
/// borrow `'a`).
///
/// `Copy`. An explicit pointer can be [`detach`](Self::detach)ed to a `'static`,
/// borrow-free `Foreign`; a `SELF` pointer cannot (it is only valid within the scope of
/// the file it was read from). `T: 'static` because a persisted block target holds no
/// in-memory borrow. See the [module docs](self) for teardown / deep-clone semantics.
pub struct Foreign<'a, T: 'static> {
    inner: BrandedWidePtr<'a>,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: 'static> Clone for Foreign<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T: 'static> Copy for Foreign<'a, T> {}

impl<'a, T: 'static> Foreign<'a, T> {
    /// Reconstruct from the stored on-disk [`WidePtr`]. A `SELF` pointer stays
    /// borrow-bound to `'a`; an explicit one ignores the borrow
    /// ([`detach`](Self::detach) it to escape).
    ///
    /// # Safety
    /// Two obligations:
    /// * `repr` must be a pointer previously stored into this file — it names a valid
    ///   `T` (explicit ⇒ in its own file; `SELF` ⇒ in the file it was read from).
    /// * the caller must bind the returned `Foreign<'a, T>`'s lifetime `'a` to that
    ///   file's borrow (a generated field accessor does this by tying `'a` to the
    ///   `&'a BStack` / `&'a A` it read through), so a `SELF` pointer cannot escape it.
    pub unsafe fn from_repr(repr: WidePtr) -> Self {
        Self {
            inner: BrandedWidePtr::from_wide(repr),
            _marker: PhantomData,
        }
    }

    /// The on-disk wire pointer, for storing into a field. Preserves the RTTI
    /// `type_index` the pointer was read with (so a round-trip keeps it typed).
    pub fn repr(self) -> WidePtr {
        self.inner.wide()
    }

    /// The target's address within its file.
    pub fn offset(self) -> u64 {
        self.inner.offset().get()
    }

    /// Whether this points into the *current* file ([`FileId::SELF`]).
    pub fn is_self(self) -> bool {
        self.inner.is_self()
    }

    /// The file this points into: [`SELF`](FileId::SELF) for a `SELF` pointer, else the
    /// explicit target file.
    pub fn file_id(self) -> FileId {
        self.inner.file()
    }

    /// Promote an **explicit** pointer to a `'static`, borrow-free [`Foreign`] — the
    /// registry-resolved form that can be stored anywhere and outlives any file handle
    /// (its deref stays fallible). `None` for a `SELF` pointer, which is only valid
    /// within the scope of the file it was read from.
    pub fn detach(self) -> Option<Foreign<'static, T>> {
        if self.inner.is_self() {
            None
        } else {
            // Explicit ⇒ borrow-free: rebrand the identical bytes to `'static`.
            Some(Foreign {
                inner: BrandedWidePtr::from_wide(self.inner.wide()),
                _marker: PhantomData,
            })
        }
    }
}

impl<T: 'static> Foreign<'static, T> {
    /// A raw explicit foreign pointer to `offset` within `file`. The sound way to obtain
    /// a `Foreign` is by reading a `#[bstack_block]` field (borrow-bound) or via
    /// [`from_local`](Self::from_local) / [`at`](Self::at); this raw form is for tests
    /// and low-level code.
    ///
    /// # Safety
    /// `offset` names a valid `T` block in `file`. If `file` is [`SELF`](FileId::SELF)
    /// the result is a `'static` `SELF` pointer whose deref against the wrong file reads
    /// the wrong data — the caller must resolve it only against its home file. (A sound
    /// `SELF` pointer is borrow-bound; read it from its field instead.)
    pub const unsafe fn new(file: FileId, offset: u64) -> Self {
        Self {
            // A raw pointer carries no RTTI ordinal (the interpreter recovers the type
            // from the target's block header), so [`BrandedWidePtr::new`] leaves it
            // untyped. `file == SELF` (0) is the `SELF` key.
            inner: BrandedWidePtr::new(file, Offset::from_raw(offset)),
            _marker: PhantomData,
        }
    }
}

impl<'a, T: BStackBlock + 'static> Foreign<'a, T> {
    /// The target's range in its file (`address` + `size_of::<T::OnDisk>()`).
    pub fn range(self) -> BStackRange {
        BStackRange::new(self.offset(), core::mem::size_of::<T::OnDisk>() as u64)
    }

    /// Resolve the pointer and run `f` with a `T` handle at the target plus the
    /// [`BStack`] of the file it lives in.
    ///
    /// The two failure modes are kept apart rather than conflated into one
    /// `Option`: `Ok(None)` is the [null niche](self) (`address == 0`, a genuinely
    /// absent pointer — not an error, same as reading a null `#[bstack_ref]`
    /// field); `Err` is an I/O-shaped [`io::ErrorKind::NotFound`] meaning the
    /// pointer is non-null but its target file can't currently be reached (a
    /// malformed / out-of-range file id, or a file that is unknown / not
    /// currently attached to the [registry](crate::registry)).
    ///
    /// There is exactly one registry — the process-wide one — so an explicit pointer
    /// (e.g. one moved out via `bstack_move!`) is always resolvable on its own; `local`
    /// is used only to resolve a [`SELF`](FileId::SELF) pointer (no registry, no lock).
    pub fn with<A, R>(self, local: &A, f: impl FnOnce(T, &BStack) -> R) -> io::Result<Option<R>>
    where
        A: BStackAllocator,
    {
        if self.offset() == 0 {
            return Ok(None);
        }
        let t = unsafe { T::from_range(self.range()) };
        if self.inner.is_self() {
            Ok(Some(f(t, local.stack())))
        } else {
            let id = self.inner.file();
            registry::with_host(id, |host| f(t, host.stack()))
                .ok_or_else(|| io_error!(NotFound, "Foreign: target file not attached"))
                .map(Some)
        }
    }

    /// **foreign → normal** (`bstack_cast!(foreign as BStackRef<T>)`): the offset-only
    /// [`BStackRef`] to the target, valid **in the target's own file**. `Some` iff
    /// that file is [`SELF`](FileId::SELF) or currently attached (so the ref is
    /// resolvable); `None` otherwise. Does no I/O — pair the ref with the target
    /// file's stack (e.g. via [`with`](Self::with)) to read it.
    pub fn into_local(self) -> Option<BStackRef<T>> {
        let resolvable = if self.inner.is_self() {
            true
        } else {
            registry::get().is_some_and(|r| r.is_live(self.inner.file()))
        };
        // SAFETY: `range()` is this pointer's target region; the returned ref is a
        // plain offset handle (no aliasing/liveness claim beyond the caller's).
        resolvable.then(|| unsafe { BStackRef::from_range(self.range()) })
    }

    /// Like [`with`](Self::with) but against an explicit `registry` — crate-internal,
    /// for tests (the global is a one-shot `OnceLock`, awkward to exercise in unit
    /// tests). Production code uses [`with`](Self::with) against the sole registry.
    /// This may seems like it's code duplication but it's for testing purposes
    #[cfg(test)]
    pub(crate) fn with_in<A, R>(
        self,
        registry: &FileRegistry,
        local: &A,
        f: impl FnOnce(T, &BStack) -> R,
    ) -> io::Result<Option<R>>
    where
        A: BStackAllocator,
    {
        if self.offset() == 0 {
            return Ok(None);
        }
        let t = unsafe { T::from_range(self.range()) };
        if self.inner.is_self() {
            Ok(Some(f(t, local.stack())))
        } else {
            let id = self.inner.file();
            registry
                .with_host(id, |host| f(t, host.stack()))
                .ok_or_else(|| io_error!(NotFound, "Foreign: target file not attached"))
                .map(Some)
        }
    }
}

impl<T: BStackBlock + 'static> Foreign<'static, T> {
    /// A **`SELF`** foreign pointer to the block `target` currently occupies.
    ///
    /// # Safety
    /// A `SELF` pointer stores only `target`'s **offset** — never its file — and
    /// resolves against whatever file it is later stored in and read back from. It is
    /// therefore sound only while it lives in `target`'s **own** file. `target: &T` is a
    /// bare offset handle that carries no file identity, so this cannot be checked: the
    /// caller asserts that the returned pointer is only ever stored into / resolved
    /// against `target`'s home file. Storing it into a block in a *different* file
    /// persists a `{file_id: 0, offset}` whose offset names nothing valid there — a later
    /// owning teardown / deep clone then frees or copies **that offset in the wrong
    /// file**, corrupting an unrelated block from otherwise-safe code.
    ///
    /// For a **safe** pointer to a local block, use [`from_local`](Self::from_local) /
    /// `bstack_cast!(slice as Foreign<T>)`, which resolves the block's file to its
    /// registered [`FileId`] through the registry — an *explicit* id that routes correctly
    /// no matter where the pointer is later stored.
    pub unsafe fn at(target: &T) -> Self {
        // SAFETY: the caller upholds that this `SELF` pointer stays in `target`'s home
        // file; `target` is a live `T` handle, so its start names a valid `T` there.
        unsafe { Self::new(FileId::SELF, target.range().start()) }
    }

    /// **normal → foreign** (`bstack_cast!(slice as Foreign<T>)`): name the block a
    /// `BStackSlice` points at as a `Foreign`, resolving the slice's file to its
    /// [`FileId`] via the registry's reverse map. `None` if that file is not
    /// currently attached (so it has no id to name). Does no I/O.
    pub fn from_local(slice: &BStackSlice<'_>) -> Option<Self> {
        let id = registry::id_of_host(slice.stack())?;
        // SAFETY: `slice` names a live block in a registered file (`id`).
        Some(unsafe { Self::new(id, slice.start()) })
    }
}
