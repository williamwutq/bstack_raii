//! [`BStackRaiiHost`] — the object-safe, cross-file projection of the allocator
//! capability, and its [`BStackRaiiAllocError`].
//!
//! A `Foreign<T>` reaches *into another file* through a type-erased
//! `Arc<dyn BStackRaiiHost>` (the concrete allocator type is not known at the use
//! site). This trait is that erased view; the stateful path↔id map that resolves a
//! `FileId` to a live host lives in [`crate::registry`].

use std::fmt;
use std::io;

use bstack::{BStack, BStackAllocator, BStackOwnedSlice, BStackRange};

use super::{BStackRaiiAllocator, SyncBStackRaiiAllocator};
use crate::handback::impl_source_error;
use crate::primitives::NonNullOffset;

/// Error returned by [`BStackRaiiHost::realloc`] / [`BStackRaiiHost::dealloc`] when
/// the operation fails — the object-safe, range-based analogue of bstack's
/// `BStackAllocError`.
///
/// A failed resize or free almost always leaves a valid allocation behind — the
/// original region untouched, or the new region fully committed. This type carries
/// that surviving region's range back to the caller so it can retry, fall back, or
/// explicitly [`dealloc`](BStackRaiiHost::dealloc) it rather than leak it. Because a
/// bare [`BStackRange`] carries no ownership or `Drop`, *not* returning it here
/// would silently lose the region.
///
/// Implements [`std::error::Error`] (delegating [`Display`](fmt::Display) to
/// [`source`](Self::source)), so `?` works in functions that return it.
pub struct BStackRaiiAllocError {
    /// The underlying I/O error that caused the operation to fail.
    pub source: io::Error,
    /// The recovered region's range, if it survived the failure.
    ///
    /// * `Some` — the allocation is intact and owned by the caller again (the
    ///   overwhelmingly common case: an untouched original or a fully committed new
    ///   region).
    /// * `None` — the region was consumed or lost during the failed operation (a
    ///   multi-step path whose later step failed, or a crash mid-op); any bytes are
    ///   recoverable only through the file's crash-recovery / WAL. Treat `None` as
    ///   "not recoverable here," not as impossible.
    pub handle: Option<BStackRange>,
}

impl BStackRaiiAllocError {
    /// Construct an error that hands the still-valid range back to the caller.
    #[inline]
    pub fn with_handle(source: io::Error, handle: BStackRange) -> Self {
        Self {
            source,
            handle: Some(handle),
        }
    }

    /// Construct an error whose region was consumed or lost and cannot be returned.
    #[inline]
    pub fn lost(source: io::Error) -> Self {
        Self {
            source,
            handle: None,
        }
    }
}

impl fmt::Debug for BStackRaiiAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BStackRaiiAllocError")
            .field("source", &self.source)
            .field("handle", &self.handle)
            .finish()
    }
}

impl_source_error!(BStackRaiiAllocError);

/// An **object-safe, range-based** view of a live file's allocator — the
/// type-erased handle a `Foreign<T>` uses to reach *into another file*.
///
/// This mirrors bstack's `BStackAllocator` surface (`stack` / `alloc` / `realloc` /
/// `dealloc`, plus `len` / `is_empty`), but is deliberately object-safe so the
/// registry can store `Arc<dyn BStackRaiiHost>` for files backed by different
/// allocator types: it drops the GAT `Allocated<'a>` handle and the associated
/// `Error` in favour of a plain [`BStackRange`] and [`io::Error`] — the very things
/// that make [`BStackRaiiAllocator`] itself non-object-safe (see
/// [`SyncBStackRaiiAllocator`]). Blanket-implemented for every
/// [`SyncBStackRaiiAllocator`], forwarding to the real allocator.
///
/// Because a [`BStackRange`] carries no ownership (unlike a `BStackOwnedSlice`),
/// [`realloc`](Self::realloc) and [`dealloc`](Self::dealloc) are `unsafe`: the
/// caller asserts the range is a live allocation in this file that no other handle
/// will also resize or free. On the failure path they return a [`BStackRaiiAllocError`]
/// carrying the surviving range, so a failed op never silently leaks. Raw reads and
/// writes go through [`stack`](Self::stack) (`get_into` / `set`).
///
/// # Crash consistency
///
/// Every method forwards to a single underlying allocator/stack call, so it
/// inherits that call's crash-consistency class (see the concrete allocator's docs).
pub trait BStackRaiiHost: Send + Sync {
    /// A shared reference to this file's underlying [`BStack`], for raw reads and
    /// writes (`get_into` / `set`) at a resolved offset.
    fn stack(&self) -> &BStack;

    /// Allocate `len` zero-initialised bytes, returning the region's range. The
    /// region is durably synced before returning; `len = 0` is valid.
    fn alloc(&self, len: u64) -> io::Result<BStackRange>;

    /// Resize the region at `handle` to `new_len` bytes, returning the (possibly
    /// moved) new range.
    ///
    /// # Safety
    /// `handle` must be a live allocation in this file, solely owned by the caller.
    ///
    /// # Errors
    /// Returns a [`BStackRaiiAllocError`] on failure (including when the allocator
    /// does not support reallocation). A failed resize leaves the original region
    /// intact, so implementations return it in [`BStackRaiiAllocError::handle`]
    /// (`Some`) whenever it survives, reserving `None` for a genuinely lost region.
    unsafe fn realloc(
        &self,
        handle: BStackRange,
        new_len: u64,
    ) -> Result<BStackRange, BStackRaiiAllocError>;

    /// Release the region at `handle`.
    ///
    /// # Safety
    /// `handle` must be a live allocation in this file, solely owned by the caller
    /// and freed exactly once.
    ///
    /// # Errors
    /// Returns a [`BStackRaiiAllocError`] on failure. A failed free normally leaves
    /// the region still allocated, so implementations return it in
    /// [`BStackRaiiAllocError::handle`] (`Some`) whenever it survives, reserving
    /// `None` for a genuinely lost region (where handing it back would risk a
    /// double-free).
    unsafe fn dealloc(&self, handle: BStackRange) -> Result<(), BStackRaiiAllocError>;

    /// This file's WAL anchor slot, if it participates in crash reclamation
    /// ([`BStackRaiiAllocator::wal_anchor`]).
    fn wal_anchor(&self) -> Option<NonNullOffset>;
}

impl<A: SyncBStackRaiiAllocator> BStackRaiiHost for A {
    fn stack(&self) -> &BStack {
        <A as BStackAllocator>::stack(self)
    }

    fn alloc(&self, len: u64) -> io::Result<BStackRange> {
        Ok(<A as BStackAllocator>::alloc(self, len)?.as_range())
    }

    unsafe fn realloc(
        &self,
        handle: BStackRange,
        new_len: u64,
    ) -> Result<BStackRange, BStackRaiiAllocError> {
        // SAFETY: caller's contract — a live, solely-owned allocation in this file.
        let slice: BStackOwnedSlice<'_, A> =
            unsafe { BStackOwnedSlice::from_raw_range(self, handle) };
        match <A as BStackAllocator>::realloc(self, slice, new_len) {
            Ok(s) => Ok(s.as_range()),
            Err(e) => Err(BStackRaiiAllocError {
                source: e.source,
                handle: e.handle.map(|h| h.as_range()),
            }),
        }
    }

    unsafe fn dealloc(&self, handle: BStackRange) -> Result<(), BStackRaiiAllocError> {
        // SAFETY: caller's contract — a live, solely-owned allocation in this file.
        let slice: BStackOwnedSlice<'_, A> =
            unsafe { BStackOwnedSlice::from_raw_range(self, handle) };
        match <A as BStackAllocator>::dealloc(self, slice) {
            Ok(()) => Ok(()),
            Err(e) => Err(BStackRaiiAllocError {
                source: e.source,
                handle: e.handle.map(|h| h.as_range()),
            }),
        }
    }

    fn wal_anchor(&self) -> Option<NonNullOffset> {
        <A as BStackRaiiAllocator>::wal_anchor(self)
    }
}
