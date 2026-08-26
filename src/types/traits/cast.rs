//! Typed ↔ untyped handle conversion — the runtime behind `bstack_cast!`.
//!
//! Upcasts (typed → untyped) are infallible. Downcasts (untyped → typed) check
//! the block's [`EightCC`] tag against the target type's and are fallible. The
//! borrowed upcast (`X` → [`BStackSlice`]) is a generated `X::as_slice(stack)`
//! method, since a bare handle carries no stack.

use std::error::Error;
use std::fmt;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::{BStackOwnedSlice, BStackSlice};

use super::block::BStackBlock;
use crate::primitives::EightCC;
use super::super::compiled::owned::BStackOwned;
use super::drop::AutoDrop;

/// Byte offset of the `tag` within a [`crate::BlockHeader`] (`size: u64` first).
const TAG_OFFSET: u64 = 8;

impl<'a, T: BStackBlock, A: BStackRaiiAllocator> AutoDrop<'a, BStackOwned<T>, A> {
    /// Upcast an auto-freeing owned handle to the untyped owned slice, discarding
    /// type info (infallible).
    ///
    /// Consumes the guard without running its disk-level `Drop`; the returned
    /// slice owns the allocation. (A bare `BStackOwned<X>` carries no allocator,
    /// so wrap it — `owned.auto(alloc)` — before upcasting to a slice.)
    pub fn into_slice(self) -> BStackOwnedSlice<'a, A> {
        let (owned, allocator) = self.into_raw_parts();
        let range = owned.into_inner().range();
        unsafe { BStackOwnedSlice::from_raw_range(allocator, range) }
    }
}

/// The error a fallible downcast ([`BStackCastInto::cast_into`]) returns. It always
/// hands the input **slice back** so an ownership-carrying [`BStackOwnedSlice`] is
/// never dropped (and thus leaked) on a failed cast — the same hand-back contract
/// as the crate's other consuming operations
/// ([`ReplaceError`](crate::ReplaceError) / [`ConstructError`](crate::ConstructError)).
///
/// It carries no `From<CastError> for io::Error` on purpose: a caller cannot `?` a
/// failed cast and silently drop the slice it was handed back — it must recover the
/// slice (try another type, or free it) via [`into_slice`](Self::into_slice).
pub enum CastError<S> {
    /// The block's tag or on-disk size is not `T`'s — not an I/O failure, the block
    /// simply is not a `T`. The slice is handed back unchanged.
    Mismatch(S),
    /// Reading the block header faulted. The slice is intact and handed back with
    /// the underlying error.
    Io(io::Error, S),
}

impl<S> CastError<S> {
    /// Recover the handed-back slice, discarding *why* the cast failed. The slice
    /// still owns its block — try another type, free it, or re-wrap it.
    #[inline]
    pub fn into_slice(self) -> S {
        match self {
            CastError::Mismatch(s) | CastError::Io(_, s) => s,
        }
    }

    /// The underlying I/O error, or `None` for a clean tag/size mismatch (which is
    /// not an error condition).
    #[inline]
    pub fn io(&self) -> Option<&io::Error> {
        match self {
            CastError::Io(e, _) => Some(e),
            CastError::Mismatch(_) => None,
        }
    }
}

// Manual, so `S` (a slice handle) need not be `Debug`.
impl<S> fmt::Debug for CastError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CastError::Mismatch(_) => f.write_str("CastError::Mismatch(..)"),
            CastError::Io(e, _) => f.debug_tuple("CastError::Io").field(e).finish(),
        }
    }
}

impl<S> fmt::Display for CastError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CastError::Mismatch(_) => {
                f.write_str("bstack_cast!: block tag/size is not the target type")
            }
            CastError::Io(e, _) => fmt::Display::fmt(e, f),
        }
    }
}

impl<S> Error for CastError<S> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CastError::Io(e, _) => Some(e),
            CastError::Mismatch(_) => None,
        }
    }
}

/// Downcast an owned slice to a typed (bare) owned handle by checking the block
/// tag. The result carries no allocator; free it with `owned.bstack_drop(alloc)`
/// or wrap it via `owned.auto(alloc)`.
pub trait BStackCastInto<'a, A: BStackRaiiAllocator>: Sized {
    /// `Ok(owned)` on a tag/size match; otherwise a [`CastError`] that hands the
    /// slice back — [`Mismatch`](CastError::Mismatch) when the block is not a `T`,
    /// [`Io`](CastError::Io) when reading the header faulted.
    fn cast_into<T: BStackBlock>(self) -> Result<BStackOwned<T>, CastError<Self>>;
}

impl<'a, A: BStackRaiiAllocator> BStackCastInto<'a, A> for BStackOwnedSlice<'a, A> {
    fn cast_into<T: BStackBlock>(self) -> Result<BStackOwned<T>, CastError<Self>> {
        // Second gate after the tag: the allocator-attested slice length must be
        // exactly the target's on-disk size. Rejects a same-tag instantiation of a
        // different size (a lossy generic-tag collision) with no extra I/O.
        if self.len() != core::mem::size_of::<T::OnDisk>() as u64 {
            return Err(CastError::Mismatch(self));
        }
        let mut tag = [0u8; 8];
        if let Err(e) = self.read_range_into(TAG_OFFSET, &mut tag) {
            return Err(CastError::Io(e, self));
        }
        if EightCC(tag) != T::eightcc() {
            return Err(CastError::Mismatch(self));
        }
        // `as_range` consumes the owned slice, defusing its own free; ownership
        // of the block transfers to the returned bare handle.
        let range = self.as_range();
        Ok(unsafe { BStackOwned::from_raw(T::from_range(range)) })
    }
}

/// Downcast a borrowed slice to a typed handle by checking the block tag.
pub trait BStackCastAs<'a> {
    /// `Some(handle)` on a tag match, `None` on mismatch, `Err` on an I/O
    /// failure reading the header.
    fn cast_as<T: BStackBlock>(&self) -> io::Result<Option<T>>;
}

impl<'a> BStackCastAs<'a> for BStackSlice<'a> {
    fn cast_as<T: BStackBlock>(&self) -> io::Result<Option<T>> {
        // Same size gate as `cast_into`: same tag but a different on-disk size is
        // a generic-tag collision, not a match.
        if self.len() != core::mem::size_of::<T::OnDisk>() as u64 {
            return Ok(None);
        }
        let mut tag = [0u8; 8];
        self.read_range_into(TAG_OFFSET, &mut tag)?;
        if EightCC(tag) != T::eightcc() {
            return Ok(None);
        }
        Ok(Some(unsafe { T::from_range(self.as_range()) }))
    }
}
