//! Typed ↔ untyped handle conversion — the runtime behind `bstack_cast!`.
//!
//! Upcasts (typed → untyped) are infallible. Downcasts (untyped → typed) check
//! the block's [`EightCC`] tag against the target type's and are fallible. The
//! borrowed upcast (`X` → [`BStackSlice`]) is a generated `X::as_slice(stack)`
//! method, since a bare handle carries no stack.

use std::io;

use crate::BStackRaiiAllocator;
use bstack::{BStackOwnedSlice, BStackSlice};

use super::super::compiled::owned::BStackOwned;
use super::block::BStackBlock;
use crate::handback::CastError;
use crate::primitives::EightCC;

/// Byte offset of the `tag` within a [`crate::BlockHeader`] (`size: u64` first).
const TAG_OFFSET: u64 = 8;

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
