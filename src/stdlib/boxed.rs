//! [`BStackBox<T>`]: an owned, single-value block for a plain [`Pod`] `T`.
//!
//! The on-disk analogue of [`std::boxed::Box`] — but, unlike `Box`, it is only
//! useful for **`Pod`** payloads. A `#[bstack_block]` type is *already* an owned
//! block: you hold it as a [`BStackOwned`], embed it, reference it, put it in a
//! [`crate::BStackCow`]. There is nothing left for a `Box` to add. What has *no*
//! owned form is a bare scalar or plain `#[repr(C)]` struct: you cannot own a
//! lone `u64` on disk without first wrapping it in a block, which today means
//! hand-writing a one-field `#[bstack_block]`. `BStackBox<T>` fills exactly that
//! gap — a generic, macro-free, childless block whose whole payload is one `T`.
//!
//! Because the payload is `Pod` the block has no children, so the deep-clone and
//! teardown reduce to a byte copy / a single free — the childless defaults on
//! [`BStackBlock`] already do the right thing. `BStackBox<T>` is a first-class
//! block: it implements [`BStackBlock`], [`TryCloneIn`], [`BStackDrop`], and
//! [`BStackMove`], so it composes as a `#[bstack_owned]` / `#[bstack_ref]` field
//! and drops into a [`crate::BStackCow`] like any generated block.

use core::marker::PhantomData;
use core::mem::size_of;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::{BStack, BStackRange};
use bytemuck::{Pod, Zeroable};

use crate::io_core::{TryCloneIn, dealloc_range};
use crate::primitives::EightCC;
use crate::types::compiled::{BStackOwned, BlockHeader, HEADER_SIZE};
use crate::types::traits::{BStackBlock, BStackCast, BStackMove, BStackRef};

/// The on-disk image of a [`BStackBox<T>`]: the standard [`BlockHeader`] followed
/// by the boxed value. `#[repr(C, packed)]` (like every generated `XOnDisk`) so
/// there is no padding between the header and `value` — a requirement for the
/// hand-written [`Pod`] impl and for reading the whole image back with
/// `bytemuck`.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BoxOnDisk<T: Pod> {
    /// The 16-byte block header (size + type tag).
    pub header: BlockHeader,
    /// The boxed value.
    pub value: T,
}

// SAFETY: `BlockHeader` is `Pod` and `T: Pod`; `#[repr(C, packed)]` removes all
// inter-field padding, so every byte of `BoxOnDisk<T>` is initialized and every
// bit pattern is valid. `T: Pod` also carries `Copy + 'static`.
unsafe impl<T: Pod> Zeroable for BoxOnDisk<T> {}
unsafe impl<T: Pod> Pod for BoxOnDisk<T> {}

/// An owned, single-value block wrapping a plain [`Pod`] `T`.
///
/// A typed handle (a newtype over a [`BStackRange`], like every generated block
/// handle), so it is `Copy` and carries no allocator. Ownership is expressed the
/// usual way: [`new`](Self::new) hands back a bare [`BStackOwned<BStackBox<T>>`]
/// that frees nothing on scope exit — free it with
/// `bstack_drop` or wrap it in an
/// [`crate::AutoDrop`]/[`crate::BStackCow`].
pub struct BStackBox<T: Pod> {
    range: BStackRange,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Pod> BStackBox<T> {
    /// The on-disk size of a boxed `T` (header + value).
    const SIZE: u64 = size_of::<BoxOnDisk<T>>() as u64;

    /// Allocate a fresh block holding `value` and return an owning handle.
    ///
    /// The header and payload are written as a single image, so the block is
    /// created with one write (and released without leaking on write failure).
    pub fn new<A: BStackRaiiAllocator>(allocator: &A, value: T) -> io::Result<BStackOwned<Self>> {
        let od = BoxOnDisk {
            header: BlockHeader {
                size: Self::SIZE,
                tag: Self::eightcc(),
            },
            value,
        };
        let mut slice = allocator.alloc(Self::SIZE)?;
        if let Err(e) = slice.write_range(0, bytemuck::bytes_of(&od)) {
            let _ = allocator.dealloc(slice);
            return Err(e);
        }
        // SAFETY: a freshly allocated block that no other handle owns — exactly
        // the sole-ownership invariant `BStackOwned::from_raw` requires.
        Ok(unsafe { BStackOwned::from_raw(Self::from_range(slice.as_range())) })
    }

    /// Read the boxed value out of the block.
    pub fn get(&self, stack: &BStack) -> io::Result<T> {
        let mut buf = std::vec![0u8; size_of::<BoxOnDisk<T>>()];
        // SAFETY: `self.range` is a live `BStackBox<T>` block.
        let r = unsafe { BStackRef::<Self>::from_range(self.range) };
        r.read_on_disk(stack, &mut buf)?;
        // Copy the value out of the packed image without forming a reference to
        // the (alignment-1) `value` field.
        let off = HEADER_SIZE as usize;
        Ok(bytemuck::pod_read_unaligned::<T>(
            &buf[off..off + size_of::<T>()],
        ))
    }

    /// Overwrite the boxed value in place.
    pub fn set<A: BStackRaiiAllocator>(&self, allocator: &A, value: T) -> io::Result<()> {
        allocator
            .stack()
            .set(self.range.start() + HEADER_SIZE, bytemuck::bytes_of(&value))
    }
}

impl<T: Pod> BStackCast for BStackBox<T> {
    /// A `"Box"` prefix over hash bytes perturbed by `size_of::<T>()`, so boxes of
    /// differently-sized payloads never share a tag (matching the generic
    /// `#[bstack_block]` POD tag scheme, which also distinguishes by size).
    fn eightcc() -> EightCC {
        const BASE: EightCC = EightCC::new([b'B', b'o', b'x', 0x80, 0x81, 0x82, 0x83, 0x84]);
        BASE.mix(EightCC::new((size_of::<T>() as u64).to_le_bytes()))
    }
}

// Self-contained (no separate control block): may be `#[embed]`ded.
impl<T: Pod> crate::types::traits::BStackEmbeddable for BStackBox<T> {}

impl<T: Pod> BStackBlock for BStackBox<T> {
    type OnDisk = BoxOnDisk<T>;

    unsafe fn from_range(range: BStackRange) -> Self {
        BStackBox {
            range,
            _marker: PhantomData,
        }
    }

    fn range(&self) -> BStackRange {
        self.range
    }
    // A `Pod` box is childless: the `__bstack_drop_children` /
    // `__bstack_clone_*` defaults (free nothing / byte-copy the OnDisk) are
    // exactly correct, so they are deliberately not overridden.
}

impl<T: Pod> TryCloneIn for BStackBox<T> {}

impl<T: Pod> BStackMove for BStackBox<T> {
    /// Moving a box out yields the plain value.
    type Fields<'a, A: BStackRaiiAllocator> = T;

    fn bstack_move<A: BStackRaiiAllocator>(
        owned: BStackOwned<Self>,
        allocator: &A,
    ) -> io::Result<T> {
        let me = owned.into_inner();
        // Read the value out before freeing the shell. `into_inner` already defused the
        // owning wrapper, so on a read fault the shell would leak — free it on the error
        // path too. The box was consumed by value and its `Pod` payload has no child to
        // hand back, so freeing-then-erroring is the correct contract.
        let value = match me.get(allocator.stack()) {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: `me` was the sole owner (from a `BStackOwned`); freed once.
                let _ = unsafe { dealloc_range(allocator, me.range) };
                return Err(e);
            }
        };
        // Childless: free the shell after reading the value out.
        // SAFETY: `me` was the sole owner (it came from a `BStackOwned`).
        unsafe { dealloc_range(allocator, me.range)? };
        Ok(value)
    }
}
