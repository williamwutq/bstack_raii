//! The on-disk [`BlockHeader`] prefixing every block, and [`HEADER_SIZE`].
//!
//! `BlockHeader` is [`bytemuck::Pod`] so it embeds directly in a generated
//! `XOnDisk` struct and reads back with `bytemuck::from_bytes`. This is the
//! compiled on-disk *shape* of a block's header — distinct from the
//! [`BStackBlock`](super::super::traits::BStackBlock) trait that describes a
//! block's behaviour.

use bytemuck::{Pod, Zeroable};

use crate::primitives::EightCC;

/// Byte length of a [`BlockHeader`] — the offset at which a block's payload
/// begins.
pub const HEADER_SIZE: u64 = core::mem::size_of::<BlockHeader>() as u64;

/// The header prefixing every on-disk block. 16 bytes.
///
/// `size` is the payload length in bytes; `tag` is the [`EightCC`] discriminant
/// written by the allocator at block creation. Declared `#[repr(C)]` rather than
/// `#[repr(C, packed)]`: a `u64` followed by an 8-byte tag is already densely
/// packed with no padding, and avoiding `packed` keeps field access sound. The
/// *generated* `XOnDisk` structs that embed this and then mix in smaller POD
/// fields are the ones that need `packed`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BlockHeader {
    pub size: u64,
    pub tag: EightCC,
}
