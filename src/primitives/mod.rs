//! The orthogonal **components of the crate's wide pointer**.
//!
//! A persisted cross-file reference is a fat pointer — on disk the 16-byte
//! [`WidePtr`] `{ file_id: u32, type_index: u32, offset: u64 }`.
//! Those three words are not one indivisible blob: each is an independent value with
//! its own **niche** and its own rules, and only the file-id half was ever modelled
//! as a real type ([`FileId`]). The rest travelled as bare `u32` / `u64`, so their
//! encoding (`type_index = ordinal + 1`, `offset == 0` = null) and their invariants
//! (overflow-checked address arithmetic) were re-implemented at every use site.
//!
//! This module gives each component its own newtype so the wide pointer becomes a
//! **composition of typed primitives** rather than a tuple of raw integers:
//!
//! * [`FileId`] — *which file* (`0` = [`SELF`](FileId::SELF), reserved descending
//!   "special" region at the top). Re-exported publicly as `registry::FileId`. Its
//!   ordinary-registered-file refinement is [`ResolvedFileId`].
//! * [`TypeId`] — *which type* (RTTI ordinal, stored as `ordinal + 1`; `0` = untyped).
//!   Its typed refinement is [`ResolvedTypeId`].
//! * [`Offset`] — *where in that file* (a byte address; `0` = null), with the
//!   overflow-checked arithmetic every on-disk-derived offset needs. Its
//!   guaranteed-non-null refinement is [`NonNullOffset`].
//!
//! Each component also has a **refinement** — a niche-backed sibling that carries "the
//! interesting bits are present" in the type ([`ResolvedFileId`], [`ResolvedTypeId`],
//! [`NonNullOffset`]), so the boundary check happens once and downstream code drops
//! the `Option`. Each mirrors [`NonZeroU32`](core::num::NonZeroU32) /
//! [`NonZeroU64`](core::num::NonZeroU64)'s relationship to its base integer
//! (`new` / `new_unchecked` / `get` / `From` / `TryFrom`), with the base component
//! standing in for the primitive integer.
//!
//! Each is `#[repr(transparent)]` and [`Pod`](bytemuck::Pod), so the wide pointer
//! [`WidePtr`] is built by *composing* them into one `#[repr(C)]` record with no
//! wire-format change — 16 bytes, byte-for-byte the raw `{ file_id, type_index,
//! offset }` triple it replaces.
// TypeId/Offset/NonNullOffset land before the fat pointer that composes them.
#![allow(dead_code, unused_imports)]

mod eightcc;
mod file_id;
mod offset;
mod ownership;
mod try_clone;
mod type_id;
mod wide_ptr;

pub use eightcc::EightCC;
pub use file_id::{FileId, ResolvedFileId, UnresolvedFileIdError};
pub use offset::{NonNullOffset, NullOffsetError, Offset, checked_off};
pub use ownership::{InvalidOwnershipKind, OwnershipKind};
pub use try_clone::TryClone;
pub use type_id::{ResolvedTypeId, TypeId, UntypedTypeIdError};
pub use wide_ptr::{BrandedWidePtr, WidePtr};
