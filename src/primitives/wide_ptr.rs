//! [`WidePtr`] — the composed wide (fat) pointer: *which file*, *which type*,
//! *where in that file*.

use bytemuck::{Pod, Zeroable};

use super::{FileId, Offset, ResolvedFileId, ResolvedTypeId, TypeId};

/// The crate's **wide pointer** (a "fat pointer"): a persisted cross-file reference,
/// assembled from its three orthogonal [components](super) — a [`FileId`] (which
/// file), a [`TypeId`] (which type, or untyped), and an [`Offset`] (where in that
/// file).
///
/// This is the canonical on-disk representation: `#[repr(C)]` over the three
/// `#[repr(transparent)]` primitives, laying out as `{ file: u32 @0, ty: u32 @4,
/// offset: u64 @8 }` — 16 bytes, [`Pod`], byte-for-byte the raw
/// `{ file_id, type_index, offset }` triple it composes. The all-zero value is a
/// valid pointer ([`SELF`](FileId::SELF), untyped, null), so it is [`Zeroable`].
///
/// It is the inert wire/value form only: resolution (registry lookup, deref) and the
/// borrow brand live on the in-memory `Foreign<T>` wrapper that carries a `WidePtr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct WidePtr {
    file: FileId,
    ty: TypeId,
    offset: Offset,
}

// The wire contract: 16 bytes, `{ file_id: u32 @0, type_index: u32 @4, offset: u64 @8 }`,
// byte-for-byte the raw triple this replaces. Enforced at compile time so a layout
// drift (a reordered field, a padded component) is a build error, not a silent
// on-disk-format break.
const _: () = {
    assert!(core::mem::size_of::<WidePtr>() == 16);
    assert!(core::mem::align_of::<WidePtr>() == 8);
    assert!(core::mem::offset_of!(WidePtr, file) == 0);
    assert!(core::mem::offset_of!(WidePtr, ty) == 4);
    assert!(core::mem::offset_of!(WidePtr, offset) == 8);
};

impl WidePtr {
    /// An **untyped** pointer to `(file, offset)` — its [`TypeId`] is
    /// [`UNTYPED`](TypeId::UNTYPED). Tag it with [`with_type`](Self::with_type) when
    /// an RTTI type is known.
    pub const fn new(file: FileId, offset: Offset) -> Self {
        WidePtr {
            file,
            ty: TypeId::UNTYPED,
            offset,
        }
    }

    /// Assemble from all three components explicitly.
    pub const fn with_parts(file: FileId, ty: TypeId, offset: Offset) -> Self {
        WidePtr { file, ty, offset }
    }

    /// This pointer tagged with RTTI type `ty` (chains off [`new`](Self::new)); pass
    /// [`TypeId::UNTYPED`] to clear it.
    pub const fn with_type(mut self, ty: TypeId) -> Self {
        self.ty = ty;
        self
    }

    /// The target file.
    pub const fn file(self) -> FileId {
        self.file
    }

    /// The RTTI type tag (possibly [`UNTYPED`](TypeId::UNTYPED)).
    pub const fn type_id(self) -> TypeId {
        self.ty
    }

    /// The target address within the file.
    pub const fn offset(self) -> Offset {
        self.offset
    }

    /// Whether the target file is [`SELF`](FileId::SELF) (the current file).
    pub const fn is_self(self) -> bool {
        self.file.is_self()
    }

    /// Whether the pointer is null — an absent target (`offset == 0`).
    pub const fn is_null(self) -> bool {
        self.offset.is_null()
    }

    /// The target file refined to an ordinary registered file
    /// ([`ResolvedFileId`]), or `None` for [`SELF`](FileId::SELF) / a special id.
    pub const fn resolved_file(self) -> Option<ResolvedFileId> {
        self.file.resolve()
    }

    /// The type tag refined to a typed id ([`ResolvedTypeId`]), or `None` if the
    /// pointer is untyped.
    pub const fn resolved_type(self) -> Option<ResolvedTypeId> {
        self.ty.resolve()
    }
}
