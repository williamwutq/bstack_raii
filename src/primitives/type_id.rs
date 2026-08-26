//! [`TypeId`] — the *which type* component of the wide pointer, and its typed
//! refinement [`ResolvedTypeId`].

use core::num::NonZeroU32;
use std::error::Error;
use std::fmt;

use bytemuck::{Pod, Zeroable};

/// The RTTI type identity a wide pointer carries — the pointee's
/// [`RttiOrdinal`](crate::rtti::RttiOrdinal), or *untyped*.
///
/// Stored as **`ordinal + 1`** so the all-zero word is the **untyped** niche: a
/// pointer minted by the static `#[bstack_block]` path (where the schema already
/// knows the target type) leaves it `0`, and an old two-word `{file_id: u64,
/// offset: u64}` pointer — whose zeroed high file-id half now overlaps this field —
/// reads back as untyped. Only RTTI-aware writers ([`typed_ptr`](crate::rtti::typed_ptr))
/// set it. Carried through a read → re-store round-trip so a typed pointer stays
/// typed (rebuilding the wire form from raw `(file_id, offset)` would zero it).
///
/// `#[repr(transparent)]` over the `u32` the wire format stores, so it composes into
/// the fat-pointer record with no encoding change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct TypeId(u32);

impl TypeId {
    /// The untyped pointer: no RTTI type recorded (the target type is recovered from
    /// the block header, or already known to the schema). The all-zero word.
    pub const UNTYPED: TypeId = TypeId(0);

    /// A typed id for RTTI `ordinal` (stored as `ordinal + 1`). Mirrors
    /// [`typed_ptr`](crate::rtti::typed_ptr)'s encoding; `ordinal` comes from RTTI
    /// registration and is far below `u32::MAX`.
    ///
    /// # Panics
    /// If `ordinal == u32::MAX`, so `ordinal + 1` overflows. Unreachable for a real
    /// RTTI ordinal.
    pub const fn from_ordinal(ordinal: u32) -> Self {
        TypeId(ordinal + 1)
    }

    /// The pointee's RTTI ordinal, or `None` if this is [`UNTYPED`](Self::UNTYPED).
    pub const fn ordinal(self) -> Option<u32> {
        match self.0 {
            0 => None,
            n => Some(n - 1),
        }
    }

    /// Whether a type is recorded (i.e. not [`UNTYPED`](Self::UNTYPED)).
    pub const fn is_typed(self) -> bool {
        self.0 != 0
    }

    /// The raw wire word (`0` = untyped, else `ordinal + 1`).
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Wrap a raw wire word (`0` = untyped, else `ordinal + 1`) unchecked — for
    /// decoding an existing on-disk pointer, whose word is already in this encoding.
    pub const fn from_raw(v: u32) -> Self {
        TypeId(v)
    }

    /// This id refined to a **typed** id (carries an ordinal), or `None` if it is
    /// [`UNTYPED`](Self::UNTYPED) — the [`NonZeroU32::new`](core::num::NonZeroU32::new)
    /// analogue (`TypeId` is the base type). The refinement carries "a type is
    /// recorded" in the type, so [`ResolvedTypeId::ordinal`] is infallible.
    pub const fn resolve(self) -> Option<ResolvedTypeId> {
        ResolvedTypeId::new(self)
    }
}

/// A [`TypeId`] **statically guaranteed to be typed** — it records a real RTTI
/// [`ordinal`](Self::ordinal), never [`UNTYPED`](TypeId::UNTYPED).
///
/// The refined half of the `TypeId` ↔ `ResolvedTypeId` pair, mirroring [`NonZeroU32`]
/// over `u32`: constructed once where a pointer is known typed, after which
/// [`ordinal`](Self::ordinal) needs no `Option`.
///
/// Backed by [`NonZeroU32`], so `Option<ResolvedTypeId>` is the same 4 bytes as a
/// `TypeId` (the `0`/untyped bit-pattern is the `None` niche). **Not**
/// [`Pod`](bytemuck::Pod): `0` is an invalid bit pattern — the wire form is always the
/// plain [`TypeId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ResolvedTypeId(NonZeroU32);

impl ResolvedTypeId {
    /// Refine a [`TypeId`] to typed, or `None` if it is [`UNTYPED`](TypeId::UNTYPED) —
    /// the [`NonZeroU32::new`] analogue.
    pub const fn new(id: TypeId) -> Option<Self> {
        match NonZeroU32::new(id.0) {
            Some(nz) => Some(ResolvedTypeId(nz)),
            None => None,
        }
    }

    /// A typed id for RTTI `ordinal` directly (stored as `ordinal + 1`) — the
    /// always-typed counterpart of [`TypeId::from_ordinal`].
    ///
    /// # Panics
    /// If `ordinal + 1` overflows to zero (i.e. `ordinal == u32::MAX`). Unreachable
    /// for a real RTTI ordinal.
    pub const fn from_ordinal(ordinal: u32) -> Self {
        // `ordinal + 1` is non-zero for every realistic ordinal (far below `u32::MAX`).
        match NonZeroU32::new(ordinal + 1) {
            Some(nz) => ResolvedTypeId(nz),
            None => panic!("RTTI ordinal + 1 overflowed to zero"),
        }
    }

    /// Refine a [`TypeId`] **without checking** — the [`NonZeroU32::new_unchecked`]
    /// analogue.
    ///
    /// # Safety
    /// `id` must be typed (not [`UNTYPED`](TypeId::UNTYPED)). An untyped id here is
    /// instant undefined behaviour, violating the backing [`NonZeroU32`]'s niche.
    pub const unsafe fn new_unchecked(id: TypeId) -> Self {
        // SAFETY: forwarded to the caller — `id` is typed (non-zero) by contract.
        ResolvedTypeId(unsafe { NonZeroU32::new_unchecked(id.0) })
    }

    /// Widen back to the base [`TypeId`] (never [`UNTYPED`](TypeId::UNTYPED)) — the
    /// [`NonZeroU32::get`] analogue.
    pub const fn get(self) -> TypeId {
        TypeId(self.0.get())
    }

    /// The raw non-zero wire word (`ordinal + 1`), skipping the [`TypeId`] wrapper.
    pub const fn as_u32(self) -> u32 {
        self.0.get()
    }

    /// The RTTI ordinal — **infallible** here, since a typed id always has one (the
    /// refinement's payoff over [`TypeId::ordinal`]'s `Option`).
    pub const fn ordinal(self) -> u32 {
        self.0.get() - 1
    }
}

impl From<ResolvedTypeId> for TypeId {
    /// Widen to the base type — the `From<NonZeroU32> for u32` analogue.
    #[inline]
    fn from(id: ResolvedTypeId) -> TypeId {
        id.get()
    }
}

impl TryFrom<TypeId> for ResolvedTypeId {
    type Error = UntypedTypeIdError;

    /// Refine to typed, failing on [`UNTYPED`](TypeId::UNTYPED) — the
    /// `TryFrom<u32> for NonZeroU32` analogue.
    #[inline]
    fn try_from(id: TypeId) -> Result<Self, UntypedTypeIdError> {
        ResolvedTypeId::new(id).ok_or(UntypedTypeIdError)
    }
}

/// The error [`ResolvedTypeId::try_from`] returns for an [`UNTYPED`](TypeId::UNTYPED)
/// id — the [`TryFromIntError`](core::num::TryFromIntError) analogue for this
/// refinement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UntypedTypeIdError;

impl fmt::Display for UntypedTypeIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("type id is untyped (records no RTTI ordinal)")
    }
}

impl Error for UntypedTypeIdError {}
