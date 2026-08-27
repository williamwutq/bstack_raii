//! [`Offset`] — the *where in the file* component of the wide pointer, and its
//! guaranteed-non-null refinement [`NonNullOffset`].

use core::num::NonZeroU64;
use std::error::Error;
use std::fmt;
use std::io;

use bstack::{BStackRange, BStackSlice};
use bytemuck::{Pod, Zeroable};

use crate::util::io_errorfn;

/// A byte address into a bstack file's stack — the target half of a wide pointer,
/// and the currency of every on-disk read/write.
///
/// **`0` is the null niche**: an absent target (`Option<Foreign<T>>` stores it
/// inline as a zero offset), so an `Offset` is *nullable*, not a `NonZero`. Use
/// [`is_null`](Self::is_null) to test it.
///
/// An offset routinely originates from **untrusted on-disk bytes** — a forged or
/// corrupted `ctrl` back-pointer, `Foreign` target, or linked-structure field — and
/// interpreter walks chain arithmetic off it (`base + field_offset`,
/// `base + index * stride`). A plain `+`/`*` would panic under `overflow-checks` or
/// silently wrap to an unrelated in-bounds address a later read/write would then
/// corrupt, so the arithmetic here is **overflow-checked** and yields `InvalidData`
/// rather than wrapping. (This is the single home for `add_off` / `mul_off`
/// [`crate::rtti`] and `checked_off` [`crate::layout`].)
///
/// `#[repr(transparent)]` over `u64`, so it composes into the fat-pointer record with
/// no encoding change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable)]
#[repr(transparent)]
pub struct Offset(u64);

impl Offset {
    /// The null offset (`0`) — an absent target.
    pub const NULL: Offset = Offset(0);

    /// Whether this is the [`NULL`](Self::NULL) offset (an absent target).
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// The raw byte address.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Wrap a raw byte address.
    pub const fn from_raw(v: u64) -> Self {
        Offset(v)
    }

    /// This address advanced by `delta` bytes (a field offset, header size, or
    /// element stride), rejecting overflow — the base may be forged/fuzzed, so a
    /// wrap must fail cleanly rather than land on an unrelated in-bounds address.
    pub fn checked_add(self, delta: u64) -> io::Result<Offset> {
        self.0
            .checked_add(delta)
            .map(Offset)
            .ok_or_else(offset_overflow)
    }

    /// This address advanced by `index * stride` bytes (an `Array` / `Vec` element
    /// address), rejecting overflow in either the multiply or the add.
    pub fn checked_add_mul(self, index: u64, stride: u64) -> io::Result<Offset> {
        let delta = index.checked_mul(stride).ok_or_else(offset_overflow)?;
        self.checked_add(delta)
    }

    /// This offset as a [`NonNullOffset`], or `None` if it is [`NULL`](Self::NULL) —
    /// the [`NonZeroU64::new`] analogue. The type-level "I have checked this points at
    /// a real target" refinement; the null check happens once, here.
    pub const fn to_non_null(self) -> Option<NonNullOffset> {
        NonNullOffset::new(self)
    }
}

/// An [`Offset`] **statically guaranteed non-null** — a resolved, real target
/// address (a control-block back-pointer, a live child, a refcount counter). Carries
/// the not-null invariant in the type, so a consumer that needs a genuine target
/// takes a `NonNullOffset` and the null check happens once, at the boundary, via
/// [`Offset::to_non_null`] — not re-checked at every deref.
///
/// Backed by [`NonZeroU64`], so `Option<NonNullOffset>` is the same 8 bytes as an
/// `Offset` (the `0` bit-pattern is the `None` niche). **Not** [`Pod`](bytemuck::Pod):
/// `0` is an invalid bit pattern, so it is an in-memory refinement, never a wire type —
/// the on-disk form is always the nullable [`Offset`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NonNullOffset(NonZeroU64);

impl NonNullOffset {
    /// Refine an [`Offset`] to non-null, or `None` if it is [`NULL`](Offset::NULL) —
    /// the [`NonZeroU64::new`] analogue (`Offset` is the base type, as `u64` is
    /// `NonZeroU64`'s).
    pub const fn new(offset: Offset) -> Option<Self> {
        match NonZeroU64::new(offset.0) {
            Some(nz) => Some(NonNullOffset(nz)),
            None => None,
        }
    }

    /// Refine an [`Offset`] to non-null **without checking** — the
    /// [`NonZeroU64::new_unchecked`] analogue.
    ///
    /// # Safety
    /// `offset` must not be [`NULL`](Offset::NULL). A null offset here is instant
    /// undefined behaviour: `NonNullOffset` is `NonZeroU64`-backed, and a zero
    /// `NonZeroU64` violates its niche.
    pub const unsafe fn new_unchecked(offset: Offset) -> Self {
        // SAFETY: forwarded to the caller — `offset` is non-null by contract.
        NonNullOffset(unsafe { NonZeroU64::new_unchecked(offset.0) })
    }

    /// Widen back to the base [`Offset`] (never [`NULL`](Offset::NULL)) — the
    /// [`NonZeroU64::get`] analogue.
    pub const fn get(self) -> Offset {
        Offset(self.0.get())
    }

    /// The raw non-zero byte address, skipping the [`Offset`] wrapper.
    pub const fn as_u64(self) -> u64 {
        self.0.get()
    }

    /// This address advanced by `delta` bytes, rejecting overflow. The result stays
    /// non-null (a non-zero base plus a non-negative delta cannot wrap to `0` without
    /// overflowing, which is rejected).
    pub fn checked_add(self, delta: u64) -> io::Result<NonNullOffset> {
        self.0
            .checked_add(delta)
            .map(NonNullOffset)
            .ok_or_else(offset_overflow)
    }

    /// The range's start as a non-null offset, or `None` at offset `0`.
    ///
    /// The direct form of [`Offset::from`]`(range)`[`.to_non_null()`](Offset::to_non_null).
    /// A `From<BStackRange> for Option<NonNullOffset>` impl is disallowed by the
    /// orphan rule (`Option` is not a local type), so this is a named constructor.
    #[inline]
    pub fn from_range(range: BStackRange) -> Option<NonNullOffset> {
        Self::new(range.into())
    }

    /// The slice's start as a non-null offset, or `None` at offset `0`.
    ///
    /// The direct form of [`Offset::from`]`(slice)`[`.to_non_null()`](Offset::to_non_null);
    /// a named constructor for the same orphan-rule reason as [`from_range`](Self::from_range).
    #[inline]
    pub fn from_slice(slice: BStackSlice<'_>) -> Option<NonNullOffset> {
        Self::new(slice.into())
    }
}

impl From<NonNullOffset> for Offset {
    /// Widen to the nullable base type — the `From<NonZeroU64> for u64` analogue.
    #[inline]
    fn from(o: NonNullOffset) -> Offset {
        o.get()
    }
}

impl From<NonNullOffset> for u64 {
    #[inline]
    fn from(o: NonNullOffset) -> u64 {
        o.as_u64()
    }
}

impl TryFrom<Offset> for NonNullOffset {
    type Error = NullOffsetError;

    /// Refine to non-null, failing on [`NULL`](Offset::NULL) — the
    /// `TryFrom<u64> for NonZeroU64` analogue.
    #[inline]
    fn try_from(offset: Offset) -> Result<Self, NullOffsetError> {
        NonNullOffset::new(offset).ok_or(NullOffsetError)
    }
}

/// The error [`NonNullOffset::try_from`] returns for a [`NULL`](Offset::NULL) offset —
/// the [`TryFromIntError`](core::num::TryFromIntError) analogue for this niche.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NullOffsetError;

impl fmt::Display for NullOffsetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("offset is null")
    }
}

impl Error for NullOffsetError {}

impl From<BStackRange> for Offset {
    /// The range's start address.
    #[inline]
    fn from(range: BStackRange) -> Offset {
        Offset::from_raw(range.start())
    }
}

impl From<BStackSlice<'_>> for Offset {
    /// The slice's start address.
    #[inline]
    fn from(slice: BStackSlice<'_>) -> Offset {
        Offset::from_raw(slice.start())
    }
}

io_errorfn!(offset_overflow, InvalidData, "on-disk offset arithmetic overflow");
