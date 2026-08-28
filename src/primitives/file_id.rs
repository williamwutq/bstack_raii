//! [`FileId`] — the *which file* component of the wide pointer, and its
//! ordinary-registered-file refinement [`ResolvedFileId`].

use core::num::NonZeroU32;
use std::error::Error;
use std::fmt;

use bytemuck::{Pod, Zeroable};

/// A small, stable identity for a registered bstack file.
///
/// Backed by a `u32` (a sane program opens far fewer than `u16::MAX` files, so
/// this is generous headroom), but a `Foreign` pointer stores it **widened to a
/// `u64`** — for alignment next to a [`BStackRange`](bstack::BStackRange), and to
/// leave room for future RTTI. [`as_u64`](Self::as_u64) / [`from_u64`](Self::from_u64)
/// bridge the two.
///
/// # Id-space layout
///
/// * **`0` = [`SELF`](Self::SELF)** — the *current* file. A `Foreign` with this id
///   points into whatever file it itself lives in, resolved directly against the
///   local allocator the caller already holds. Registry lookup (and its lock) is
///   never consulted for `SELF`. Never assigned to a registered path.
/// * **`1, 2, 3, …` (ascending) = ordinary registered files** — assigned in order
///   of registration; the id is `1 + ` the file's index in the append-only path
///   table.
/// * **`u32::MAX, u32::MAX - 1, …` (descending) = reserved "special" meanings** —
///   sentinels beyond a single concrete file, allocated from the top down so they
///   never collide with the ascending ordinary ids (`SELF` is the sole exception
///   at the bottom). Only `SELF` is defined so far; the descending region is
///   reserved for future use (see [`is_special`](Self::is_special)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable)]
#[repr(transparent)]
pub struct FileId(u32);

impl FileId {
    /// The self-referential id (`0`): a `Foreign` bearing it points into the
    /// *current* file and is resolved against the local allocator **without**
    /// touching the registry or its lock. Never assigned to a registered path.
    pub const SELF: FileId = FileId(0);

    /// Lowest id treated as a reserved special sentinel (special ids grow *down*
    /// from `u32::MAX`). Chosen far above any plausible number of open files.
    pub const SPECIAL_FLOOR: u32 = u32::MAX - 0xFFFF;

    /// Whether this is [`SELF`](Self::SELF) (the current file).
    pub const fn is_self(self) -> bool {
        self.0 == 0
    }

    /// Whether this id is in the reserved descending "special" region (top of the
    /// `u32` space). Ordinary registered files and `SELF` are **not** special.
    /// The boundary is generous — far above any realistic file count.
    pub const fn is_special(self) -> bool {
        self.0 >= Self::SPECIAL_FLOOR
    }

    /// The raw `u32` value.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The id widened to the `u64` a `Foreign` pointer stores on disk.
    pub const fn as_u64(self) -> u64 {
        self.0 as u64
    }

    /// Wrap a raw `u32` as a `FileId`, unchecked. Crate-internal: the registry mints
    /// ids from path-table indices (`index + 1`) and `next`, which are `u32` by
    /// construction. Public reconstruction goes through [`from_u64`](Self::from_u64),
    /// which range-checks.
    pub(crate) const fn from_raw(v: u32) -> Self {
        FileId(v)
    }

    /// Reconstruct a `FileId` from its on-disk `u64` form, rejecting values that
    /// do not fit the `u32` id space (corruption / a foreign id from a wider build).
    pub const fn from_u64(v: u64) -> Option<Self> {
        if v <= u32::MAX as u64 {
            Some(FileId(v as u32))
        } else {
            None
        }
    }

    /// This id's index into the registry's append-only path table, or `None` for
    /// [`SELF`](Self::SELF) and reserved special ids (neither of which is a concrete
    /// registered file). Ordinary ids are 1-based, so the index is `id - 1`.
    pub(crate) const fn table_index(self) -> Option<usize> {
        if self.0 >= 1 && !self.is_special() {
            Some((self.0 - 1) as usize)
        } else {
            None
        }
    }

    /// This id refined to an **ordinary registered file** (non-[`SELF`](Self::SELF),
    /// non-[`special`](Self::is_special)), or `None` otherwise — the
    /// [`NonZeroU32::new`](core::num::NonZeroU32::new) analogue (`FileId` is the base
    /// type). The refinement carries "this names a concrete registered file" in the
    /// type, so [`ResolvedFileId::table_index`] is infallible.
    pub const fn resolve(self) -> Option<ResolvedFileId> {
        ResolvedFileId::new(self)
    }
}

/// A [`FileId`] **statically guaranteed to name an ordinary registered file** —
/// neither [`SELF`](FileId::SELF) nor a reserved [`special`](FileId::is_special) id,
/// i.e. one of the ascending `1..` ids with a real slot in the registry's path table.
///
/// The refined half of the `FileId` ↔ `ResolvedFileId` pair, mirroring
/// [`NonZeroU32`] over `u32`: constructed once at the boundary (a registry resolve),
/// after which [`table_index`](Self::table_index) needs no `Option`.
///
/// Backed by [`NonZeroU32`], so `Option<ResolvedFileId>` is the same 4 bytes as a
/// `FileId` (the `0`/`SELF` bit-pattern is the `None` niche). The non-special half of
/// the invariant is a value range, not a niche, so it is upheld by construction
/// ([`new`](Self::new) rejects specials). **Not** [`Pod`](bytemuck::Pod): `0` is an
/// invalid bit pattern — the wire/storage form is always the plain [`FileId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ResolvedFileId(NonZeroU32);

impl ResolvedFileId {
    /// Refine a [`FileId`] to an ordinary registered file, or `None` if it is
    /// [`SELF`](FileId::SELF) or [`special`](FileId::is_special) — the
    /// [`NonZeroU32::new`] analogue.
    pub const fn new(id: FileId) -> Option<Self> {
        if id.is_self() || id.is_special() {
            return None;
        }
        // `is_self()` already excluded `0`, so this is always `Some`.
        match NonZeroU32::new(id.0) {
            Some(nz) => Some(ResolvedFileId(nz)),
            None => None,
        }
    }

    /// Refine a [`FileId`] **without checking** — the [`NonZeroU32::new_unchecked`]
    /// analogue.
    ///
    /// # Safety
    /// `id` must be an ordinary registered file: non-[`SELF`](FileId::SELF) **and**
    /// non-[`special`](FileId::is_special). The non-`SELF` half is a soundness
    /// requirement — a `0`/`SELF` id is instant undefined behaviour, violating the
    /// backing [`NonZeroU32`]'s niche. The non-special half is a logical precondition
    /// that keeps [`table_index`](Self::table_index) correct.
    pub const unsafe fn new_unchecked(id: FileId) -> Self {
        // SAFETY: forwarded to the caller — `id` is non-`SELF` by contract.
        ResolvedFileId(unsafe { NonZeroU32::new_unchecked(id.0) })
    }

    /// Widen back to the base [`FileId`] (never [`SELF`](FileId::SELF)) — the
    /// [`NonZeroU32::get`] analogue.
    pub const fn get(self) -> FileId {
        FileId(self.0.get())
    }

    /// The raw non-zero id word, skipping the [`FileId`] wrapper.
    pub const fn as_u32(self) -> u32 {
        self.0.get()
    }

    /// This file's index into the registry's append-only path table — **infallible**
    /// here, since an ordinary id always has a slot (the refinement's payoff over
    /// [`FileId::table_index`]'s `Option`). Ordinary ids are 1-based.
    pub const fn table_index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl From<ResolvedFileId> for FileId {
    /// Widen to the base type — the `From<NonZeroU32> for u32` analogue.
    #[inline]
    fn from(id: ResolvedFileId) -> FileId {
        id.get()
    }
}

impl TryFrom<FileId> for ResolvedFileId {
    type Error = UnresolvedFileIdError;

    /// Refine to an ordinary registered file, failing on [`SELF`](FileId::SELF) or a
    /// special id — the `TryFrom<u32> for NonZeroU32` analogue.
    #[inline]
    fn try_from(id: FileId) -> Result<Self, UnresolvedFileIdError> {
        ResolvedFileId::new(id).ok_or(UnresolvedFileIdError)
    }
}

/// The error [`ResolvedFileId::try_from`] returns for a [`SELF`](FileId::SELF) or
/// [`special`](FileId::is_special) id — the
/// [`TryFromIntError`](core::num::TryFromIntError) analogue for this refinement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnresolvedFileIdError;

impl fmt::Display for UnresolvedFileIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("file id is SELF or a reserved special id, not an ordinary registered file")
    }
}

impl Error for UnresolvedFileIdError {}
