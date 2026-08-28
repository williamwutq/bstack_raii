//! The interpreters' **output vocabulary** — the structured values read out of, or
//! moved out of, a data file with no compiled-in Rust type: [`AnyRef`] (the RTTI
//! `&dyn Any`), the read [`Value`] tree, and the `move_out` transfer types
//! ([`Moved`] / [`VecRef`] / [`ForeignPtr`]).

use std::io;

use bstack::{BStack, BStackRange};

use crate::primitives::{EightCC, OwnershipKind};
use crate::types::traits::{BStackBlock, BStackCast};

use super::{HEADER_TAG_OFFSET, Shape, add_off};

/// A **runtime-typed reference** — an `(EightCC, offset)` into a data file, the RTTI
/// analog of `&dyn Any`. It bridges the interpreted world back to compiled-in types:
/// [`downcast`](Self::downcast) hands back a real typed block handle when the
/// reference's tag matches a type's compile-time [`eightcc`](BStackCast::eightcc),
/// otherwise the structure can be read generically (via [`RttiRegistry::read_any`]).
///
/// Obtain one from a typed pointer with [`RttiRegistry::any_ref`] (its tag is then
/// registry-authoritative — a stray pointer resolves to `None`), or straight from a
/// block's on-disk header with [`AnyRef::from_block`].
///
/// The match is an eightcc (hash) equality, so it is only as sound as tag
/// uniqueness. Within a program whose types were registered by
/// [`sync`](RttiRegistry::sync_compiled) that holds — sync rejects colliding types
/// (`[BSTACK0806]`) — so a successful `downcast` truly is that type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnyRef {
    tag: EightCC,
    offset: u64,
}

impl AnyRef {
    /// Construct from a known tag + offset. Prefer [`RttiRegistry::any_ref`] (which
    /// resolves the tag through the registry) or [`AnyRef::from_block`] — both are
    /// safe because they read the tag from an authoritative source.
    ///
    /// # Safety
    ///
    /// `offset` must name a live block whose on-disk header carries exactly `tag`.
    /// [`downcast`](Self::downcast) trusts the pair as given: a fabricated pair
    /// yields an owning handle over an arbitrary range, whose safe `bstack_drop`
    /// frees storage the caller does not own.
    #[inline(always)]
    pub unsafe fn new(tag: EightCC, offset: u64) -> Self {
        Self { tag, offset }
    }

    /// Recover the type tag from the target block's on-disk [`BlockHeader`](crate::BlockHeader)
    /// (`tag` at offset 8) — the no-registry path, one small read.
    pub fn from_block(data: &BStack, offset: u64) -> io::Result<Self> {
        let mut tag = [0u8; 8];
        data.get_into(add_off(offset, HEADER_TAG_OFFSET)?, &mut tag)?;
        Ok(Self {
            tag: EightCC(tag),
            offset,
        })
    }

    /// The reference's RTTI type tag.
    #[inline(always)]
    pub fn tag(&self) -> EightCC {
        self.tag
    }

    /// The reference's block offset.
    #[inline(always)]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Whether this reference is of the compiled-in type `T` (eightcc match).
    #[inline(always)]
    pub fn is<T: BStackBlock>(&self) -> bool {
        self.tag == <T as BStackCast>::eightcc()
    }

    /// Downcast to a `T` handle when the tag matches `T`'s compile-time eightcc,
    /// else `None` — the RTTI `Any::downcast`. The handle borrows the block at this
    /// reference's offset (length recovered from `size_of::<T::OnDisk>()`).
    pub fn downcast<T: BStackBlock>(&self) -> Option<T> {
        self.is::<T>().then(|| unsafe {
            T::from_range(BStackRange::new(
                self.offset,
                core::mem::size_of::<T::OnDisk>() as u64,
            ))
        })
    }
}

/// A structured value read out of a data file **with no compiled-in Rust type** —
/// the interpreter's output. Mirrors the [`Shape`] grammar. A reader (debugger,
/// generic serializer, repair tool) matches on this instead of a concrete type.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Raw POD bytes (a leaf): the on-disk little-endian bytes, undecoded.
    Pod(Box<[u8]>),
    /// A followed child block (`owned` / `strong` / `embed`): its tag and named
    /// fields, in declaration order.
    Block {
        tag: EightCC,
        fields: Box<[(String, Value)]>,
    },
    /// A followed enum block: its tag, the active variant's name, and that variant's
    /// named fields.
    Enum {
        tag: EightCC,
        variant: String,
        fields: Box<[(String, Value)]>,
    },
    /// An in-file reference that is **not** followed (`weak` / `ref`): the target's
    /// tag and the raw stored offset (`0` == null).
    Ref { tag: EightCC, offset: u64 },
    /// A cross-file [`Foreign`](crate::Foreign) pointer, recorded (not followed): the
    /// target's tag, ownership kind, file id (`0` == the current file), and offset.
    Foreign {
        tag: EightCC,
        kind: OwnershipKind,
        file_id: u64,
        offset: u64,
    },
    /// An absent nullable (`Option` niche `0`, or an empty/absent vector slot).
    Null,
    /// A present `Option`, wrapping its inner value.
    Some(Box<Value>),
    /// A fixed `[T; N]` array of elements.
    Array(Box<[Value]>),
    /// A dynamic `Vec<T>` / `String` of elements.
    Vec(Box<[Value]>),
    /// A tuple of elements.
    Tuple(Box<[Value]>),
    /// A class variable's value bytes (read from the **schema** record, not an
    /// instance — a class variable is not per-instance).
    Class(Box<[u8]>),
}

/// A whole vector moved out of a block by [`RttiRegistry::move_out`]: ownership of its
/// data block and every element, transferred as a unit — the RTTI analog of a
/// detached `BStackVec` handle. (A vec data block has no eightcc, so [`AnyRef`] can't
/// represent it.) The caller owns it: free the data block (and its owned elements) to
/// discard, or re-attach it to another vector field.
#[derive(Clone, Debug, PartialEq)]
pub struct VecRef {
    /// The vector's data block start (a `BStackByteVec`: `len` @0, `cap` @8, elements
    /// from 16).
    pub data_off: u64,
    /// The data block's allocated byte size (for reclaiming it).
    pub data_size: u64,
    /// The element shape (POD width, or a reference kind carrying the element tag).
    pub elem: Shape,
}

/// One immediate field moved out of a block by [`RttiRegistry::move_out`], with its
/// **ownership transferred to the caller** — the RTTI analog of a `bstack_move!` tuple
/// element. POD comes out by value; references come out as [`AnyRef`]s the caller now
/// owns (downcast / tear down / `swap` elsewhere).
#[derive(Clone, Debug, PartialEq)]
pub enum Moved {
    /// A POD field — or an inline POD array / tuple — copied out by value.
    Pod(Box<[u8]>),
    /// A single `owned` / `strong` / `ref` / (materialized) `embed` reference.
    /// `None` if the field was null.
    Ref(Option<AnyRef>),
    /// A `weak` reference (its control block). `None` if unset.
    Weak(Option<AnyRef>),
    /// A whole vector, transferred as a unit (see [`VecRef`]). `None` if the vec slot
    /// was empty / null.
    Vec(Option<VecRef>),
    /// A fixed reference **array** (`owned` / `strong` / `ref`), moved element-by-element
    /// — its inline offset storage lives in the freed shell, so unlike a vector there is
    /// no block to hand back whole. Each element is a **data** offset. `None` per null
    /// element.
    List(Box<[Option<AnyRef>]>),
    /// A fixed **weak** reference array (`[#[bstack_weak] T; N]`), moved element-by-
    /// element. Each element is its **control-block** offset — exactly like a scalar
    /// [`Weak`](Self::Weak), and *unlike* a data-offset [`List`](Self::List) — so the
    /// caller never mistakes control bytes for a `T` (e.g. `swap`ping one into a non-weak
    /// slot). `None` per unset element.
    WeakList(Box<[Option<AnyRef>]>),
    /// A cross-file [`Foreign`](crate::Foreign) pointer, transferred whole (the target
    /// lives in another file and outlives the freed shell): tag, ownership kind, file
    /// id, and offset (`offset == 0` == null). The caller now owns the reference.
    Foreign {
        tag: EightCC,
        kind: OwnershipKind,
        file_id: u64,
        offset: u64,
    },
    /// A fixed array of cross-file [`Foreign`](crate::Foreign) pointers (`[Foreign; N]`),
    /// moved element-by-element — the foreign analog of [`List`](Self::List). Its inline
    /// `WidePtr` storage dies with the freed shell, so each pointer is handed back
    /// (a `ForeignPtr` whose `offset == 0` is null). The caller now owns each reference.
    ForeignList(Box<[ForeignPtr]>),
    /// A tuple with at least one `Foreign` member, moved member-by-member: each element
    /// as its own [`Moved`] (POD by value, foreign as [`Foreign`](Self::Foreign)). Pure
    /// POD tuples come out as [`Pod`](Self::Pod) instead.
    Tuple(Box<[Moved]>),
    /// A **nested** reference array (`[[T; M]; N]`, …), moved outer-element-by-element —
    /// each inner container as its own [`Moved`] (a [`List`](Self::List) /
    /// [`ForeignList`](Self::ForeignList) / nested `Array`). A flat reference array is a
    /// [`List`](Self::List) / [`ForeignList`](Self::ForeignList); a pure-POD array
    /// (nested or not) is a [`Pod`](Self::Pod) blob.
    Array(Box<[Moved]>),
}

/// One cross-file [`Foreign`](crate::Foreign) pointer handed out by
/// [`move_out`](RttiRegistry::move_out) as an element of a [`Moved::ForeignList`]:
/// the target's tag, its ownership kind, and its `(file_id, offset)` (`offset == 0`
/// == null). The caller owns the reference and reclaims it in its own file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForeignPtr {
    pub tag: EightCC,
    pub kind: OwnershipKind,
    pub file_id: u64,
    pub offset: u64,
}

/// What a field path resolves to (see `RttiRegistry::resolve_field`): a per-instance
/// slot in the data file, or a `#[bstack_static]` class variable living schema-side.
pub(in crate::rtti) enum Resolved {
    /// A per-instance field: its absolute offset in the data file, and its shape.
    Instance { offset: u64, shape: Shape },
    /// A class variable, addressed by its owning type's tag + its name.
    Class { tag: EightCC, name: String },
}
