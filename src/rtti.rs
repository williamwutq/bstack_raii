//! Runtime Type Information (RTTI) — a persisted, self-describing schema stack.
//!
//! RTTI lets a **general program** read and interpret `bstack_raii` structures on
//! disk with **no compiled-in Rust types** — just `bstack_raii::rtti::…`. It is a
//! path *parallel* to the static macro-generated one; code that never opts in
//! never dispatches here.
//!
//! ## Model (see `BYTECODE.md` for the exact wire format)
//!
//! The RTTI file **is a plain [`bstack::BStack`]**: a sequence of records, one per
//! registered type, each appended with a single [`BStack::push`] (atomic +
//! crash-safe by bstack's own contract — no allocator, no WAL, no hand-rolled
//! header). A type's identity on the wire is its **ordinal** in that stack (the
//! order it was appended); because the stack is append-only, an ordinal is never
//! renumbered, so it is a stable, compact, `u32`-sized handle that resolves to the
//! full [`EightCC`] and descriptor by a direct array index.
//!
//! On open the whole stack is scanned into memory ([`RttiRegistry`]) so every
//! lookup is O(1). The scanned *structure* is immutable and safe to cache; a
//! **mutable class-variable value**, however, must be read live from the bstack —
//! another handle can rewrite its fixed-size slot in place.
//!
//! ## Codec
//!
//! [`encode_type`] / [`decode_type`] are the symmetric serialize / deserialize
//! pair for a type's record body; [`RttiRegistry::append`] frames + `push`es one,
//! [`RttiRegistry::load_type`] reads + decodes one. The `Shape` grammar parses
//! recursively — safe because a shape's depth is bounded by the *source type*
//! nesting, never by data depth (the data walk, added later, is the one that must
//! stay non-recursive).
//!
//! ## Status
//!
//! Read + write of struct and enum records is in place; the on-disk RTTI-typed
//! pointer is the existing [`WidePtr`] (its `type_index` is the ordinal `+ 1`).
//! The `#[bstack_class]` macro fills [`RTTI_TYPES`] at link time, and [`sync`]
//! appends every missing schema to a file. [`RttiRegistry::read_value`] /
//! [`RttiRegistry::read_ptr`] are the non-recursive **read interpreter** (schema over
//! a live data file → a [`Value`] tree, no compiled-in types), and
//! [`RttiRegistry::teardown`] is the non-recursive **free interpreter** (reclaims
//! `owned` / `embed` / `strong` / `weak` / `ref` / `vec` / array / tuple / option,
//! refcount decrements and all), and [`RttiRegistry::clone_value`] is the
//! non-recursive **deep-clone interpreter** (owned deep-copied, shared
//! refcount-bumped). [`RttiRegistry::class_value`] / [`RttiRegistry::set_class_value`]
//! read and write a `#[bstack_static]` class variable's value live in the schema
//! stack (a `#[bstack_mut]` one is set in place, crash-atomically). Cross-file
//! `Foreign` pointers are handled too — scalar or inside a `Vec` / array / tuple —
//! with teardown / clone / move_out resolving the target file through the registry and
//! acting on it in place.
//!
//! Individual fields are reached by a **path** (`["outer", "inner", …]`):
//! [`get`](RttiRegistry::get) reads one field, [`set`](RttiRegistry::set) overwrites
//! a POD / `ref` leaf, and [`swap`](RttiRegistry::swap) exchanges an owning reference
//! for another (eightcc-checked), handing the old target back.
//! [`move_out`](RttiRegistry::move_out) disassembles a block into its owned parts (a
//! [`SmallStringMap`]`<`[`Moved`]`>`), freeing only the shell — the RTTI `bstack_move!`.
//!
//! [`AnyRef`] bridges back to compiled-in types: it is the RTTI `&dyn Any`, whose
//! [`downcast`](AnyRef::downcast) hands back a real typed handle on an eightcc match,
//! falling back to generic interpretation ([`RttiRegistry::read_any`]) otherwise.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use bstack::{BStack, BStackRange};
use linkme::distributed_slice;

use crate::BStackRaiiAllocator;
use crate::block::{BStackBlock, BStackCast};
use crate::layout::read_u64_at;
use crate::primitives::{EightCC, WidePtr};
use crate::layout::{
    CTRL_BACKPTR_OFFSET, CTRL_DATA_OFFSET, CTRL_STRONG_OFFSET, CTRL_WEAK_OFFSET,
    RC_REFCOUNT_OFFSET, get_u64,
};
use crate::io_core::refcount;
use crate::registry::{self, FileId, ForeignHostAllocator};
use crate::util::small_map::SmallStringMap;
use crate::wal::{
    HeldLock, WalEntry, WalLog, WalStatus, finish_at_locked, persist_at, wal_append_alloc,
    wal_capacity_of, wal_set_idle,
};

/// Re-exports so the `#[bstack_class]` macro's generated registration code can name
/// `linkme` without the downstream crate depending on it directly. The generated
/// element uses `#[linkme(crate = ::bstack_raii::rtti::linkme)]` to override
/// linkme's hard-coded `::linkme` path against this re-export.
#[doc(hidden)]
pub use linkme;
#[doc(hidden)]
pub use linkme::distributed_slice as distributed_slice_reexport;

/// A type's stable identity within the one RTTI stack: its 0-based ordinal (the
/// order it was appended). Append-only ⇒ never renumbered.
pub type RttiOrdinal = u32;

// -- On-disk framing (mirrors BYTECODE.md) ---------------------------------

/// Bytes of a record's framing header: `eightcc[8] + body_len:u32 + _pad:u32`,
/// after which the `TypeDesc` body begins (8-aligned).
const RECORD_HEADER_LEN: u64 = 16;

const FLAG_ENUM: u8 = 0b0000_0001;
const FLAG_RC: u8 = 0b0000_0010;
const FLAG_WEAK: u8 = 0b0000_0100;

mod shape_tag {
    pub const POD: u8 = 0x00;
    pub const OWNED: u8 = 0x01;
    pub const STRONG: u8 = 0x02;
    pub const WEAK: u8 = 0x03;
    pub const REF: u8 = 0x04;
    pub const EMBED: u8 = 0x05;
    pub const FOREIGN: u8 = 0x06;
    pub const OPTION: u8 = 0x10;
    pub const ARRAY: u8 = 0x11;
    pub const VEC: u8 = 0x12;
    pub const TUPLE: u8 = 0x13;
    pub const CLASS: u8 = 0x20;
}

// NOTE: The invalid data thing might be widely used even outside of RTTI. Check if my claim
// is accurate, and if is, we may want to refactor this.
#[inline(always)]
fn corrupt(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// NOTE: An interesting idea had came to my mind: what if we brand our u64 offsets?
// Maybe a DiskOffset or something could make our code more maintainable
// We can also make DiskOffset behave instead like NonZeroU64, due to the null niche
// requirement that is generally applied in this crate
/// Add two on-disk offsets/lengths, rejecting overflow. Every interpreter walk
/// (`read_value` / `teardown` / `clone_value`) chains additions off a **root**
/// offset that can be entirely attacker/caller-controlled (a forged pointer, or
/// — as here — a fuzzed argument); an unchecked `+` either panics under
/// `overflow-checks` or silently wraps to an unrelated in-bounds offset in a
/// release build. Reject cleanly instead.
fn add_off(a: u64, b: u64) -> io::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| corrupt("[BSTACK081A] RTTI offset arithmetic overflow"))
}

/// Multiply an on-disk element stride by an index, rejecting overflow — the
/// `mul` counterpart of [`add_off`] for `Array`/`Vec` element offsets.
fn mul_off(a: u64, b: u64) -> io::Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| corrupt("[BSTACK081A] RTTI offset arithmetic overflow"))
}

/// Narrow a compile-time-fixed layout quantity (a field's `offset_of!`, a POD
/// field's `size_of`, or an array's element count) to the `u32` the RTTI wire
/// format stores it as, panicking with a clear diagnostic instead of silently
/// wrapping. Called from `#[bstack_class]`-generated schema-builder code
/// (`RttiRegistration::build: fn() -> RttiType`, which cannot return `Result`),
/// so the only way to trip this is a pathologically large compiled type (a
/// multi-GiB inline field, or an array length in the billions) — never
/// attacker-controlled data, but a silent wraparound here would otherwise
/// persist a schema whose recorded offset/count doesn't match the real layout.
#[doc(hidden)]
pub fn rtti_narrow_u32(x: usize, what: &str) -> u32 {
    u32::try_from(x).unwrap_or_else(|_| {
        panic!("[BSTACK0817] RTTI {what} exceeds the maximum encodable size (u32)")
    })
}

/// Error for an RTTI record component whose length overflows its fixed on-disk field
/// width. Encode-side lengths (name, field/variant count, tuple arity, shape length,
/// class-value length, body length) are written as `u8`/`u16`/`u32`; a type too large
/// or too deeply nested to serialize is rejected at `append`/`sync` **before** a
/// silently-truncated, permanently-unreadable record is written.
fn too_large(what: &str, limit: &str) -> io::Error {
    corrupt(format!(
        "[BSTACK0817] RTTI {what} exceeds the maximum encodable size ({limit})"
    ))
}

/// Build an RTTI-typed pointer: a [`WidePtr`] to `(file_id, offset)` tagged
/// with `ordinal`. `file_id == 0` ⇒ `SELF`. For an untyped pointer (type recovered
/// from the target block header on deref) use [`WidePtr::from_raw`] with a `0` type
/// index. The raw `(file_id, offset)` inputs are decoded through the wire boundary.
pub fn typed_ptr(file_id: u64, offset: u64, ordinal: RttiOrdinal) -> WidePtr {
    WidePtr::from_raw(file_id, ordinal + 1, offset)
}

/// Build an RTTI field path from **dotted field names** for
/// [`get`](RttiRegistry::get) / [`set`](RttiRegistry::set) / [`swap`](RttiRegistry::swap):
/// `rtti_path!(outer.inner.leaf)` expands to `&["outer", "inner", "leaf"]`, and a
/// single name `rtti_path!(field)` to `&["field"]`. The result is a `&[&str]`, so it
/// drops straight into the path argument.
///
/// ```
/// # use bstack_raii::rtti_path;
/// let p: &[&str] = rtti_path!(inner.x);
/// assert_eq!(p, &["inner", "x"]);
/// ```
#[macro_export]
macro_rules! rtti_path {
    ($($seg:ident).+) => {
        &[$(::core::stringify!($seg)),+]
    };
}

// NOTE: for our types, check the consistency of the inclusion of "BStack"
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

    /// Recover the type tag from the target block's on-disk [`BlockHeader`](crate::layout::BlockHeader)
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

// -- Little-endian cursor codec --------------------------------------------

/// Append-only little-endian writer over a growing byte buffer.
#[derive(Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    #[inline(always)]
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    #[inline(always)]
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    fn eightcc(&mut self, v: EightCC) {
        self.buf.extend_from_slice(&v.0);
    }
    #[inline(always)]
    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    /// Pad with zero bytes up to the next `a`-byte boundary (`a` a power of two).
    ///
    /// Requires `a` to be a power of 2
    #[inline(always)]
    fn align(&mut self, a: usize) {
        let mask = a - 1;
        let new_len = (self.buf.len() + mask) & !mask;
        self.buf.resize(new_len, 0);
    }
}

/// Bounds-checked little-endian reader over a byte slice.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    #[inline(always)]
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| corrupt("[BSTACK0804] truncated RTTI record"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    #[inline(always)]
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    #[inline(always)]
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    #[inline(always)]
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    #[inline(always)]
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    #[inline(always)]
    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    #[inline(always)]
    fn eightcc(&mut self) -> io::Result<EightCC> {
        Ok(EightCC(self.take(8)?.try_into().unwrap()))
    }
    #[inline]
    fn string(&mut self, n: usize) -> io::Result<String> {
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| corrupt("[BSTACK0802] RTTI name is not valid UTF-8"))
    }
    #[inline(always)]
    /// Skip zero-padding up to the next `a`-byte boundary.
    fn align(&mut self, a: usize) -> io::Result<()> {
        let aligned = (self.pos + a - 1) & !(a - 1);
        if aligned > self.buf.len() {
            return Err(corrupt("[BSTACK0804] truncated RTTI record (alignment)"));
        }
        self.pos = aligned;
        Ok(())
    }
}

// -- Parsed, in-memory schema (structure only) -----------------------------

/// The four ownership kinds an interpreted reference can carry — re-exported from
/// [the primitives](crate::primitives::OwnershipKind), the crate-wide vocabulary.
/// A `Foreign` leaf, a struct field, and a variant payload all classify with it.
pub use crate::primitives::OwnershipKind;

/// The info-complex node — a field's type structure, its leaves carrying the RAII
/// kind the interpreter dispatches on.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Pod {
        width: u32,
    },
    Owned(EightCC),
    Strong(EightCC),
    Weak(EightCC),
    Ref(EightCC),
    Embed(EightCC),
    /// A cross-file [`Foreign`](crate::Foreign) pointer: the target's tag **and** the
    /// ownership kind of the target *in its own file* (which drives teardown / clone).
    Foreign {
        tag: EightCC,
        kind: OwnershipKind,
    },
    Option(Box<Shape>),
    Array {
        n: u32,
        inner: Box<Shape>,
    },
    Vec(Box<Shape>),
    Tuple(Box<[Shape]>),
    /// A class variable stored inline in the record. For the const case the bytes
    /// are the snapshot here; for the mutable case they are the initial value (the
    /// live value is read from the stack at the field's slot).
    Class {
        mutable: bool,
        inner: Box<Shape>,
        value: Box<[u8]>,
    },
}

/// Maximum shape-tree nesting depth accepted when decoding an on-disk record. A real
/// `#[bstack_class]` field nests only a handful of levels (e.g. `Option<Vec<[T; N]>>`);
/// anything deeper is a corrupt or hand-forged record, and decoding it recursively
/// would otherwise overflow the native stack. Generous relative to any real type, small
/// relative to the stack.
const MAX_SHAPE_DEPTH: usize = 64;

impl Shape {
    fn encode(&self, w: &mut Writer) -> io::Result<()> {
        use shape_tag as t;
        match self {
            Shape::Pod { width } => {
                w.u8(t::POD);
                w.u32(*width);
            }
            Shape::Owned(cc) => {
                w.u8(t::OWNED);
                w.eightcc(*cc);
            }
            Shape::Strong(cc) => {
                w.u8(t::STRONG);
                w.eightcc(*cc);
            }
            Shape::Weak(cc) => {
                w.u8(t::WEAK);
                w.eightcc(*cc);
            }
            Shape::Ref(cc) => {
                w.u8(t::REF);
                w.eightcc(*cc);
            }
            Shape::Embed(cc) => {
                w.u8(t::EMBED);
                w.eightcc(*cc);
            }
            Shape::Foreign { tag, kind } => {
                w.u8(t::FOREIGN);
                w.eightcc(*tag);
                w.u8(*kind as u8);
            }
            Shape::Option(inner) => {
                w.u8(t::OPTION);
                inner.encode(w)?;
            }
            Shape::Array { n, inner } => {
                w.u8(t::ARRAY);
                w.u32(*n);
                inner.encode(w)?;
            }
            Shape::Vec(inner) => {
                w.u8(t::VEC);
                inner.encode(w)?;
            }
            Shape::Tuple(items) => {
                w.u8(t::TUPLE);
                // Arity is stored in one byte (a tuple's decode reads a `u8` count).
                let arity = u8::try_from(items.len())
                    .map_err(|_| too_large("tuple arity", "255 elements"))?;
                w.u8(arity);
                for it in items {
                    it.encode(w)?;
                }
            }
            Shape::Class {
                mutable,
                inner,
                value,
            } => {
                w.u8(t::CLASS);
                w.u8(u8::from(*mutable));
                inner.encode(w)?;
                let value_len = u32::try_from(value.len())
                    .map_err(|_| too_large("class-variable value length", "4 GiB"))?;
                w.u32(value_len);
                w.bytes(value);
            }
        }
        Ok(())
    }

    fn decode(r: &mut Reader) -> io::Result<Shape> {
        Self::decode_at(r, 0)
    }

    /// Decode one shape at nesting `depth`, refusing to recurse past
    /// [`MAX_SHAPE_DEPTH`]. Untrusted on-disk bytes drive the recursion (one nesting
    /// tag per `Option` / `Array` / `Vec` / `Tuple` / `Class` level), so an
    /// unbounded decode would let a corrupt record overflow the native stack during
    /// `load_type` / `open`. (Width is already bounded — a tuple's arity is a `u8`.)
    fn decode_at(r: &mut Reader, depth: usize) -> io::Result<Shape> {
        use shape_tag as t;
        if depth >= MAX_SHAPE_DEPTH {
            return Err(corrupt(
                "[BSTACK0818] RTTI shape nesting exceeds the maximum depth",
            ));
        }
        let tag = r.u8()?;
        Ok(match tag {
            t::POD => Shape::Pod { width: r.u32()? },
            t::OWNED => Shape::Owned(r.eightcc()?),
            t::STRONG => Shape::Strong(r.eightcc()?),
            t::WEAK => Shape::Weak(r.eightcc()?),
            t::REF => Shape::Ref(r.eightcc()?),
            t::EMBED => Shape::Embed(r.eightcc()?),
            t::FOREIGN => {
                let tag = r.eightcc()?;
                let kb = r.u8()?;
                let kind = OwnershipKind::from_u8(kb).ok_or_else(|| {
                    corrupt(format!("[BSTACK0803] unknown RTTI foreign kind {kb:#04x}"))
                })?;
                Shape::Foreign { tag, kind }
            }
            t::OPTION => Shape::Option(Box::new(Shape::decode_at(r, depth + 1)?)),
            t::ARRAY => {
                let n = r.u32()?;
                Shape::Array {
                    n,
                    inner: Box::new(Shape::decode_at(r, depth + 1)?),
                }
            }
            t::VEC => Shape::Vec(Box::new(Shape::decode_at(r, depth + 1)?)),
            t::TUPLE => {
                let k = r.u8()? as usize;
                let items = (0..k)
                    .map(|_| Shape::decode_at(r, depth + 1))
                    .collect::<Result<Box<[_]>, _>>()?;
                Shape::Tuple(items)
            }
            t::CLASS => {
                let mutable = r.u8()? != 0;
                let inner = Box::new(Shape::decode_at(r, depth + 1)?);
                let value_len = r.u32()? as usize;
                let value = r.take(value_len)?.into();
                Shape::Class {
                    mutable,
                    inner,
                    value,
                }
            }
            other => {
                return Err(corrupt(format!(
                    "[BSTACK0803] unknown RTTI shape tag {other:#04x}"
                )));
            }
        })
    }
}

/// One field of a type: its name, its absolute byte offset within the block's
/// `OnDisk` (unused for `CLASS` fields), and its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct RttiField {
    pub name: String,
    pub offset: u32,
    pub shape: Shape,
}

impl RttiField {
    fn encode(&self, w: &mut Writer) -> io::Result<()> {
        let mut sw = Writer::default();
        self.shape.encode(&mut sw)?;
        let name = self.name.as_bytes();
        let name_len =
            u16::try_from(name.len()).map_err(|_| too_large("field name length", "65535 bytes"))?;
        let shape_len = u16::try_from(sw.buf.len())
            .map_err(|_| too_large("field shape encoding length", "65535 bytes"))?;
        w.u32(self.offset);
        w.u16(name_len);
        w.u16(shape_len);
        w.bytes(name);
        w.align(4); // name pad → shape 4-aligned
        w.bytes(&sw.buf);
        w.align(4); // end pad → next field 4-aligned
        Ok(())
    }

    fn decode(r: &mut Reader) -> io::Result<RttiField> {
        let offset = r.u32()?;
        let name_len = r.u16()? as usize;
        let shape_len = r.u16()? as usize;
        let name = r.string(name_len)?;
        r.align(4)?;
        let shape_start = r.pos;
        let shape = Shape::decode(r)?;
        if r.pos - shape_start != shape_len {
            return Err(corrupt("[BSTACK0805] RTTI field shape length mismatch"));
        }
        r.align(4)?;
        Ok(RttiField {
            name,
            offset,
            shape,
        })
    }
}

/// One variant of an enum: its name, discriminant value, and fields (offsets
/// relative to the payload).
#[derive(Clone, Debug, PartialEq)]
pub struct RttiVariant {
    pub name: String,
    pub disc_value: i64,
    pub fields: Box<[RttiField]>,
}

impl RttiVariant {
    fn encode(&self, w: &mut Writer) -> io::Result<()> {
        w.align(8); // each variant is 8-aligned
        w.i64(self.disc_value);
        let name = self.name.as_bytes();
        let name_len = u16::try_from(name.len())
            .map_err(|_| too_large("variant name length", "65535 bytes"))?;
        let field_count = u16::try_from(self.fields.len())
            .map_err(|_| too_large("variant field count", "65535 fields"))?;
        w.u16(name_len);
        w.u16(field_count);
        w.u32(0); // _pad
        w.bytes(name);
        w.align(8); // name pad → fields aligned
        for f in &self.fields {
            f.encode(w)?;
        }
        Ok(())
    }

    fn decode(r: &mut Reader) -> io::Result<RttiVariant> {
        r.align(8)?;
        let disc_value = r.i64()?;
        let name_len = r.u16()? as usize;
        let field_count = r.u16()? as usize;
        let _pad = r.u32()?;
        let name = r.string(name_len)?;
        r.align(8)?;
        let fields = (0..field_count)
            .map(|_| RttiField::decode(r))
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(RttiVariant {
            name,
            disc_value,
            fields,
        })
    }
}

/// The struct fields or enum variants of a type.
#[derive(Clone, Debug, PartialEq)]
pub enum RttiBody {
    Struct(Box<[RttiField]>),
    Enum(RttiEnum),
}

/// The enum-specific header + variants.
#[derive(Clone, Debug, PartialEq)]
pub struct RttiEnum {
    pub disc_width: u8,
    pub disc_off: u16,
    pub payload_off: u16,
    pub variants: Box<[RttiVariant]>,
}

/// A parsed type descriptor. Structure only — mutable class-variable *values* are
/// read live from the stack, never cached here.
#[derive(Clone, Debug, PartialEq)]
pub struct RttiType {
    pub tag: EightCC,
    pub name: String,
    /// Target has an inline refcount (`rc` mode) — how a `Strong` field to it is
    /// bumped on clone.
    pub rc: bool,
    /// Target has a control block (`rc, weak` mode).
    pub weak: bool,
    /// The control block's tag (`Some` iff `weak`) — persisted so `swap` can confirm
    /// a `weak` target's control block directly by its header tag, not only via its
    /// forward data pointer.
    pub ctrl_tag: Option<EightCC>,
    pub ondisk_size: u64,
    pub body: RttiBody,
}

/// Serialize a type's record **body** (the `TypeDesc`, without the record framing).
pub fn encode_type(ty: &RttiType) -> io::Result<Vec<u8>> {
    let mut w = Writer::default();

    let raw_count = match &ty.body {
        RttiBody::Struct(fields) => fields.len(),
        RttiBody::Enum(e) => e.variants.len(),
    };
    let count = u16::try_from(raw_count).map_err(|_| match &ty.body {
        RttiBody::Struct(_) => too_large("struct field count", "65535 fields"),
        RttiBody::Enum(_) => too_large("enum variant count", "65535 variants"),
    })?;
    let (flags_kind, disc_width, disc_off, payload_off) = match &ty.body {
        RttiBody::Struct(_) => (0u8, 0u8, 0u16, 0u16),
        RttiBody::Enum(e) => (FLAG_ENUM, e.disc_width, e.disc_off, e.payload_off),
    };
    let mut flags = flags_kind;
    if ty.rc {
        flags |= FLAG_RC;
    }
    if ty.weak {
        flags |= FLAG_WEAK;
    }

    let name = ty.name.as_bytes();
    let name_len =
        u16::try_from(name.len()).map_err(|_| too_large("type name length", "65535 bytes"))?;
    w.u8(flags);
    w.u8(disc_width);
    w.u16(name_len);
    w.u16(count);
    w.u16(disc_off);
    w.u16(payload_off);
    w.u64(ty.ondisk_size);
    // Control tag: zero for a non-weak type, the control-block tag for a
    // weak one. Always 8 bytes so the header stays fixed-width.
    w.eightcc(ty.ctrl_tag.unwrap_or(EightCC([0u8; 8])));
    w.bytes(name);
    w.align(8); // name pad → body 8-aligned

    match &ty.body {
        RttiBody::Struct(fields) => {
            for f in fields {
                f.encode(&mut w)?;
            }
        }
        RttiBody::Enum(e) => {
            for v in &e.variants {
                v.encode(&mut w)?;
            }
        }
    }
    Ok(w.buf)
}

/// Deserialize a type's record **body** back into an [`RttiType`], given its tag
/// (which lives in the record framing, not the body).
pub fn decode_type(tag: EightCC, body: &[u8]) -> io::Result<RttiType> {
    let mut r = Reader::new(body);
    let flags = r.u8()?;
    let disc_width = r.u8()?;
    let name_len = r.u16()? as usize;
    let count = r.u16()? as usize;
    let disc_off = r.u16()?;
    let payload_off = r.u16()?;
    let ondisk_size = r.u64()?;
    let ctrl_tag_raw = r.eightcc()?;
    let name = r.string(name_len)?;
    r.align(8)?;
    let weak = flags & FLAG_WEAK != 0;

    let body = if flags & FLAG_ENUM != 0 {
        if disc_width > 8 {
            // A discriminant is read into a `u64`; reject a corrupt wider width on
            // load so no interpreter path later slices past an 8-byte buffer.
            return Err(corrupt(
                "[BSTACK0816] RTTI enum discriminant width exceeds 8 bytes",
            ));
        }
        if disc_width == 0 {
            // `disc_mask(0)` is 0 and a 0-byte read yields 0, so every variant
            // search would silently match the first variant; a corrupt record
            // must error, not mis-parse.
            return Err(corrupt("[BSTACK0816] RTTI enum discriminant width is zero"));
        }
        let variants = (0..count)
            .map(|_| RttiVariant::decode(&mut r))
            .collect::<Result<Box<[_]>, _>>()?;
        RttiBody::Enum(RttiEnum {
            disc_width,
            disc_off,
            payload_off,
            variants,
        })
    } else {
        let fields = (0..count)
            .map(|_| RttiField::decode(&mut r))
            .collect::<Result<Box<[_]>, _>>()?;
        RttiBody::Struct(fields)
    };

    Ok(RttiType {
        tag,
        name,
        rc: flags & FLAG_RC != 0,
        weak,
        ctrl_tag: weak.then_some(ctrl_tag_raw),
        ondisk_size,
        body,
    })
}

/// Frame an already-encoded body into a full record: `eightcc + body_len + _pad +
/// body`, padded to 8. Returns the framed record and the validated `body_len`.
fn frame_record(tag: EightCC, body: &[u8]) -> io::Result<(Vec<u8>, u32)> {
    let body_len =
        u32::try_from(body.len()).map_err(|_| too_large("record body length", "4 GiB"))?;
    let mut w = Writer::default();
    w.eightcc(tag);
    w.u32(body_len);
    w.u32(0); // _pad
    w.bytes(body);
    w.align(8); // whole record 8-aligned
    Ok((w.buf, body_len))
}

// -- Compile-time registration (linkme) ------------------------------------

/// One compiled-in type's link-time registration: a builder for its parsed
/// schema descriptor. The `#[bstack_class]` macro emits exactly one of these per
/// type into [`RTTI_TYPES`]; [`sync`] walks the slice and appends any missing
/// schema to the file.
pub struct RttiRegistration {
    /// Builds the type's descriptor. Allocates (`String`/`Vec`); called once per
    /// type at [`sync`] time, never on a hot path.
    pub build: fn() -> RttiType,
}

/// The set of every `#[bstack_class]` type compiled into this binary, collected
/// at **link time** via `linkme` — no life-before-main (unlike `inventory`'s
/// `ctor`), no instantiation required, no hand-enumeration. Each entry is emitted
/// by the macro. [`sync`] is the sole consumer.
#[distributed_slice]
pub static RTTI_TYPES: [RttiRegistration];

/// Open (creating if absent) the RTTI file at `path`, append every compiled-in
/// schema it does not already carry, and return the loaded registry. Idempotent
/// and safe to call on every open. Runs eightcc-collision detection. This is the
/// producer-side entry point; see [`RttiRegistry::sync_compiled`] for the details.
pub fn sync(path: impl AsRef<Path>) -> io::Result<RttiRegistry> {
    let mut reg = RttiRegistry::open(path)?;
    reg.sync_compiled()?;
    Ok(reg)
}

// -- The in-memory registry ------------------------------------------------

/// A scanned RTTI record: its tag, where its framing header begins in the stack,
/// and its body length. Ordinal = position in [`RttiRegistry::records`].
struct RecordRef {
    tag: EightCC,
    offset: u64,
    body_len: u32,
}

/// The whole RTTI stack loaded into memory: the ordered records plus a
/// tag→ordinal index. Holds the open [`BStack`] so mutable class-variable values
/// can be read (and later written) live, and so new types can be appended.
pub struct RttiRegistry {
    stack: BStack,
    records: Vec<RecordRef>,
    by_tag: HashMap<EightCC, RttiOrdinal>,
}

impl RttiRegistry {
    /// Open (creating if absent) an RTTI stack and scan every record into memory.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let stack = BStack::open(path)?;
        let mut reg = Self {
            stack,
            records: Vec::new(),
            by_tag: HashMap::new(),
        };
        reg.scan()?;
        Ok(reg)
    }

    /// Walk the stack front-to-back, recording each record's tag / offset / length
    /// and building the tag→ordinal map. A repeated tag is corruption (eightcc is
    /// the resolution key; two distinct types must never share one).
    fn scan(&mut self) -> io::Result<()> {
        let len = self.stack.len()?;
        let mut off = 0u64;
        while off < len {
            let mut header = [0u8; RECORD_HEADER_LEN as usize];
            self.stack.get_into(off, &mut header)?;
            let tag = EightCC(header[0..8].try_into().unwrap());
            let body_len = u32::from_le_bytes(header[8..12].try_into().unwrap());
            // Bound the untrusted length against the stack: a truncated or
            // hand-edited record must fail as `InvalidData` here, not size a
            // multi-GiB allocation in `load_type` (or scan past the end).
            if RECORD_HEADER_LEN + body_len as u64 > len - off {
                return Err(corrupt(
                    "[BSTACK0800] RTTI record body length runs past the end of the schema stack",
                ));
            }
            self.index(tag, off, body_len)?;
            // Advance to the next record, 8-byte aligned.
            off += (RECORD_HEADER_LEN + body_len as u64 + 7) & !7;
        }
        Ok(())
    }

    /// Record one scanned/appended entry, rejecting a duplicate tag.
    fn index(&mut self, tag: EightCC, offset: u64, body_len: u32) -> io::Result<RttiOrdinal> {
        let ordinal = self.records.len() as RttiOrdinal;
        if self.by_tag.insert(tag, ordinal).is_some() {
            return Err(corrupt(
                "[BSTACK0800] duplicate RTTI eightcc — two types share one tag",
            ));
        }
        self.records.push(RecordRef {
            tag,
            offset,
            body_len,
        });
        Ok(ordinal)
    }

    /// Append a new type's record to the stack and index it. Errors if its tag is
    /// already registered.
    pub fn append(&mut self, ty: &RttiType) -> io::Result<RttiOrdinal> {
        if self.by_tag.contains_key(&ty.tag) {
            return Err(corrupt(
                "[BSTACK0800] duplicate RTTI eightcc — type already registered",
            ));
        }
        // Encode once and validate every on-disk length fits its field width, so a
        // type too large / deeply nested to serialize is rejected here rather than
        // written as a silently-truncated, permanently-unreadable record.
        let body = encode_type(ty)?;
        let (record, body_len) = frame_record(ty.tag, &body)?;
        let offset = self.stack.push(&record)?;
        self.index(ty.tag, offset, body_len)
    }

    /// Append every compiled-in [`RTTI_TYPES`] descriptor this file is missing, in
    /// registration order. Idempotent: a type already present (matched by tag) is
    /// skipped. Returns the number of newly appended types.
    ///
    /// Runs **eightcc-collision detection** — the write-side guard: because a tag
    /// is the resolution key, two *distinct* types (different names) hashing to one
    /// eightcc is corruption, caught here rather than silently overwriting. A tag
    /// that is already on disk under the *same* name is a benign re-sync.
    pub fn sync_compiled(&mut self) -> io::Result<usize> {
        let mut appended = 0;
        // Guards a collision *within* the compiled-in set (two builders, one tag).
        let mut seen: HashMap<EightCC, RttiType> = HashMap::new();
        for reg in RTTI_TYPES.iter() {
            let ty = (reg.build)();
            if let Some(prev) = seen.get(&ty.tag) {
                if prev.name != ty.name {
                    return Err(corrupt(format!(
                        "[BSTACK0806] RTTI eightcc collision: '{}' and '{}' \
                         hash to one tag",
                        prev.name, ty.name
                    )));
                }
                // Same tag AND same name is still a collision when the layouts
                // differ — the tag ignores the module path, so `v1::Node` and
                // `v2::Node` arrive here as one name. Only a byte-identical
                // layout is genuinely "the same type registered twice".
                if !layouts_match(prev, &ty) {
                    return Err(corrupt(format!(
                        "[BSTACK0806] RTTI eightcc collision: two distinct types \
                         both named '{}' (same-named types in different modules?) \
                         share one tag",
                        ty.name
                    )));
                }
                continue; // same type registered twice — nothing to do
            }
            seen.insert(ty.tag, ty.clone());

            match self.ordinal_of(ty.tag) {
                Some(ord) => {
                    // Already on disk: it must be the SAME type AND the SAME layout.
                    let existing = self.load_type(ord)?;
                    if existing.name != ty.name {
                        // Different type, same tag — an eightcc collision.
                        return Err(corrupt(format!(
                            "[BSTACK0806] RTTI eightcc collision: on-disk '{}' vs \
                             compiled '{}' share one tag",
                            existing.name, ty.name
                        )));
                    }
                    if !layouts_match(&existing, &ty) {
                        // Same name, different layout: fields added / removed / reordered
                        // / resized, `rc`/`weak` mode, `ondisk_size`, or a *const*
                        // class-variable value changed. (A *mutable* class variable's
                        // value is updated in place and so legitimately differs between
                        // the compiled initial value and the persisted current one — it
                        // is excluded from the comparison.) The eightcc is derived from
                        // the name only, so neither it nor the name moved, but the
                        // persisted offsets / shapes no longer describe the compiled
                        // type. Reject rather than silently keep the stale descriptor.
                        return Err(corrupt(format!(
                            "[BSTACK0814] RTTI schema mismatch for '{}': the persisted \
                             layout differs from the compiled type (a field was added, \
                             removed, reordered, or resized). The on-disk data was \
                             written against the old layout.",
                            ty.name
                        )));
                    }
                }
                None => {
                    self.append(&ty)?;
                    appended += 1;
                }
            }
        }
        Ok(appended)
    }

    /// Number of registered types.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no types are registered.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The ordinal a tag resolves to, if present.
    pub fn ordinal_of(&self, tag: EightCC) -> Option<RttiOrdinal> {
        self.by_tag.get(&tag).copied()
    }

    /// The tag at an ordinal, if in range.
    pub fn tag_of(&self, ordinal: RttiOrdinal) -> Option<EightCC> {
        self.records.get(ordinal as usize).map(|r| r.tag)
    }

    /// Resolve a pointer's `type_index` to a live ordinal. `None` for an untyped
    /// pointer or an out-of-range index.
    pub fn resolve_ptr(&self, ptr: WidePtr) -> Option<RttiOrdinal> {
        let ord = ptr.type_id().ordinal()?;
        ((ord as usize) < self.records.len()).then_some(ord)
    }

    /// Read + decode the full descriptor for a type.
    pub fn load_type(&self, ordinal: RttiOrdinal) -> io::Result<RttiType> {
        let rec = self.records.get(ordinal as usize).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "[BSTACK0801] RTTI ordinal out of range",
            )
        })?;
        // NOTE: is rec.body_len bounded and will not result in dangerous allocations?
        // check similar patterns
        let mut body = vec![0u8; rec.body_len as usize];
        self.stack
            .get_into(rec.offset + RECORD_HEADER_LEN, &mut body)?;
        decode_type(rec.tag, &body)
    }

    // NOTE: as I increasingly see these kind of functions, I realized the convenience
    // of a reader / writer / buffer, which can be passed in
    /// Read a class variable's current value bytes **live** from the schema stack.
    /// A mutable one may have been rewritten (by [`set_class_value`](Self::set_class_value),
    /// possibly through another handle) since it was registered, so the snapshot in a
    /// cached [`load_type`](Self::load_type) can be stale — this always reads the file.
    /// Works for const and mutable class variables alike.
    pub fn class_value(&self, tag: EightCC, name: &str) -> io::Result<Vec<u8>> {
        let (off, len, _mutable) = self.locate_class_value(tag, name)?;
        let mut buf = vec![0u8; len];
        self.stack.get_into(off, &mut buf)?;
        Ok(buf)
    }

    /// Overwrite a **mutable** (`#[bstack_mut]`) class variable's value in place — one
    /// atomic write to the schema stack. The value slot is fixed-size (the mutable
    /// case requires a `Sized` type), so the record never moves and the append-only
    /// structure is preserved; the write is crash-atomic under the bstack's own lock.
    ///
    /// Errors if the class variable is absent, is `const` (not `#[bstack_mut]`), or
    /// `value` is not the slot's exact width.
    pub fn set_class_value(&self, tag: EightCC, name: &str, value: &[u8]) -> io::Result<()> {
        let (off, len, mutable) = self.locate_class_value(tag, name)?;
        if !mutable {
            return Err(class_error(format!(
                "`{name}` is a const class variable; only a `#[bstack_mut]` one is settable"
            )));
        }
        if value.len() != len {
            return Err(class_error(format!(
                "class variable `{name}` is {len} bytes, got {}",
                value.len()
            )));
        }
        self.stack.set(off, value)
    }

    /// Locate a class variable's value slot: its absolute offset in the schema stack,
    /// byte length, and mutability.
    fn locate_class_value(&self, tag: EightCC, name: &str) -> io::Result<(u64, usize, bool)> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        let rec = &self.records[ord as usize];
        let mut body = vec![0u8; rec.body_len as usize];
        self.stack
            .get_into(rec.offset + RECORD_HEADER_LEN, &mut body)?;
        let (pos, len, mutable) = class_value_slot(&body, name)?
            .ok_or_else(|| class_error(format!("no class variable named `{name}`")))?;
        Ok((rec.offset + RECORD_HEADER_LEN + pos as u64, len, mutable))
    }
}

// -- The read interpreter --------------------------------------------------

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
enum Resolved {
    /// A per-instance field: its absolute offset in the data file, and its shape.
    Instance { offset: u64, shape: Shape },
    /// A class variable, addressed by its owning type's tag + its name.
    Class { tag: EightCC, name: String },
}

/// One step of the non-recursive walk. The interpreter runs a `work` stack of these
/// against a `results` value stack: leaf steps push a [`Value`]; an `Assemble*` step
/// pops the `n` values its children pushed and combines them into one.
enum Op {
    /// Read the block of type `ord` at `block_off` (its whole `OnDisk`, header + fields).
    Block {
        ord: RttiOrdinal,
        block_off: u64,
    },
    /// Interpret one shape at an absolute data offset.
    Shape {
        shape: Shape,
        offset: u64,
    },
    /// Pop `n` field values (child-first order) and assemble a struct block.
    MakeBlock {
        tag: EightCC,
        names: Box<[String]>,
    },
    /// Pop `n` field values and assemble an enum block.
    MakeEnum {
        tag: EightCC,
        variant: String,
        names: Box<[String]>,
    },
    /// Pop `n` values into an array / vec / tuple.
    MakeArray(usize),
    MakeVec(usize),
    MakeTuple(usize),
    /// Pop one value and wrap it in `Some`.
    MakeSome,
}

/// One step of the non-recursive teardown walk (see [`RttiRegistry::teardown`]).
enum TdOp {
    /// Visit the block of type `ord` at `block_off`: free its range (unless `emit`
    /// is false, for an inline `#[embed]` child) and walk its fields.
    Block {
        ord: RttiOrdinal,
        block_off: u64,
        emit: bool,
    },
    /// Interpret one shape at an absolute data offset, freeing what it owns.
    Shape { shape: Shape, offset: u64 },
}

// NOTE: refactor: we may want to refacto these into a rtti module, as it is basically
// the normal things, and this is its equivelant of clone.rs
/// One step of the non-recursive clone walk (see [`RttiRegistry::clone_value`]).
enum CloneOp {
    /// Allocate + byte-copy a fresh block of type `ord` from `src_off`, then walk its
    /// fields to clone owned sub-structure and record shared-target bumps.
    Block { src_off: u64, ord: RttiOrdinal },
    /// Walk an inline `#[embed]` region's fields (no allocation — its bytes are part
    /// of the already-copied parent block), fixing up owned grandchildren.
    Inline {
        src_base: u64,
        new_base: u64,
        ord: RttiOrdinal,
    },
    /// Interpret one shape given its source and (already-copied) destination offsets.
    Field {
        shape: Shape,
        src_off: u64,
        new_off: u64,
    },
}

/// The accumulating state of one deep clone (see [`RttiRegistry::clone_value`]).
#[derive(Default)]
struct CloneState {
    /// Decoded-type cache, shared with [`RttiRegistry::shape_stride`].
    cache: HashMap<RttiOrdinal, RttiType>,
    /// `source block offset → fresh clone offset`, for repointing owned children.
    map: HashMap<u64, u64>,
    /// Deferred child-pointer patches: `(new slot offset, source child offset)` — the
    /// slot is set to `map[source child]` once every block has been cloned.
    patches: Vec<(u64, u64)>,
    /// Refcount counters to bump (shared `strong` / `weak` targets), applied last.
    bumps: Vec<u64>,
    /// Every freshly allocated range, so a failed clone frees its orphans.
    allocated: Vec<BStackRange>,
    /// The in-flight intention-first WAL transaction: when the allocator
    /// names a WAL anchor, each `alloc_copy` block is logged `Pending` before it is
    /// used, so a **crash** mid-clone is reclaimed by [`wal::finish`](crate::wal::finish)
    /// on the next open (the in-process error path already frees `allocated`). `None`
    /// when the allocator opts out of reclamation or nothing has been allocated yet.
    wal: Option<CloneWal>,
}

/// The in-flight intention-first WAL transaction of a `clone_value` walk — the file's
/// WAL lock held for the descent, plus the persistent block's offset, entry-slot
/// capacity, and how many entries have been published. Mirrors `clone::CloneWal`.
struct CloneWal {
    /// Holds the file's WAL lock until the clone completes / errors.
    _held: HeldLock,
    /// Offset of the persistent WAL block (moves if a grow reallocates it).
    block_off: u64,
    /// Entry slots the block currently has.
    capacity: u64,
    /// Entries published so far (== `CloneState::allocated.len()`).
    logged: u64,
}

/// Bytes of a `VecDesc` (`data_off:u64` @0, `data_size:u64` @8) — the inline
/// descriptor of a persistent vector.
const VECDESC_LEN: u64 = 16;
/// A byte-vec data block's header (`len:u64` @0, `cap:u64` @8, elements from 16).
const BYTEVEC_HEADER: u64 = 16;
/// Bytes of a `WidePtr` on the wire.
const FOREIGN_REPR_LEN: u64 = 16;
/// Offset of the `tag: EightCC` within a block's `BlockHeader` (`size: u64` @0).
const HEADER_TAG_OFFSET: u64 = 8;
/// Bytes of an `(rc, weak)` control block (`XOnDiskRef`): a 16-byte header, then the
/// `strong`, `weak`, and data-back-pointer `u64`s. Fixed for every weakable type (the
/// control layout does not depend on `T`).
const CONTROL_SIZE: u64 = CTRL_DATA_OFFSET + 8;

impl RttiRegistry {
    /// Read a structure of type `ordinal` at `block_off` in `data` into a [`Value`]
    /// tree — the core RTTI operation: interpret an on-disk structure with no
    /// compiled-in Rust type.
    ///
    /// The walk is **non-recursive** (an explicit `work` stack), so arbitrarily deep
    /// or self-referential data cannot blow the call stack. It **follows** owning
    /// edges (`owned` / `strong` / `embed`) into child blocks, and **stops** at
    /// non-owning ones (`weak` / `ref` / `foreign`), recording just their offset —
    /// which also breaks any reference cycle. A node budget guards against a corrupt
    /// file describing an unterminated walk.
    pub fn read_value(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
    ) -> io::Result<Value> {
        self.run_read(
            data,
            vec![Op::Block {
                ord: ordinal,
                block_off,
            }],
        )
    }

    /// The read machine: run a `work` stack of [`Op`]s to a single [`Value`]. Seeded
    /// with a `Block` op by [`read_value`](Self::read_value) (a whole block) or a
    /// `Shape` op by [`get`](Self::get) (one field).
    fn run_read(&self, data: &BStack, initial: Vec<Op>) -> io::Result<Value> {
        let mut cache: HashMap<RttiOrdinal, RttiType> = HashMap::new();
        let mut work: Vec<Op> = initial;
        let mut results: Vec<Value> = Vec::new();
        // Bounds the total nodes visited: a corrupt schema/data pair (or a strong
        // cycle) can otherwise loop forever.
        let mut budget: u64 = 4_000_000;

        while let Some(op) = work.pop() {
            budget = budget.checked_sub(1).ok_or_else(|| {
                corrupt("[BSTACK0807] RTTI interpret budget exceeded (corrupt data or a cycle?)")
            })?;
            match op {
                Op::Block { ord, block_off } => {
                    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ord) {
                        e.insert(self.load_type(ord)?);
                    }
                    let ty = &cache[&ord];
                    match &ty.body {
                        RttiBody::Struct(fields) => {
                            let names = fields.iter().map(|f| f.name.clone()).collect();
                            // Assemble marker first (popped last), then fields in
                            // order (so they pop child-first into the marker).
                            let field_ops: Vec<Op> = fields
                                .iter()
                                .map(|f| -> io::Result<Op> {
                                    Ok(Op::Shape {
                                        shape: f.shape.clone(),
                                        offset: add_off(block_off, f.offset as u64)?,
                                    })
                                })
                                .collect::<io::Result<Vec<Op>>>()?;
                            work.push(Op::MakeBlock { tag: ty.tag, names });
                            work.extend(field_ops);
                        }
                        RttiBody::Enum(e) => {
                            let raw = read_disc(
                                data,
                                add_off(block_off, e.disc_off as u64)?,
                                e.disc_width,
                            )?;
                            let mask = disc_mask(e.disc_width);
                            let variant = e
                                .variants
                                .iter()
                                .find(|v| (v.disc_value as u64) & mask == raw)
                                .ok_or_else(|| {
                                    corrupt(format!(
                                        "[BSTACK0808] no RTTI variant for discriminant {raw}"
                                    ))
                                })?;
                            let names = variant.fields.iter().map(|f| f.name.clone()).collect();
                            let payload_base = add_off(block_off, e.payload_off as u64)?;
                            let field_ops: Vec<Op> = variant
                                .fields
                                .iter()
                                .map(|f| -> io::Result<Op> {
                                    Ok(Op::Shape {
                                        shape: f.shape.clone(),
                                        offset: add_off(payload_base, f.offset as u64)?,
                                    })
                                })
                                .collect::<io::Result<Vec<Op>>>()?;
                            work.push(Op::MakeEnum {
                                tag: ty.tag,
                                variant: variant.name.clone(),
                                names,
                            });
                            work.extend(field_ops);
                        }
                    }
                }

                Op::Shape { shape, offset } => match shape {
                    Shape::Pod { width } => {
                        // `width` is an untrusted record field; bound it against the
                        // stack before sizing an allocation with it (the read after
                        // would fail anyway — this fails first, without the alloc).
                        if width as u64 > data.len()?.saturating_sub(offset) {
                            return Err(corrupt(
                                "[BSTACK0800] RTTI POD width runs past the end of the data stack",
                            ));
                        }
                        let mut buf = vec![0u8; width as usize];
                        data.get_into(offset, &mut buf)?;
                        results.push(Value::Pod(buf.into()));
                    }
                    Shape::Class { value, .. } => {
                        // A class variable's value is schema-side, not per-instance.
                        results.push(Value::Class(value));
                    }
                    Shape::Owned(tag) | Shape::Strong(tag) => {
                        let child = read_u64_at(data, offset)?;
                        if child == 0 {
                            results.push(Value::Null);
                        } else {
                            let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
                            work.push(Op::Block {
                                ord,
                                block_off: child,
                            });
                        }
                    }
                    Shape::Embed(tag) => {
                        // The child's whole OnDisk is inline at this slot (no pointer).
                        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
                        work.push(Op::Block {
                            ord,
                            block_off: offset,
                        });
                    }
                    Shape::Weak(tag) | Shape::Ref(tag) => {
                        results.push(Value::Ref {
                            tag,
                            offset: read_u64_at(data, offset)?,
                        });
                    }
                    Shape::Foreign { tag, kind } => {
                        // WidePtr { file_id:u32 @0, type_index:u32 @4, offset:u64 @8 }.
                        // The target is in another file — recorded, not followed.
                        let __wp = WidePtr::read_from_stack(data, offset)?;
                        let (file_id, off) = (__wp.file_id(), __wp.offset().get());
                        results.push(Value::Foreign {
                            tag,
                            kind,
                            file_id,
                            offset: off,
                        });
                    }
                    Shape::Option(inner) => {
                        // Niche location depends on the inner shape (a `Foreign`'s is
                        // its offset word @8, not the leading word).
                        if option_present(data, &inner, offset)? {
                            work.push(Op::MakeSome);
                            work.push(Op::Shape {
                                shape: *inner,
                                offset,
                            });
                        } else {
                            results.push(Value::Null);
                        }
                    }
                    Shape::Array { n, inner } => {
                        // Charge the budget for all elements up front, as the `Vec`
                        // arm does — `n` comes off an untrusted record, and the ops
                        // are materialized eagerly, so an absurd count must fail
                        // cleanly rather than pre-allocate past the budget.
                        budget = budget.checked_sub(n as u64).ok_or_else(budget_exceeded)?;
                        let stride = self.shape_stride(&inner, &mut cache)?;
                        let elem_ops: Vec<Op> = (0..n as u64)
                            .map(|i| -> io::Result<Op> {
                                Ok(Op::Shape {
                                    shape: (*inner).clone(),
                                    offset: add_off(offset, mul_off(i, stride)?)?,
                                })
                            })
                            .collect::<io::Result<Vec<Op>>>()?;
                        work.push(Op::MakeArray(n as usize));
                        work.extend(elem_ops);
                    }
                    Shape::Vec(inner) => {
                        let data_off = read_u64_at(data, offset)?; // VecDesc.data_off @0
                        if data_off == 0 {
                            results.push(Value::Vec(Box::default()));
                        } else {
                            // `@0` is the byte length, validated against the block size
                            // (`VecDesc.data_size` @8) so a forged length can't drive an
                            // out-of-block read / petabyte allocation.
                            let data_size = read_u64_at(data, add_off(offset, 8)?)?;
                            let base = add_off(data_off, BYTEVEC_HEADER)?;
                            let stride = self.shape_stride(&inner, &mut cache)?;
                            let byte_len = read_u64_at(data, data_off)?;
                            let len = checked_vec_len(byte_len, data_size, stride)?;
                            // Charge the budget for all elements up front — the ops are
                            // materialized eagerly, so a huge (but in-block) length must
                            // fail cleanly rather than pre-allocate past the budget.
                            budget = budget.checked_sub(len).ok_or_else(budget_exceeded)?;
                            let elem_ops: Vec<Op> = (0..len)
                                .map(|i| -> io::Result<Op> {
                                    Ok(Op::Shape {
                                        shape: (*inner).clone(),
                                        offset: add_off(base, mul_off(i, stride)?)?,
                                    })
                                })
                                .collect::<io::Result<Vec<Op>>>()?;
                            work.push(Op::MakeVec(len as usize));
                            work.extend(elem_ops);
                        }
                    }
                    Shape::Tuple(items) => {
                        let mut elem_ops: Vec<Op> = Vec::with_capacity(items.len());
                        let mut off = offset;
                        for it in &items {
                            elem_ops.push(Op::Shape {
                                shape: it.clone(),
                                offset: off,
                            });
                            off = add_off(off, self.shape_stride(it, &mut cache)?)?;
                        }
                        work.push(Op::MakeTuple(items.len()));
                        work.extend(elem_ops);
                    }
                },

                Op::MakeBlock { tag, names } => {
                    let fields = pop_named(&mut results, &names)?;
                    results.push(Value::Block {
                        tag,
                        fields: fields.into(),
                    });
                }
                Op::MakeEnum {
                    tag,
                    variant,
                    names,
                } => {
                    let fields = pop_named(&mut results, &names)?;
                    results.push(Value::Enum {
                        tag,
                        variant,
                        fields: fields.into(),
                    });
                }
                Op::MakeArray(n) => {
                    let v = pop_n(&mut results, n)?;
                    results.push(Value::Array(v.into()));
                }
                Op::MakeVec(n) => {
                    let v = pop_n(&mut results, n)?;
                    results.push(Value::Vec(v.into()));
                }
                Op::MakeTuple(n) => {
                    let v = pop_n(&mut results, n)?;
                    results.push(Value::Tuple(v.into()));
                }
                Op::MakeSome => {
                    let inner = results
                        .pop()
                        .ok_or_else(|| corrupt("[BSTACK0809] RTTI interpret stack underflow"))?;
                    results.push(Value::Some(Box::new(inner)));
                }
            }
        }

        match results.len() {
            1 => Ok(results.pop().unwrap()),
            n => Err(corrupt(format!(
                "[BSTACK0809] RTTI interpret produced {n} values (expected 1)"
            ))),
        }
    }

    /// Read the structure a typed [`WidePtr`] points at, within `data`. The
    /// pointer must be **typed** (carry an RTTI ordinal), and `data` must be the file
    /// it targets. This resolves within a single file by design — for a cross-file
    /// pointer, resolve its `file_id` through the [`registry`](crate::registry) first
    /// and call this against that file's stack.
    pub fn read_ptr(&self, data: &BStack, ptr: WidePtr) -> io::Result<Value> {
        let ord = self.resolve_ptr(ptr).ok_or_else(|| {
            corrupt("[BSTACK080A] cannot read an untyped / out-of-range RTTI pointer")
        })?;
        self.read_value(data, ord, ptr.offset().get())
    }

    /// The runtime-typed [`AnyRef`] a **typed** pointer denotes — its registry tag
    /// (resolved from the pointer's `type_index`) plus offset. `None` for an untyped
    /// (`type_index == 0`) or out-of-range pointer, so a stray pointer can never
    /// masquerade as a registered type. Downcast the result with
    /// [`AnyRef::downcast`], or read it generically with [`read_any`](Self::read_any).
    pub fn any_ref(&self, ptr: WidePtr) -> Option<AnyRef> {
        let ord = self.resolve_ptr(ptr)?;
        let tag = self.tag_of(ord)?;
        // SAFETY: the tag is registry-authoritative for the pointer's type_index,
        // and the offset is the typed pointer's own target.
        Some(unsafe { AnyRef::new(tag, ptr.offset().get()) })
    }

    /// Interpret the structure an [`AnyRef`] points at into a [`Value`] tree — the
    /// generic fallback when [`AnyRef::downcast`] does not match a compiled-in type.
    /// Errors if the reference's tag is not a registered type.
    pub fn read_any(&self, data: &BStack, any: &AnyRef) -> io::Result<Value> {
        let ord = self.ordinal_of(any.tag()).ok_or_else(unknown_tag)?;
        self.read_value(data, ord, any.offset())
    }

    /// Tear down (free) the structure of type `ordinal` at `block_off` in `alloc`'s
    /// file — the interpreted equivalent of a generated `bstack_drop`.
    ///
    /// # Safety
    ///
    /// `block_off` must name a live block of type `ordinal` that the caller owns,
    /// and the root **must already be detached** (unlinked from any parent): this
    /// frees it unconditionally, so a still-linked root leaves its parent pointing
    /// at freed storage and a wrong offset frees ranges the caller does not own —
    /// the same obligation that makes [`BStackBlock::from_range`] `unsafe`, reached
    /// with a bare integer instead of a fabricated handle.
    ///
    /// The walk
    /// is **non-recursive**; it collects every block to reclaim then frees them all in
    /// one [`free_many`](BStackRaiiAllocator::free_many) (bulk when the allocator
    /// supports it, else sequential — orphan-only on a crash, never a torn structure).
    ///
    /// Per RAII kind: `owned` / `embed` recurse-free the child subtree; a **`strong`**
    /// reference decrements the target's refcount (inline for `rc`, or the control
    /// block's `strong` for `rc, weak`) and frees the data (and, when the phantom weak
    /// then hits zero, the control) block only when the **last** owner drops; a
    /// **`weak`** reference decrements the control block's `weak` and frees the control
    /// block alone when last; `ref` is non-owning and left alone; `pod` / class own
    /// nothing. Vectors free their data block plus any owning/shared element blocks.
    /// Cross-file `foreign` references (scalar or in a container) are torn down in the
    /// target's own file through the registry; a detached target file leaks.
    pub unsafe fn teardown<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        ordinal: RttiOrdinal,
        block_off: u64,
    ) -> io::Result<()> {
        // Bound cross-file recursion (each `Foreign` hop recurses here natively).
        let _depth = DepthGuard::enter()?;
        let data = alloc.stack();
        let mut cache: HashMap<RttiOrdinal, RttiType> = HashMap::new();
        let mut work: Vec<TdOp> = vec![TdOp::Block {
            ord: ordinal,
            block_off,
            emit: true,
        }];
        let mut to_free: Vec<BStackRange> = Vec::new();
        // Destructive side effects are **collected** during the (read-only) walk and
        // applied only after it completes, so a mid-walk error (unknown tag, budget,
        // corrupt discriminant, a failed read) leaves the structure — and every shared
        // refcount / cross-file target — untouched, and a retry re-does nothing.
        // `to_free` is the home-file owned ranges; these are the shared / cross-file
        // mutations that a mid-walk abort must not have started.
        let mut strong_releases: Vec<(EightCC, u64)> = Vec::new(); // (tag, data offset)
        let mut weak_releases: Vec<u64> = Vec::new(); // control offsets
        let mut foreign_releases: Vec<(EightCC, OwnershipKind, u64, u64)> = Vec::new();
        let mut budget: u64 = 4_000_000;

        while let Some(op) = work.pop() {
            budget = budget.checked_sub(1).ok_or_else(|| {
                corrupt("[BSTACK0807] RTTI teardown budget exceeded (corrupt data or a cycle?)")
            })?;
            match op {
                TdOp::Block {
                    ord,
                    block_off,
                    emit,
                } => {
                    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ord) {
                        e.insert(self.load_type(ord)?);
                    }
                    let ty = &cache[&ord];
                    // An embedded child (`emit == false`) has no block of its own — its
                    // storage is part of the parent — but its owned sub-blocks are still
                    // freed by walking its fields.
                    if emit {
                        to_free.push(BStackRange::new(block_off, ty.ondisk_size));
                    }
                    match &ty.body {
                        RttiBody::Struct(fields) => {
                            for f in fields {
                                work.push(TdOp::Shape {
                                    shape: f.shape.clone(),
                                    offset: add_off(block_off, f.offset as u64)?,
                                });
                            }
                        }
                        RttiBody::Enum(e) => {
                            let raw = read_disc(
                                data,
                                add_off(block_off, e.disc_off as u64)?,
                                e.disc_width,
                            )?;
                            let mask = disc_mask(e.disc_width);
                            let variant = e
                                .variants
                                .iter()
                                .find(|v| (v.disc_value as u64) & mask == raw)
                                .ok_or_else(|| {
                                    corrupt(format!(
                                        "[BSTACK0808] no RTTI variant for discriminant {raw}"
                                    ))
                                })?;
                            let payload_base = add_off(block_off, e.payload_off as u64)?;
                            for f in &variant.fields {
                                work.push(TdOp::Shape {
                                    shape: f.shape.clone(),
                                    offset: add_off(payload_base, f.offset as u64)?,
                                });
                            }
                        }
                    }
                }

                TdOp::Shape { shape, offset } => match shape {
                    // Nothing to free: inline bytes, an alias, or a schema-side value.
                    Shape::Pod { .. } | Shape::Ref(_) | Shape::Class { .. } => {}
                    Shape::Owned(tag) => {
                        let child = read_u64_at(data, offset)?;
                        if child != 0 {
                            work.push(TdOp::Block {
                                ord: self.ordinal_of(tag).ok_or_else(unknown_tag)?,
                                block_off: child,
                                emit: true,
                            });
                        }
                    }
                    Shape::Embed(tag) => {
                        // Inline child: walk its fields (freeing its sub-blocks) but do
                        // not free the slot itself — it is part of this block.
                        work.push(TdOp::Block {
                            ord: self.ordinal_of(tag).ok_or_else(unknown_tag)?,
                            block_off: offset,
                            emit: false,
                        });
                    }
                    Shape::Strong(tag) => {
                        let data_off = read_u64_at(data, offset)?;
                        if data_off != 0 {
                            strong_releases.push((tag, data_off));
                        }
                    }
                    Shape::Weak(_) => {
                        // A weak field's slot holds the *control* offset directly.
                        let ctrl_off = read_u64_at(data, offset)?;
                        if ctrl_off != 0 {
                            weak_releases.push(ctrl_off);
                        }
                    }
                    Shape::Foreign { tag, kind } => {
                        // Cross-file: the target's file + offset, torn down in the commit
                        // phase (a self-contained transaction on that file).
                        let __wp = WidePtr::read_from_stack(data, offset)?;
                        let (file_id, off) = (__wp.file_id(), __wp.offset().get());
                        foreign_releases.push((tag, kind, file_id, off));
                    }
                    Shape::Option(inner) => {
                        if option_present(data, &inner, offset)? {
                            work.push(TdOp::Shape {
                                shape: *inner,
                                offset,
                            });
                        }
                    }
                    Shape::Array { n, inner } => {
                        // Charge for all elements up front — `n` is untrusted and
                        // the ops are materialized eagerly (see the read walk).
                        budget = budget.checked_sub(n as u64).ok_or_else(|| {
                            corrupt(
                                "[BSTACK0807] RTTI teardown budget exceeded (corrupt data or a cycle?)",
                            )
                        })?;
                        let stride = self.shape_stride(&inner, &mut cache)?;
                        for i in 0..n as u64 {
                            work.push(TdOp::Shape {
                                shape: (*inner).clone(),
                                offset: add_off(offset, mul_off(i, stride)?)?,
                            });
                        }
                    }
                    Shape::Tuple(items) => {
                        let mut off = offset;
                        for it in &items {
                            work.push(TdOp::Shape {
                                shape: it.clone(),
                                offset: off,
                            });
                            off = add_off(off, self.shape_stride(it, &mut cache)?)?;
                        }
                    }
                    Shape::Vec(inner) => {
                        let data_off = read_u64_at(data, offset)?; // VecDesc.data_off @0
                        if data_off != 0 {
                            let data_size = read_u64_at(data, add_off(offset, 8)?)?; // .data_size @8
                            // A vector of owning/shared elements releases each element
                            // from the data block's element area too. The `@0` word is
                            // the byte length, so the count is `byte_len / stride`
                            // (stride = 8 for a `u64` offset, 16 for a `WidePtr`).
                            let base = add_off(data_off, BYTEVEC_HEADER)?;
                            let stride = self.shape_stride(&inner, &mut cache)?;
                            let byte_len = read_u64_at(data, data_off)?;
                            let len = checked_vec_len(byte_len, data_size, stride)?;
                            match &*inner {
                                Shape::Owned(tag) => {
                                    let ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                                    for i in 0..len {
                                        let e =
                                            read_u64_at(data, add_off(base, mul_off(i, stride)?)?)?;
                                        if e != 0 {
                                            work.push(TdOp::Block {
                                                ord,
                                                block_off: e,
                                                emit: true,
                                            });
                                        }
                                    }
                                }
                                Shape::Strong(tag) => {
                                    for i in 0..len {
                                        let e =
                                            read_u64_at(data, add_off(base, mul_off(i, stride)?)?)?;
                                        if e != 0 {
                                            strong_releases.push((*tag, e));
                                        }
                                    }
                                }
                                Shape::Weak(_) => {
                                    for i in 0..len {
                                        let c =
                                            read_u64_at(data, add_off(base, mul_off(i, stride)?)?)?;
                                        if c != 0 {
                                            weak_releases.push(c);
                                        }
                                    }
                                }
                                // A vector of `Foreign` pointers: each element is a
                                // 16-byte `WidePtr`; its target is torn down in its
                                // own file in the commit phase (a null offset is a no-op).
                                other if foreign_leaf(other).is_some() => {
                                    let (tag, kind) = foreign_leaf(other).unwrap();
                                    for i in 0..len {
                                        let __wp = WidePtr::read_from_stack(
                                            data,
                                            add_off(base, mul_off(i, stride)?)?,
                                        )?;
                                        let (file_id, foff) = (__wp.file_id(), __wp.offset().get());
                                        foreign_releases.push((tag, kind, file_id, foff));
                                    }
                                }
                                // POD / `ref` elements own no sub-blocks.
                                _ => {}
                            }
                            to_free.push(BStackRange::new(data_off, data_size));
                        }
                    }
                },
            }
        }

        // --- Commit phase (the walk completed without error) ---
        // Doing these only now means a walk that failed validation (unknown tag, budget,
        // corrupt discriminant, a bad read) left no refcount decremented and no foreign
        // target freed — so a retry decrements / frees each exactly once, not twice.
        //
        // Shared / cross-file releases run **before** the home `free_many`: a `Foreign`
        // (or `strong`) that recurses — e.g. a cycle — then trips the depth guard while
        // the home block is still present, so nothing is freed on that error rather than
        // the root being freed and then double-freed through the cycle edge.
        for (tag, data_off) in strong_releases {
            self.commit_strong_release(alloc, tag, data_off)?;
        }
        for ctrl_off in weak_releases {
            commit_weak_release(alloc, ctrl_off)?;
        }
        for (tag, kind, file_id, off) in foreign_releases {
            self.teardown_foreign(alloc, tag, kind, file_id, off)?;
        }
        // Free children before parents (post-order): the walk collects ranges
        // pre-order (a block before its sub-blocks — line above), so reverse. For a
        // well-formed structure the order is immaterial (the sub-blocks are separate
        // allocations). It matters only for a *forged* owned pointer into a parent's
        // own interior (installable only via `unsafe`):
        // freeing the parent *first* leaves the interior slice sitting inside a freed
        // region, and applying it then writes a bogus free-list node (an actual
        // corruption). Child-first keeps every free consistent — the forged interior
        // just double-frees within the parent, which the allocator merges (or a
        // debug allocator flags) rather than corrupting.
        to_free.reverse();
        // Route through the WAL (or the allocator's atomic bulk free) so a crash
        // mid-free is reclaimed on the next open, matching the static teardown
        // rather than leaking permanently.
        // SAFETY: every range in `to_free` was collected by the walk from
        // owned slots of the structure being torn down, in this file.
        unsafe { crate::teardown::commit_home_frees(alloc, to_free) }
    }

    /// Release one deferred `strong` reference (commit phase of [`teardown`](Self::teardown)):
    /// decrement the target's strong count in its (home) file and, only if it was the last
    /// owner, tear the data subtree down (its own transaction) and release the phantom weak,
    /// freeing the control block if no real weak handles remain. The target's `weak` flag
    /// selects the inline-refcount vs control-block path.
    fn commit_strong_release<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        tag: EightCC,
        data_off: u64,
    ) -> io::Result<()> {
        let data = alloc.stack();
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        if self.load_type(ord)?.weak {
            let ctrl_off = read_u64_at(data, add_off(data_off, CTRL_BACKPTR_OFFSET)?)?;
            if refcount::fetch_sub(data, add_off(ctrl_off, CTRL_STRONG_OFFSET)?, 1)? == 1 {
                // SAFETY: this caller was the last strong owner (the fetch_sub hit
                // zero), and `data_off` came from the owning slot being released.
                unsafe { self.teardown(alloc, ord, data_off)? };
                if refcount::fetch_sub(data, add_off(ctrl_off, CTRL_WEAK_OFFSET)?, 1)? == 1 {
                    // SAFETY: last weak released — the control block is unreferenced.
                    unsafe { alloc.free_many([BStackRange::new(ctrl_off, CONTROL_SIZE)])? };
                }
            }
        } else if refcount::fetch_sub(data, add_off(data_off, RC_REFCOUNT_OFFSET)?, 1)? == 1 {
            // SAFETY: as above — sole owner, slot-derived offset.
            unsafe { self.teardown(alloc, ord, data_off)? };
        }
        Ok(())
    }

    /// Tear down a `Foreign` reference's target **in the target's own file**. `SELF`
    /// (`file_id == 0`) resolves against `home`; a registered file is reached through
    /// its [`ForeignHost`](crate::registry::ForeignHost) — a detached / unknown file
    /// leaks (the design permits it) rather than erroring. `offset == 0` (null) is a
    /// no-op.
    fn teardown_foreign<A: BStackRaiiAllocator>(
        &self,
        home: &A,
        tag: EightCC,
        kind: OwnershipKind,
        file_id: u64,
        offset: u64,
    ) -> io::Result<()> {
        if offset == 0 || matches!(kind, OwnershipKind::Ref) {
            return Ok(()); // null, or a non-owning alias
        }
        if file_id == 0 {
            self.teardown_foreign_in(home, tag, kind, offset)
        } else {
            let Some(fid) = FileId::from_u64(file_id) else {
                return Ok(());
            };
            match registry::host_arc(fid) {
                Some(host) => {
                    let alloc = ForeignHostAllocator::new(host, fid);
                    self.teardown_foreign_in(&alloc, tag, kind, offset)
                }
                // File not attached / detached → leak (never a premature free).
                None => Ok(()),
            }
        }
    }

    /// The per-kind foreign teardown against an already-resolved `target` allocator:
    /// `owned` recurses a full teardown; `strong` / `weak` decrement (in the target
    /// file) and free only the last owner. `offset` is the target's data offset for
    /// `owned` / `strong`, its control offset for `weak`.
    fn teardown_foreign_in<A: BStackRaiiAllocator>(
        &self,
        target: &A,
        tag: EightCC,
        kind: OwnershipKind,
        offset: u64,
    ) -> io::Result<()> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        let data = target.stack();
        match kind {
            OwnershipKind::Ref => Ok(()),
            // SAFETY: `offset` came from the owning foreign slot being torn down;
            // ownership of the target transfers with the slot.
            OwnershipKind::Owned => unsafe { self.teardown(target, ord, offset) },
            OwnershipKind::Strong => {
                if self.load_type(ord)?.weak {
                    let ctrl = read_u64_at(data, add_off(offset, CTRL_BACKPTR_OFFSET)?)?;
                    if refcount::fetch_sub(data, add_off(ctrl, CTRL_STRONG_OFFSET)?, 1)? == 1 {
                        // SAFETY: last strong owner; slot-derived offset.
                        unsafe { self.teardown(target, ord, offset)? };
                        if refcount::fetch_sub(data, add_off(ctrl, CTRL_WEAK_OFFSET)?, 1)? == 1 {
                            // SAFETY: last weak released — control block unreferenced.
                            unsafe { target.free_many([BStackRange::new(ctrl, CONTROL_SIZE)])? };
                        }
                    }
                } else if refcount::fetch_sub(data, add_off(offset, RC_REFCOUNT_OFFSET)?, 1)? == 1 {
                    // SAFETY: last strong owner; slot-derived offset.
                    unsafe { self.teardown(target, ord, offset)? };
                }
                Ok(())
            }
            OwnershipKind::Weak => {
                // A weak foreign's offset is the control offset.
                if refcount::fetch_sub(data, add_off(offset, CTRL_WEAK_OFFSET)?, 1)? == 1 {
                    // SAFETY: last weak released — control block unreferenced.
                    unsafe { target.free_many([BStackRange::new(offset, CONTROL_SIZE)])? };
                }
                Ok(())
            }
        }
    }

    /// Clone a `Foreign` reference across the file boundary. The clone's slot was
    /// already byte-copied (so a `ref` alias and a null are done); an `owned` target
    /// is **deep-copied in its own file** and the copied slot repointed at the new
    /// offset; a `strong` / `weak` bumps the target's count in its own file. `SELF`
    /// resolves against `home`; a detached target file **errors** (fail-safe — never
    /// alias an owner, which would double-free later — matching the generated path).
    ///
    /// `src_off` / `new_off` are the source / clone `WidePtr` slot locations in the
    /// home file.
    fn clone_foreign<A: BStackRaiiAllocator>(
        &self,
        home: &A,
        home_data: &BStack,
        tag: EightCC,
        kind: OwnershipKind,
        src_off: u64,
        new_off: u64,
    ) -> io::Result<()> {
        if matches!(kind, OwnershipKind::Ref) {
            return Ok(()); // aliased — the copied slot is correct
        }
        let __wp = WidePtr::read_from_stack(home_data, src_off)?;
        let (file_id, src_target) = (__wp.file_id(), __wp.offset().get());
        if src_target == 0 {
            return Ok(()); // null — copied as 0
        }
        if file_id == 0 {
            self.clone_foreign_in(home, home_data, tag, kind, src_target, new_off)
        } else {
            let fid = FileId::from_u64(file_id).ok_or_else(clone_unsupported)?;
            let host = registry::host_arc(fid).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "[BSTACK080F] RTTI clone: the foreign target's file is detached / not attached",
                )
            })?;
            let alloc = ForeignHostAllocator::new(host, fid);
            self.clone_foreign_in(&alloc, home_data, tag, kind, src_target, new_off)
        }
    }

    /// The per-kind foreign clone against an already-resolved `target` allocator. For
    /// `owned` it patches the clone's `WidePtr.offset` (@ `new_off + 8`, in the
    /// home file) to the freshly-cloned target offset.
    fn clone_foreign_in<A: BStackRaiiAllocator>(
        &self,
        target: &A,
        home_data: &BStack,
        tag: EightCC,
        kind: OwnershipKind,
        src_target: u64,
        new_off: u64,
    ) -> io::Result<()> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        let tstack = target.stack();
        match kind {
            OwnershipKind::Ref => Ok(()),
            OwnershipKind::Owned => {
                // SAFETY: `src_target` came from the source's validated foreign slot.
                let new_target = unsafe { self.clone_value(target, ord, src_target)? };
                // Repoint only the address word of the copied WidePtr.
                home_data.set(
                    add_off(new_off, FOREIGN_REPR_LEN - 8)?,
                    new_target.to_le_bytes(),
                )
            }
            OwnershipKind::Strong => {
                let off = if self.load_type(ord)?.weak {
                    let ctrl = read_u64_at(tstack, add_off(src_target, CTRL_BACKPTR_OFFSET)?)?;
                    add_off(ctrl, CTRL_STRONG_OFFSET)?
                } else {
                    add_off(src_target, RC_REFCOUNT_OFFSET)?
                };
                refcount::fetch_add(tstack, off, 1)?;
                Ok(())
            }
            OwnershipKind::Weak => {
                refcount::fetch_add(tstack, add_off(src_target, CTRL_WEAK_OFFSET)?, 1)?;
                Ok(())
            }
        }
    }

    /// Resolve a **field path** (`["outer", "inner", …]`) from the root of type
    /// `ordinal` at `block_off` to its target field's absolute offset and shape.
    ///
    /// Navigation descends through **block references** — `owned` / `strong` / `ref`
    /// (follow the stored offset into the child) and `embed` (inline, same offset
    /// base) — and through a struct's fields or an enum's active variant. A `pod` /
    /// `vec` / array / tuple / `weak` / `foreign` field is a leaf: it may be the last
    /// segment, but the path cannot continue *through* it. An empty path, an unknown
    /// field, or a null reference on the way is an error.
    fn resolve_field(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
        path: &[&str],
    ) -> io::Result<Resolved> {
        if path.is_empty() {
            return Err(set_error("empty field path"));
        }
        let mut ord = ordinal;
        let mut base = block_off;
        for (i, seg) in path.iter().enumerate() {
            let ty = self.load_type(ord)?;
            // A struct's fields are block-relative; an enum's active variant's fields
            // are payload-relative.
            let (fields, field_base): (&[RttiField], u64) = match &ty.body {
                RttiBody::Struct(f) => (f, base),
                RttiBody::Enum(e) => {
                    let raw = read_disc(data, add_off(base, e.disc_off as u64)?, e.disc_width)?;
                    let mask = disc_mask(e.disc_width);
                    let variant = e
                        .variants
                        .iter()
                        .find(|v| (v.disc_value as u64) & mask == raw)
                        .ok_or_else(|| set_error(format!("no variant for discriminant {raw}")))?;
                    (&variant.fields, add_off(base, e.payload_off as u64)?)
                }
            };
            let field = fields
                .iter()
                .find(|f| &f.name == seg)
                .ok_or_else(|| set_error(format!("no field named `{seg}`")))?;
            let field_off = add_off(field_base, field.offset as u64)?;

            if i + 1 == path.len() {
                // A `#[bstack_static]` class variable lives in the schema record for
                // *this type*, not in the instance — resolve it by (type tag, name),
                // not an instance offset.
                if matches!(field.shape, Shape::Class { .. }) {
                    return Ok(Resolved::Class {
                        tag: ty.tag,
                        name: field.name.clone(),
                    });
                }
                return Ok(Resolved::Instance {
                    offset: field_off,
                    shape: field.shape.clone(),
                });
            }
            // Descend into a block reference for the next segment.
            match &field.shape {
                Shape::Owned(tag) | Shape::Strong(tag) | Shape::Ref(tag) => {
                    let child = read_u64_at(data, field_off)?;
                    if child == 0 {
                        return Err(set_error(format!("null reference at `{seg}`")));
                    }
                    ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                    base = child;
                }
                Shape::Embed(tag) => {
                    ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                    base = field_off;
                }
                _ => {
                    return Err(set_error(format!(
                        "cannot descend through non-block field `{seg}`"
                    )));
                }
            }
        }
        unreachable!("the last segment returns inside the loop")
    }

    /// Read a single field named by `path` into a [`Value`] — a targeted `get`
    /// (`read_value` scoped to one field). Follows an owning reference into its child,
    /// exactly as a full read would. A `#[bstack_static]` class variable at the path
    /// yields its **live** value ([`Value::Class`]), read from the schema.
    pub fn get(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
        path: &[&str],
    ) -> io::Result<Value> {
        match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => {
                self.run_read(data, vec![Op::Shape { shape, offset }])
            }
            Resolved::Class { tag, name } => Ok(Value::Class(self.class_value(tag, &name)?.into())),
        }
    }

    /// Overwrite the field named by `path` with `value` — the interpreted
    /// `set_<field>`, one atomic write, reaching any depth.
    ///
    /// A **POD** field takes its exact-width bytes; a **`ref`** field an 8-byte target
    /// offset (a non-owning alias); a **`#[bstack_static]` mutable class variable**
    /// routes to [`set_class_value`](Self::set_class_value) (written in the schema, not
    /// the instance). An `owned` / `strong` / `weak` field is *replaced*, not
    /// overwritten — that is [`swap`](Self::swap). Errors on any other target or a
    /// wrong-width value.
    ///
    /// # Safety
    ///
    /// `block_off` must name a live block of type `ordinal`. The resolved write is
    /// raw: with a wrong `block_off` the bytes land at an arbitrary in-file
    /// location, and even with a right one a caller-chosen POD image overwrites
    /// whatever invariant-bearing bytes the field holds (e.g. an inline `VecDesc`,
    /// whose forged `data_off` a later safe drop would free).
    pub unsafe fn set(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
        path: &[&str],
        value: &[u8],
    ) -> io::Result<()> {
        let (offset, shape) = match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => (offset, shape),
            // A class variable is schema-side; write it in place there.
            Resolved::Class { tag, name } => return self.set_class_value(tag, &name, value),
        };
        match shape {
            Shape::Pod { width } => {
                if value.len() != width as usize {
                    return Err(set_error(format!(
                        "field is {width} bytes, got {}",
                        value.len()
                    )));
                }
            }
            // A `ref` is a bare `u64` target offset (a non-owning alias).
            Shape::Ref(t) => {
                if value.len() != 8 {
                    return Err(set_error(format!(
                        "a `ref` field is an 8-byte offset, got {}",
                        value.len()
                    )));
                }
                // Validate the offset names a live block of the ref's type — an
                // unchecked offset would let a later path descend into an arbitrary
                // in-file location.
                let target = get_u64(value);
                verify_data_block(data, target, t)?;
            }
            _ => {
                return Err(set_error(
                    "field is not POD / `ref` / class variable; an owning reference is \
                     `swap`ped, not set",
                ));
            }
        }
        data.set(offset, value)
    }

    /// **Swap** the reference field named by `path` to point at `new`, returning the
    /// previous target as an [`AnyRef`] (`None` if it was null). A pointer exchange:
    /// the field takes ownership of `new`, and the old reference is handed back for
    /// the caller to reuse or tear down — no refcount changes (ownership moves, it is
    /// not duplicated).
    ///
    /// `new` is **validated** against the on-disk header before it is installed: a live
    /// block of the field's type must sit at its offset (for a `weak` field, at the
    /// control block's forward data pointer). This keeps a fabricated [`AnyRef`] from
    /// pointing an owning slot at an arbitrary location — rejected with `[BSTACK0815]`.
    ///
    /// `new`'s [`tag`](AnyRef::tag) **must equal the field's declared type** (an
    /// eightcc mismatch is rejected), and the target must be an in-file reference
    /// (`owned` / `strong` / `weak` / `ref`, optionally `Option`-wrapped). For a `weak`
    /// field, `new` and the returned old reference are the target's **control-block**
    /// [`AnyRef`] (exactly what [`move_out`](Self::move_out) hands back). A POD field or
    /// a container is rejected; a cross-file `foreign` field uses
    /// [`swap_foreign`](Self::swap_foreign) instead (an [`AnyRef`] can't name its file).
    pub fn swap(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
        path: &[&str],
        new: AnyRef,
    ) -> io::Result<Option<AnyRef>> {
        let (offset, mut shape) = match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => (offset, shape),
            Resolved::Class { .. } => {
                return Err(swap_error(
                    "a class variable is a value, not a reference — use `set`",
                ));
            }
        };
        // A nullable reference swaps its inner target; remember the nullability —
        // it gates whether a null `new` may be installed at all.
        let nullable = matches!(shape, Shape::Option(_));
        if let Shape::Option(inner) = shape {
            shape = *inner;
        }
        let (tag, is_weak) = match shape {
            // `owned`/`strong`/`ref` hold a data offset; `weak` holds a control offset —
            // both are a single `u64` slot exchanged the same way (no refcount change).
            Shape::Owned(t) | Shape::Strong(t) | Shape::Ref(t) => (t, false),
            Shape::Weak(t) => (t, true),
            Shape::Foreign { .. } => {
                return Err(swap_error(
                    "a `foreign` reference names a cross-file target — use `swap_foreign`",
                ));
            }
            _ => return Err(swap_error("field is not a swappable reference")),
        };
        if new.tag() != tag {
            return Err(swap_error(
                "eightcc mismatch: `new` is not the field's type",
            ));
        }
        // A non-nullable slot must never hold the `0` niche: the generated walks
        // treat non-nullable as proof of non-null, so installing a null here would
        // persist a handle over offset 0 and derail every later read / teardown.
        if new.offset() == 0 && !nullable {
            return Err(swap_error(
                "[BSTACK0815] RTTI mutator: a null reference cannot be installed \
                 into a non-nullable field",
            ));
        }
        // Validate that `new` actually names a live block of the field's type before
        // installing its raw offset — an unchecked offset would let a later teardown
        // free (or a path descend into) an arbitrary location.
        if is_weak {
            // `new`'s offset must name a *control* block. Validate it two ways:
            // (1) directly — its own header tag must equal the target
            // type's control tag, so a region that merely forward-points at a live
            // target (an ordinary byte vector's data block whose bytes happen to line
            // up) is rejected outright; (2) through its forward data pointer, then
            // require the data block's backpointer to round-trip to `new` — the
            // structural cross-check the direct tag alone does not give.
            if new.offset() != 0 {
                // (1) Direct: the control block's header tag. Enforced whenever the
                // target type's schema records a control tag (every `weak` type does);
                // if unresolvable, fall back to the forward-pointer check below.
                let ctrl_tag = self
                    .ordinal_of(tag)
                    .and_then(|ord| self.load_type(ord).ok())
                    .and_then(|t| t.ctrl_tag);
                if let Some(ctrl_tag) = ctrl_tag {
                    let mut hdr = [0u8; 8];
                    data.get_into(add_off(new.offset(), HEADER_TAG_OFFSET)?, &mut hdr)?;
                    if EightCC(hdr) != ctrl_tag {
                        return Err(swap_error(format!(
                            "[BSTACK0815] RTTI mutator: offset {} does not hold a live \
                             control block of the target type (its header tag is not the \
                             type's control tag)",
                            new.offset()
                        )));
                    }
                }
                // (2) Forward data pointer + backpointer round-trip.
                let data_ptr = read_u64_at(data, add_off(new.offset(), CTRL_DATA_OFFSET)?)?;
                verify_data_block(data, data_ptr, tag)?;
                let backptr = read_u64_at(data, add_off(data_ptr, CTRL_BACKPTR_OFFSET)?)?;
                if backptr != new.offset() {
                    return Err(swap_error(format!(
                        "[BSTACK0815] RTTI mutator: offset {} is not the target's \
                         control block (its backpointer names {backptr})",
                        new.offset()
                    )));
                }
            }
        } else {
            verify_data_block(data, new.offset(), tag)?;
        }
        // Atomic exchange: install the new offset and take the displaced one in one
        // locked step, so concurrent callers each get the distinct old target they
        // displaced — never both hand back an owning `AnyRef` to the same block.
        let old_bytes = data.swap(offset, new.offset().to_le_bytes())?;
        let old = u64::from_le_bytes(old_bytes[..8].try_into().unwrap());
        // SAFETY: `old` was displaced from the field's own slot, which held a live
        // target of the field's declared (schema-resolved) tag.
        Ok((old != 0).then(|| unsafe { AnyRef::new(tag, old) }))
    }

    /// **Swap** the cross-file `Foreign` reference named by `path` to point at `new`,
    /// returning the previous target as a [`ForeignPtr`] (`None` if it was null) — the
    /// foreign analog of [`swap`](Self::swap). A wholesale exchange of the 16-byte
    /// pointer with **no** cross-file refcount change: ownership moves, so the old
    /// target is handed back for the caller to reclaim in its own file (or re-store)
    /// and the new one is installed. (`swap` takes an [`AnyRef`], which can't name a
    /// target's file — hence the separate entry.)
    ///
    /// `new.tag` **must equal the field's foreign target type** (an eightcc mismatch is
    /// rejected); `new.kind` is informational (the field's schema kind governs). The
    /// path must resolve to a scalar `Foreign` (optionally `Option`-wrapped). `new` is
    /// **validated** against the target file's on-disk header before install — a live
    /// block of the field's type must sit at `(file_id, offset)` — so the target's file
    /// must be `attach`ed (or `SELF`); a fabricated or unresolvable pointer is rejected
    /// with `[BSTACK0815]`.
    pub fn swap_foreign(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
        path: &[&str],
        new: ForeignPtr,
    ) -> io::Result<Option<ForeignPtr>> {
        let (offset, mut shape) = match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => (offset, shape),
            Resolved::Class { .. } => {
                return Err(swap_error(
                    "a class variable is a value, not a reference — use `set`",
                ));
            }
        };
        let nullable = matches!(shape, Shape::Option(_));
        if let Shape::Option(inner) = shape {
            shape = *inner;
        }
        let (tag, kind) = match shape {
            Shape::Foreign { tag, kind } => (tag, kind),
            _ => {
                return Err(swap_error(
                    "field is not a `foreign` reference — use `swap` for in-file references",
                ));
            }
        };
        if new.tag != tag {
            return Err(swap_error(
                "eightcc mismatch: `new` is not the field's foreign target type",
            ));
        }
        // As `swap`: a non-nullable slot must never hold the null niche.
        if new.offset == 0 && !nullable {
            return Err(swap_error(
                "[BSTACK0815] RTTI mutator: a null foreign reference cannot be \
                 installed into a non-nullable field",
            ));
        }
        // Validate the new target names a live block of the field's type in its own
        // file before installing the raw pointer — an unchecked `(file_id, offset)`
        // would let a later cross-file teardown free an arbitrary range in that file.
        if new.offset != 0 {
            let fid = FileId::from_u64(new.file_id)
                .ok_or_else(|| swap_error("invalid foreign file id in `new`"))?;
            if fid.is_self() {
                verify_data_block(data, new.offset, new.tag)?;
            } else {
                registry::with_host(fid, |h| verify_data_block(h.stack(), new.offset, new.tag))
                    .ok_or_else(|| {
                    swap_error(
                        "the new target's file is not attached — cannot validate the pointer",
                    )
                })??;
            }
        }
        // Build the new 16-byte `WidePtr { file_id:u32, type_index:u32, offset:u64 }`
        // (type_index = the target's ordinal + 1, per `typed_ptr`).
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        let mut b = [0u8; FOREIGN_REPR_LEN as usize];
        b[0..4].copy_from_slice(&(new.file_id as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(ord + 1).to_le_bytes());
        b[8..16].copy_from_slice(&new.offset.to_le_bytes());
        // Atomic exchange of the whole 16-byte pointer, taking the old one this
        // caller displaced.
        let old_repr = data.swap(offset, b)?;
        let __wp = WidePtr::decode(&old_repr);
        let (old_file, old_off) = (__wp.file_id(), __wp.offset().get());
        Ok((old_off != 0).then_some(ForeignPtr {
            tag,
            kind,
            file_id: old_file,
            offset: old_off,
        }))
    }

    /// **Disassemble** the block of type `ordinal` at `block_off`: hand back its
    /// immediate fields as owned [`Moved`] parts, freeing **only the parent shell** —
    /// the interpreted `bstack_move!`, the inverse of construction.
    ///
    /// Each field's ownership transfers to the returned [`SmallStringMap`] entry: POD
    /// (and inline POD arrays / tuples) come out by value; `owned` / `strong` / `ref`
    /// children come out as [`AnyRef`]s (the child block stays alive); a `weak` comes
    /// out as its control [`AnyRef`]; a whole **vector** transfers as a [`VecRef`]
    /// (its data block untouched, exactly like a detached `BStackVec`); a flat reference
    /// **array** is handed out element-by-element as a [`Moved::List`] (`owned`/`ref`,
    /// data offsets) or a [`Moved::WeakList`] (`weak`, control-block offsets — kept
    /// distinct so control bytes are never mistaken for a `T`); a foreign array comes out
    /// as a [`Moved::ForeignList`], a nested reference array as a [`Moved::Array`] — the
    /// inline storage dies with the shell.
    ///
    /// A `rc` / `(rc, weak)` root is disassembled only when the caller is its **sole**
    /// strong owner (a try_unwrap, exactly like `bstack_move!` on a `BStackRc`); a shared
    /// root is refused (`[BSTACK0819]`) untouched. For `(rc, weak)` the control block is
    /// released as part of the disassembly. An **`#[embed]`** child (scalar or an
    /// array of them) is *materialized* — copied into a fresh block (so its
    /// grandchildren transfer with it) and returned as an `AnyRef`. A cross-file
    /// **`foreign`** pointer (scalar, in a vector kept whole, in an array, or a tuple
    /// member) is handed back verbatim ([`Moved::Foreign`] / [`Moved::ForeignList`] /
    /// [`Moved::Tuple`]); its target lives in another file and outlives the shell. Class
    /// variables are skipped (schema-side).
    ///
    /// After this the block itself is gone; the caller owns every returned part and
    /// must reuse or tear each down. On any error nothing is freed *except* orphaned
    /// embed copies, so the object is left intact.
    ///
    /// # Safety
    ///
    /// `block_off` must name a live block of type `ordinal` that the caller owns,
    /// already detached from any parent: the shell is freed unconditionally, so a
    /// wrong offset frees storage the caller does not own and a still-linked root
    /// leaves its parent pointing at freed storage.
    pub unsafe fn move_out<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        ordinal: RttiOrdinal,
        block_off: u64,
    ) -> io::Result<SmallStringMap<Moved>> {
        let data = alloc.stack();
        let mut cache: HashMap<RttiOrdinal, RttiType> = HashMap::new();
        let mut materialized: Vec<BStackRange> = Vec::new();

        // Load the root type up front so reference counting is honoured before anything
        // is touched. A `rc` / `(rc, weak)` root follows `bstack_move!`'s try_unwrap: only
        // the *sole* strong owner may disassemble it. A shared root is refused untouched —
        // freeing its shell would be a use-after-free for the other owners, and for
        // `(rc, weak)` leave the control block naming freed data. `ctrl_off` is `Some` for
        // the `(rc, weak)` case (its separate control block), `None` otherwise.
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ordinal) {
            e.insert(self.load_type(ordinal)?);
        }
        let (strong_slot, ctrl_off) = {
            let root = &cache[&ordinal];
            if root.rc {
                let (strong_slot, ctrl) = if root.weak {
                    let c = read_u64_at(data, add_off(block_off, CTRL_BACKPTR_OFFSET)?)?;
                    (add_off(c, CTRL_STRONG_OFFSET)?, Some(c))
                } else {
                    (add_off(block_off, RC_REFCOUNT_OFFSET)?, None)
                };
                // Atomic try-unwrap, exactly as `BStackRc::try_move`: claim sole
                // ownership by CAS `strong: 1 -> 0`, so a concurrent clone/upgrade
                // either beats the move (the CAS fails cleanly) or is refused by
                // the zero count for the whole field walk — never both succeeding.
                if !refcount::cas(data, strong_slot, 1, 0)? {
                    let strong = read_u64_at(data, strong_slot)?;
                    return Err(corrupt(format!(
                        "[BSTACK0819] RTTI move_out of a shared reference-counted block \
                         (strong count {strong}); only the sole owner may disassemble it"
                    )));
                }
                (Some(strong_slot), ctrl)
            } else {
                (None, None)
            }
        };

        let map = match self.move_fields(
            alloc,
            data,
            ordinal,
            block_off,
            &mut cache,
            &mut materialized,
        ) {
            Ok(m) => m,
            Err(e) => {
                // Object untouched; only orphaned embed copies (if any) are reclaimed.
                // Restore the strong count the CAS took, so the still-intact object
                // keeps its sole owner.
                if let Some(slot) = strong_slot {
                    let _ = refcount::fetch_add(data, slot, 1);
                }
                // SAFETY: `materialized` are this call's own embed copies.
                let _ = unsafe { alloc.free_many(std::mem::take(&mut materialized)) };
                return Err(e);
            }
        };
        // The strong count is already 0 (claimed by the CAS above), so no
        // outstanding weak can upgrade to the data block being freed.
        // Free the shell only — children / vec data / embed copies are all
        // transferred — plus, for `(rc, weak)`, the control block once its phantom
        // weak is released and no real weak handles remain (else it stays, with
        // strong == 0 refusing any upgrade).
        let mut to_free = vec![BStackRange::new(block_off, cache[&ordinal].ondisk_size)];
        if let Some(ctrl_off) = ctrl_off
            && refcount::fetch_sub(data, add_off(ctrl_off, CTRL_WEAK_OFFSET)?, 1)? == 1
        {
            to_free.push(BStackRange::new(ctrl_off, CONTROL_SIZE));
        }
        // Route through the WAL (or the allocator's atomic bulk free) like the
        // static `bstack_move!` shell teardown, so a crash after the fields moved
        // out but before these frees commit is reclaimed on the next open, not
        // leaked permanently.
        // SAFETY: the shell is the caller-owned root (its fields already moved out);
        // the control block, if included, has no remaining references; both live in
        // this file.
        if let Err(e) = unsafe { crate::teardown::commit_home_frees(alloc, to_free) } {
            // SAFETY: `materialized` are this call's own embed copies.
            let _ = unsafe { alloc.free_many(std::mem::take(&mut materialized)) };
            return Err(e);
        }
        Ok(map)
    }

    /// Read the root's immediate fields into a [`Moved`] map (the shell is freed by
    /// the caller). Loads the root type into `cache`.
    fn move_fields<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
        cache: &mut HashMap<RttiOrdinal, RttiType>,
        materialized: &mut Vec<BStackRange>,
    ) -> io::Result<SmallStringMap<Moved>> {
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ordinal) {
            e.insert(self.load_type(ordinal)?);
        }
        // Own the (fields, base) so the `cache` borrow is released before `move_field`
        // needs it mutably (for embed / stride lookups).
        let (fields, base): (Box<[RttiField]>, u64) = {
            let ty = &cache[&ordinal];
            match &ty.body {
                RttiBody::Struct(f) => (f.clone(), block_off),
                RttiBody::Enum(e) => {
                    let raw =
                        read_disc(data, add_off(block_off, e.disc_off as u64)?, e.disc_width)?;
                    let mask = disc_mask(e.disc_width);
                    let variant = e
                        .variants
                        .iter()
                        .find(|v| (v.disc_value as u64) & mask == raw)
                        .ok_or_else(|| {
                            corrupt(format!(
                                "[BSTACK0808] no RTTI variant for discriminant {raw}"
                            ))
                        })?;
                    (
                        variant.fields.clone(),
                        add_off(block_off, e.payload_off as u64)?,
                    )
                }
            }
        };

        let mut map = SmallStringMap::with_capacity(fields.len());
        for f in &fields {
            // Class variables live in the schema, not the instance — nothing to move.
            if matches!(f.shape, Shape::Class { .. }) {
                continue;
            }
            let moved = self.move_field(
                alloc,
                data,
                &f.shape,
                add_off(base, f.offset as u64)?,
                cache,
                materialized,
            )?;
            // A duplicate field name (a corrupt schema) would silently replace —
            // and so discard — the first field's transferred ownership. Error
            // instead: the caller's error path reclaims `materialized`, and per
            // move_out's contract nothing else has been freed yet.
            if map.insert(f.name.clone(), moved).is_some() {
                return Err(corrupt(format!(
                    "[BSTACK0800] RTTI record has two fields named '{}'",
                    f.name
                )));
            }
        }
        Ok(map)
    }

    /// Move one field out: read its value / capture its reference, transferring
    /// ownership. `#[embed]` allocates a materialized copy (recorded in `materialized`
    /// so a later failure can reclaim it).
    fn move_field<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        data: &BStack,
        shape: &Shape,
        off: u64,
        cache: &mut HashMap<RttiOrdinal, RttiType>,
        materialized: &mut Vec<BStackRange>,
    ) -> io::Result<Moved> {
        Ok(match shape {
            Shape::Pod { width } => {
                // Untrusted width: bound against the stack before allocating.
                if *width as u64 > data.len()?.saturating_sub(off) {
                    return Err(corrupt(
                        "[BSTACK0800] RTTI POD width runs past the end of the data stack",
                    ));
                }
                let mut buf = vec![0u8; *width as usize];
                data.get_into(off, &mut buf)?;
                Moved::Pod(buf.into())
            }
            // SAFETY (all `AnyRef::new` in this fn): each offset is read from the
            // moved-out block's own slot (or is a block this fn just allocated),
            // and each tag is the slot's schema-declared element tag.
            Shape::Owned(tag) | Shape::Strong(tag) | Shape::Ref(tag) => {
                let child = read_u64_at(data, off)?;
                Moved::Ref((child != 0).then(|| unsafe { AnyRef::new(*tag, child) }))
            }
            Shape::Weak(tag) => {
                let ctrl = read_u64_at(data, off)?;
                Moved::Weak((ctrl != 0).then(|| unsafe { AnyRef::new(*tag, ctrl) }))
            }
            Shape::Embed(tag) => {
                // Materialize the inline child into a standalone block (its offsets are
                // byte-copied, so its grandchildren transfer with it).
                let ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ord) {
                    e.insert(self.load_type(ord)?);
                }
                let size = cache[&ord].ondisk_size;
                let slice = alloc.alloc(size)?;
                let range = slice.as_range();
                materialized.push(range);
                data.copy(off, range.start(), size)?;
                Moved::Ref(Some(unsafe { AnyRef::new(*tag, range.start()) }))
            }
            Shape::Foreign { tag, kind } => {
                // The target lives in another file; hand its pointer to the caller.
                let __wp = WidePtr::read_from_stack(data, off)?;
                let (file_id, offset) = (__wp.file_id(), __wp.offset().get());
                Moved::Foreign {
                    tag: *tag,
                    kind: *kind,
                    file_id,
                    offset,
                }
            }
            // The niche's `0` is handled by the inner leaf (which reads the same slot).
            Shape::Option(inner) => {
                self.move_field(alloc, data, inner, off, cache, materialized)?
            }
            Shape::Vec(inner) => {
                let data_off = read_u64_at(data, off)?; // VecDesc.data_off @0
                if data_off == 0 {
                    Moved::Vec(None)
                } else {
                    Moved::Vec(Some(VecRef {
                        data_off,
                        data_size: read_u64_at(data, add_off(off, 8)?)?,
                        elem: (**inner).clone(),
                    }))
                }
            }
            Shape::Array { n, inner } => {
                if let Some((tag, kind)) = foreign_leaf(inner) {
                    // A foreign array: each element is a 16-byte `WidePtr` inline
                    // in the shell; hand every cross-file pointer back to the caller.
                    let mut list = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        let __wp =
                            WidePtr::read_from_stack(data, add_off(off, mul_off(i, FOREIGN_REPR_LEN)?)?)?;
                        let (file_id, offset) = (__wp.file_id(), __wp.offset().get());
                        list.push(ForeignPtr {
                            tag,
                            kind,
                            file_id,
                            offset,
                        });
                    }
                    Moved::ForeignList(list.into())
                } else if let Some(tag) = weak_element_tag(inner) {
                    // A weak array (`[#[bstack_weak] T; N]`, opt): each element is a
                    // `u64` **control-block** offset at `off + i*8`. Kept distinct from a
                    // data-ref list (`Moved::WeakList`, the array analog of `Moved::Weak`)
                    // so a control offset is never handed back as if it named a `T`.
                    let mut list = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        let e = read_u64_at(data, add_off(off, mul_off(i, 8)?)?)?;
                        list.push((e != 0).then(|| unsafe { AnyRef::new(tag, e) }));
                    }
                    Moved::WeakList(list.into())
                } else if let Some(tag) = element_ref_tag(inner) {
                    // A flat data-reference array (`owned` / `strong` / `ref`, opt): each
                    // element is a `u64` **data** offset at `off + i*8`.
                    let mut list = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        let e = read_u64_at(data, add_off(off, mul_off(i, 8)?)?)?;
                        list.push((e != 0).then(|| unsafe { AnyRef::new(tag, e) }));
                    }
                    Moved::List(list.into())
                } else if let Shape::Embed(etag) = &**inner {
                    // An array of embedded children (`#[embed] [Child; N]`): each is
                    // stored inline; materialize each into a fresh standalone block (its
                    // grandchildren transfer via the copied offsets), recorded so a later
                    // failure reclaims them.
                    let ord = self.ordinal_of(*etag).ok_or_else(unknown_tag)?;
                    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ord) {
                        e.insert(self.load_type(ord)?);
                    }
                    let size = cache[&ord].ondisk_size;
                    let mut list = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        let range = alloc.alloc(size)?.as_range();
                        materialized.push(range);
                        data.copy(add_off(off, mul_off(i, size)?)?, range.start(), size)?;
                        list.push(Some(unsafe { AnyRef::new(*etag, range.start()) }));
                    }
                    Moved::List(list.into())
                } else if matches!(&**inner, Shape::Array { .. }) && shape_has_reference(inner) {
                    // A nested reference array (`[[T; M]; N]`, …): move each outer
                    // element (itself a container) as its own `Moved`.
                    let stride = self.shape_stride(inner, cache)?;
                    let mut parts = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        parts.push(self.move_field(
                            alloc,
                            data,
                            inner,
                            add_off(off, mul_off(i, stride)?)?,
                            cache,
                            materialized,
                        )?);
                    }
                    Moved::Array(parts.into())
                } else if shape_has_reference(inner) {
                    // Any other reference-bearing element we can't flatten one-per-`u64`.
                    return Err(move_unsupported());
                } else {
                    // A POD array (nested or not): the whole inline run of bytes.
                    let total = mul_off(*n as u64, self.shape_stride(inner, cache)?)?;
                    let mut buf = vec![0u8; total as usize];
                    data.get_into(off, &mut buf)?;
                    Moved::Pod(buf.into())
                }
            }
            Shape::Tuple(items) => {
                if items.iter().any(|it| foreign_leaf(it).is_some()) {
                    // A tuple with a `Foreign` member: move each member individually —
                    // POD by value, a foreign member as its own `Moved::Foreign` — at
                    // cumulative element offsets.
                    let mut parts = Vec::with_capacity(items.len());
                    let mut eo = off;
                    for it in items {
                        parts.push(self.move_field(alloc, data, it, eo, cache, materialized)?);
                        eo = add_off(eo, self.shape_stride(it, cache)?)?;
                    }
                    Moved::Tuple(parts.into())
                } else {
                    // A POD aggregate: its inline bytes (sum of element strides).
                    let mut total = 0u64;
                    for it in items {
                        total = add_off(total, self.shape_stride(it, cache)?)?;
                    }
                    let mut buf = vec![0u8; total as usize];
                    data.get_into(off, &mut buf)?;
                    Moved::Pod(buf.into())
                }
            }
            // Filtered out before this call, but keep the match total.
            Shape::Class { .. } => Moved::Pod(Box::default()),
        })
    }

    /// Deep-clone the structure of type `ordinal` at `src_off` in `alloc`'s file,
    /// returning the **detached** clone's root offset (the caller links it) — the
    /// inverse of [`teardown`](Self::teardown).
    ///
    /// The walk is **non-recursive**. Owned (`owned` / `embed`) sub-structure is
    /// byte-copied into fresh blocks and repointed; shared references stay shared —
    /// a `strong` bumps the target's strong count, a `weak` bumps its weak count,
    /// a `ref` is a byte-copied alias; POD is copied verbatim. Vectors get a fresh
    /// data block (and cloned owned elements). Every allocation is orphaned until the
    /// caller links the root, so a crash mid-clone leaks but never corrupts; on any
    /// error the partial clone's blocks are freed. Cross-file `foreign` references —
    /// scalar or inside a `vec` / array / tuple — are handled per their kind in the
    /// target's own file (`owned` deep-copied there, `strong` / `weak` bumped, `ref`
    /// aliased); a detached target file is a hard error.
    ///
    /// # Safety
    ///
    /// `src_off` must name a live block of type `ordinal`. The walk reads and
    /// deep-copies whatever bytes sit there as that type's layout: a wrong offset
    /// duplicates arbitrary storage into new blocks, bumps refcounts at whatever
    /// offsets the misread slots hold, and hands back a root whose later teardown
    /// frees ranges derived from those misreads.
    pub unsafe fn clone_value<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        ordinal: RttiOrdinal,
        src_off: u64,
    ) -> io::Result<u64> {
        // Bound cross-file recursion (each `Foreign` hop recurses here natively).
        let _depth = DepthGuard::enter()?;
        let data = alloc.stack();
        let mut st = CloneState::default();
        match self.clone_build(alloc, data, ordinal, src_off, &mut st) {
            Ok(new_root) => {
                // Success: the intention-first-logged allocations are now real blocks.
                // Mark the WAL transaction idle so `finish` keeps them instead of
                // reclaiming (a clone logs only `Alloc` entries, so idling == "these
                // are real"). A crash before this leaves them `Pending` and reclaimable
                // — correct, since `clone_value` has not returned the tree yet.
                if let Some(w) = st.wal.as_ref() {
                    wal_set_idle(alloc, w.block_off)?;
                }
                Ok(new_root)
            }
            Err(e) => {
                // Reclaim the orphaned partial clone (leak-free error path). With a WAL
                // transaction in flight, abandon it — `finish_at_locked` frees exactly
                // the still-`Pending` `Alloc`s (== `st.allocated`) and marks the block
                // idle, the same path a crash takes. Otherwise free the ranges directly.
                if st.wal.is_some() {
                    let _ = finish_at_locked(alloc);
                } else {
                    // SAFETY: `st.allocated` are this clone's own partial allocations.
                    let _ = unsafe { alloc.free_many(std::mem::take(&mut st.allocated)) };
                }
                Err(e)
            }
        }
    }

    /// The clone walk + the deferred child-repointing and refcount bumps. Split out
    /// so [`clone_value`](Self::clone_value) can free `st.allocated` on any error.
    fn clone_build<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        data: &BStack,
        ordinal: RttiOrdinal,
        root_src: u64,
        st: &mut CloneState,
    ) -> io::Result<u64> {
        let mut work: Vec<CloneOp> = vec![CloneOp::Block {
            src_off: root_src,
            ord: ordinal,
        }];
        let mut budget: u64 = 4_000_000;

        while let Some(op) = work.pop() {
            budget = budget.checked_sub(1).ok_or_else(|| {
                corrupt("[BSTACK0807] RTTI clone budget exceeded (corrupt data or a cycle?)")
            })?;
            match op {
                CloneOp::Block { src_off, ord } => {
                    self.ensure_type(ord, st)?;
                    let size = st.cache[&ord].ondisk_size;
                    let new_off = self.alloc_copy(alloc, data, src_off, size, st)?;
                    st.map.insert(src_off, new_off);
                    // Walk the fields at matching source / destination offsets.
                    let ty = &st.cache[&ord];
                    match &ty.body {
                        RttiBody::Struct(fields) => {
                            for f in fields {
                                work.push(CloneOp::Field {
                                    shape: f.shape.clone(),
                                    src_off: add_off(src_off, f.offset as u64)?,
                                    new_off: add_off(new_off, f.offset as u64)?,
                                });
                            }
                        }
                        RttiBody::Enum(e) => {
                            let raw = read_disc(
                                data,
                                add_off(src_off, e.disc_off as u64)?,
                                e.disc_width,
                            )?;
                            let mask = disc_mask(e.disc_width);
                            let variant = e
                                .variants
                                .iter()
                                .find(|v| (v.disc_value as u64) & mask == raw)
                                .ok_or_else(|| {
                                    corrupt(format!(
                                        "[BSTACK0808] no RTTI variant for discriminant {raw}"
                                    ))
                                })?;
                            let sp = add_off(src_off, e.payload_off as u64)?;
                            let np = add_off(new_off, e.payload_off as u64)?;
                            for f in &variant.fields {
                                work.push(CloneOp::Field {
                                    shape: f.shape.clone(),
                                    src_off: add_off(sp, f.offset as u64)?,
                                    new_off: add_off(np, f.offset as u64)?,
                                });
                            }
                        }
                    }
                }

                CloneOp::Inline {
                    src_base,
                    new_base,
                    ord,
                } => {
                    self.ensure_type(ord, st)?;
                    let ty = &st.cache[&ord];
                    match &ty.body {
                        RttiBody::Struct(fields) => {
                            for f in fields {
                                work.push(CloneOp::Field {
                                    shape: f.shape.clone(),
                                    src_off: add_off(src_base, f.offset as u64)?,
                                    new_off: add_off(new_base, f.offset as u64)?,
                                });
                            }
                        }
                        RttiBody::Enum(e) => {
                            let raw = read_disc(
                                data,
                                add_off(src_base, e.disc_off as u64)?,
                                e.disc_width,
                            )?;
                            let mask = disc_mask(e.disc_width);
                            let variant = e
                                .variants
                                .iter()
                                .find(|v| (v.disc_value as u64) & mask == raw)
                                .ok_or_else(|| {
                                    corrupt(format!(
                                        "[BSTACK0808] no RTTI variant for discriminant {raw}"
                                    ))
                                })?;
                            let sp = add_off(src_base, e.payload_off as u64)?;
                            let np = add_off(new_base, e.payload_off as u64)?;
                            for f in &variant.fields {
                                work.push(CloneOp::Field {
                                    shape: f.shape.clone(),
                                    src_off: add_off(sp, f.offset as u64)?,
                                    new_off: add_off(np, f.offset as u64)?,
                                });
                            }
                        }
                    }
                }

                CloneOp::Field {
                    shape,
                    src_off,
                    new_off,
                } => match shape {
                    // Copied verbatim: inline bytes, a schema value, or a `ref` alias.
                    Shape::Pod { .. } | Shape::Class { .. } | Shape::Ref(_) => {}
                    Shape::Owned(tag) => {
                        let child = read_u64_at(data, src_off)?;
                        if child != 0 {
                            let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
                            st.patches.push((new_off, child));
                            work.push(CloneOp::Block {
                                src_off: child,
                                ord,
                            });
                        }
                    }
                    Shape::Embed(tag) => {
                        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
                        work.push(CloneOp::Inline {
                            src_base: src_off,
                            new_base: new_off,
                            ord,
                        });
                    }
                    Shape::Strong(tag) => {
                        let child = read_u64_at(data, src_off)?;
                        if child != 0 {
                            let off = self.strong_bump_off(data, tag, child, st)?;
                            st.bumps.push(off);
                        }
                    }
                    Shape::Weak(_) => {
                        let ctrl = read_u64_at(data, src_off)?;
                        if ctrl != 0 {
                            st.bumps.push(add_off(ctrl, CTRL_WEAK_OFFSET)?);
                        }
                    }
                    Shape::Foreign { tag, kind } => {
                        // The slot was byte-copied (same target); repoint an `owned`
                        // deep-copy across the file boundary, bump `strong`/`weak`.
                        self.clone_foreign(alloc, data, tag, kind, src_off, new_off)?;
                    }
                    Shape::Option(inner) => {
                        // The slot (and its `0` niche) is already copied; only a
                        // present reference needs its child cloned / bumped. The niche
                        // location depends on the inner shape (`Foreign` → offset @8).
                        if option_present(data, &inner, src_off)? {
                            work.push(CloneOp::Field {
                                shape: *inner,
                                src_off,
                                new_off,
                            });
                        }
                    }
                    Shape::Array { n, inner } => {
                        // Charge for all elements up front — `n` is untrusted and
                        // the ops are materialized eagerly (see the read walk).
                        budget = budget.checked_sub(n as u64).ok_or_else(|| {
                            corrupt(
                                "[BSTACK0807] RTTI clone budget exceeded (corrupt data or a cycle?)",
                            )
                        })?;
                        let stride = self.shape_stride(&inner, &mut st.cache)?;
                        for i in 0..n as u64 {
                            let delta = mul_off(i, stride)?;
                            work.push(CloneOp::Field {
                                shape: (*inner).clone(),
                                src_off: add_off(src_off, delta)?,
                                new_off: add_off(new_off, delta)?,
                            });
                        }
                    }
                    Shape::Tuple(items) => {
                        let mut so = src_off;
                        let mut no = new_off;
                        for it in &items {
                            work.push(CloneOp::Field {
                                shape: it.clone(),
                                src_off: so,
                                new_off: no,
                            });
                            let s = self.shape_stride(it, &mut st.cache)?;
                            so = add_off(so, s)?;
                            no = add_off(no, s)?;
                        }
                    }
                    Shape::Vec(inner) => {
                        let src_data = read_u64_at(data, src_off)?; // VecDesc.data_off
                        if src_data != 0 {
                            let data_size = read_u64_at(data, add_off(src_off, 8)?)?;
                            let new_data = self.alloc_copy(alloc, data, src_data, data_size, st)?;
                            // Repoint the (freshly-copied) descriptor's data pointer;
                            // its size word was copied verbatim.
                            data.set(new_off, new_data.to_le_bytes())?;
                            // `@0` is the byte length; count is `byte_len / stride`
                            // (8 per `u64` offset, 16 per `WidePtr`).
                            let stride = self.shape_stride(&inner, &mut st.cache)?;
                            let byte_len = read_u64_at(data, src_data)?;
                            let len = checked_vec_len(byte_len, data_size, stride)?;
                            let sbase = add_off(src_data, BYTEVEC_HEADER)?;
                            let nbase = add_off(new_data, BYTEVEC_HEADER)?;
                            match &*inner {
                                Shape::Owned(tag) => {
                                    let ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                                    for i in 0..len {
                                        let delta = mul_off(i, stride)?;
                                        let e = read_u64_at(data, add_off(sbase, delta)?)?;
                                        if e != 0 {
                                            st.patches.push((add_off(nbase, delta)?, e));
                                            work.push(CloneOp::Block { src_off: e, ord });
                                        }
                                    }
                                }
                                Shape::Strong(tag) => {
                                    for i in 0..len {
                                        let e = read_u64_at(
                                            data,
                                            add_off(sbase, mul_off(i, stride)?)?,
                                        )?;
                                        if e != 0 {
                                            let off = self.strong_bump_off(data, *tag, e, st)?;
                                            st.bumps.push(off);
                                        }
                                    }
                                }
                                Shape::Weak(_) => {
                                    for i in 0..len {
                                        let c = read_u64_at(
                                            data,
                                            add_off(sbase, mul_off(i, stride)?)?,
                                        )?;
                                        if c != 0 {
                                            st.bumps.push(add_off(c, CTRL_WEAK_OFFSET)?);
                                        }
                                    }
                                }
                                // A vector of `Foreign` pointers: the data block (and
                                // its reprs) was byte-copied above; deep-copy each
                                // `owned` target across the boundary, bump `strong` /
                                // `weak` — the per-element mirror of the scalar path.
                                other if foreign_leaf(other).is_some() => {
                                    let (tag, kind) = foreign_leaf(other).unwrap();
                                    for i in 0..len {
                                        let delta = mul_off(i, stride)?;
                                        let so = add_off(sbase, delta)?;
                                        let no = add_off(nbase, delta)?;
                                        self.clone_foreign(alloc, data, tag, kind, so, no)?;
                                    }
                                }
                                // POD / `ref` elements are copied verbatim.
                                _ => {}
                            }
                        }
                    }
                },
            }
        }

        // Every block is cloned and in `map`; repoint owned child pointers.
        for &(new_slot, src_child) in &st.patches {
            let new_child = *st
                .map
                .get(&src_child)
                .ok_or_else(|| corrupt("[BSTACK080E] RTTI clone: an owned child was not cloned"))?;
            data.set(new_slot, new_child.to_le_bytes())?;
        }
        // Then bump every shared target's refcount (over-count-safe, never under).
        for &off in &st.bumps {
            refcount::fetch_add(data, off, 1)?;
        }

        st.map
            .get(&root_src)
            .copied()
            .ok_or_else(|| corrupt("[BSTACK080E] RTTI clone: the root was not cloned"))
    }

    /// Allocate a `size`-byte block and byte-copy `[src_off, src_off+size)` into it,
    /// recording it in `st.allocated`. Returns the new block's start offset.
    fn alloc_copy<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        data: &BStack,
        src_off: u64,
        size: u64,
        st: &mut CloneState,
    ) -> io::Result<u64> {
        let slice = alloc.alloc(size)?;
        let range = slice.as_range();
        // Log the allocation intention-first so a crash mid-clone leaves it
        // reclaimable, keeping the WAL's live entries in lockstep with `allocated`: if
        // logging fails, undo the alloc and log nothing.
        if alloc.wal_anchor().is_some()
            && let Err(e) = Self::wal_log_alloc(alloc, st, range)
        {
            let _ = alloc.dealloc(slice);
            return Err(e);
        }
        st.allocated.push(range);
        data.copy(src_off, range.start(), size)?;
        Ok(range.start())
    }

    /// Log a just-made clone allocation to the intention-first WAL, (lazily) beginning
    /// the transaction (and taking the file's WAL lock) on the first call. Mirrors
    /// [`crate::clone::ClonePlan`]'s `wal_log_alloc`: a cheap append while the block
    /// has spare slots, a full re-`persist_at` (which grows the block) when it is full.
    /// `st.allocated` does **not** yet contain `range` (the caller pushes it only after
    /// this succeeds), so the grow path logs all of `allocated` *plus* `range`.
    fn wal_log_alloc<A: BStackRaiiAllocator>(
        alloc: &A,
        st: &mut CloneState,
        range: BStackRange,
    ) -> io::Result<()> {
        match &mut st.wal {
            None => {
                let held = HeldLock::acquire(alloc)?;
                let mut log = WalLog::with_capacity(1);
                log.append(WalEntry::alloc(WalStatus::Pending, range));
                let block = persist_at(alloc, &log, WalStatus::Pending)?;
                st.wal = Some(CloneWal {
                    _held: held,
                    block_off: block.start(),
                    capacity: wal_capacity_of(block),
                    logged: 1,
                });
                Ok(())
            }
            Some(w) if w.logged < w.capacity => {
                wal_append_alloc(alloc, w.block_off, w.logged, range)?;
                w.logged += 1;
                Ok(())
            }
            Some(_) => {
                let mut log = WalLog::with_capacity(st.allocated.len() + 1);
                for &r in &st.allocated {
                    log.append(WalEntry::alloc(WalStatus::Pending, r));
                }
                log.append(WalEntry::alloc(WalStatus::Pending, range));
                let block = persist_at(alloc, &log, WalStatus::Pending)?;
                let w = st.wal.as_mut().unwrap();
                w.block_off = block.start();
                w.capacity = wal_capacity_of(block);
                w.logged = st.allocated.len() as u64 + 1;
                Ok(())
            }
        }
    }

    /// The counter offset to bump when cloning a `strong` reference to `data_child`
    /// of type `tag`: an `rc` block's inline refcount, or an `(rc, weak)` block's
    /// `ctrl.strong` (reached via the data block's `ctrl` back-pointer). Strong only —
    /// a strong clone never adds a phantom weak.
    fn strong_bump_off(
        &self,
        data: &BStack,
        tag: EightCC,
        data_child: u64,
        st: &mut CloneState,
    ) -> io::Result<u64> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        self.ensure_type(ord, st)?;
        if st.cache[&ord].weak {
            let ctrl = read_u64_at(data, add_off(data_child, CTRL_BACKPTR_OFFSET)?)?;
            add_off(ctrl, CTRL_STRONG_OFFSET)
        } else {
            add_off(data_child, RC_REFCOUNT_OFFSET)
        }
    }

    /// Load + cache a type descriptor if not already present.
    fn ensure_type(&self, ord: RttiOrdinal, st: &mut CloneState) -> io::Result<()> {
        if let std::collections::hash_map::Entry::Vacant(e) = st.cache.entry(ord) {
            e.insert(self.load_type(ord)?);
        }
        Ok(())
    }

    /// The on-disk byte width of one element of `shape` — the stride for array / vec /
    /// tuple element addressing. References are a `u64` offset; a foreign is a
    /// `WidePtr`; an embedded child is its whole block; a vector is its inline
    /// `VecDesc`.
    fn shape_stride(
        &self,
        shape: &Shape,
        cache: &mut HashMap<RttiOrdinal, RttiType>,
    ) -> io::Result<u64> {
        Ok(match shape {
            Shape::Pod { width } => *width as u64,
            Shape::Owned(_) | Shape::Strong(_) | Shape::Weak(_) | Shape::Ref(_) => 8,
            Shape::Foreign { .. } => FOREIGN_REPR_LEN,
            Shape::Vec(_) => VECDESC_LEN,
            Shape::Option(inner) => self.shape_stride(inner, cache)?,
            Shape::Array { n, inner } => mul_off(*n as u64, self.shape_stride(inner, cache)?)?,
            Shape::Tuple(items) => {
                let mut sum = 0u64;
                for it in items {
                    sum = add_off(sum, self.shape_stride(it, cache)?)?;
                }
                sum
            }
            Shape::Embed(tag) => {
                let ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ord) {
                    e.insert(self.load_type(ord)?);
                }
                cache[&ord].ondisk_size
            }
            // A class variable is not part of the instance layout.
            Shape::Class { .. } => 0,
        })
    }
}

fn unknown_tag() -> io::Error {
    corrupt("[BSTACK080B] RTTI pointer/field references an unregistered type tag")
}

fn set_error(msg: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("[BSTACK080D] RTTI set: {msg}"),
    )
}

fn swap_error(msg: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("[BSTACK0810] RTTI swap: {msg}"),
    )
}

/// Error for a mutator ([`set`](RttiRegistry::set) / [`swap`](RttiRegistry::swap) /
/// [`swap_foreign`](RttiRegistry::swap_foreign)) whose caller-supplied target offset
/// does not name a live block of the field's declared type.
fn bad_target(off: u64, want: EightCC, found: Option<EightCC>) -> io::Error {
    let found = match found {
        Some(t) => format!("found {t:?}"),
        None => "out of bounds or unreadable".to_string(),
    };
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "[BSTACK0815] RTTI mutator: offset {off} does not hold a live {want:?} block ({found})"
        ),
    )
}

/// Verify a **live block of type `tag`** sits at `off` in `data` (`off == 0` is the null
/// sentinel, allowed). The safe RTTI mutators install caller-supplied offsets into
/// owning slots; without this check a fabricated [`AnyRef`] / [`ForeignPtr`] could point
/// a slot at an arbitrary location that a later teardown would free (recursively, for
/// `owned`) or a later path would descend into — the same hazard `Foreign::new` /
/// `raw_<field>_slice` are `unsafe` for, but here checkable against the on-disk header.
fn verify_data_block(data: &BStack, off: u64, tag: EightCC) -> io::Result<()> {
    if off == 0 {
        return Ok(());
    }
    match AnyRef::from_block(data, off) {
        Ok(a) if a.tag() == tag => Ok(()),
        Ok(a) => Err(bad_target(off, tag, Some(a.tag()))),
        Err(_) => Err(bad_target(off, tag, None)),
    }
}

fn move_unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "[BSTACK0811] RTTI move_out of an array whose element is a vector (or other \
         reference-bearing container that is neither a flat reference, an `#[embed]`, nor a \
         nested array) is not yet supported",
    )
}

/// Whether `shape` contains any block reference anywhere (so it is not pure POD).
fn shape_has_reference(shape: &Shape) -> bool {
    match shape {
        Shape::Pod { .. } | Shape::Class { .. } => false,
        Shape::Owned(_)
        | Shape::Strong(_)
        | Shape::Weak(_)
        | Shape::Ref(_)
        | Shape::Embed(_)
        | Shape::Foreign { .. } => true,
        Shape::Option(inner) | Shape::Vec(inner) | Shape::Array { inner, .. } => {
            shape_has_reference(inner)
        }
        Shape::Tuple(items) => items.iter().any(shape_has_reference),
    }
}

/// The element tag of a reference-array element (`owned` / `strong` / `weak` / `ref`,
/// optionally `Option`-wrapped) — its slot is a single `u64` offset. `None` for an
/// element the move interpreter can't hand out one-per-`u64` (embed / foreign / nested).
fn element_ref_tag(shape: &Shape) -> Option<EightCC> {
    match shape {
        Shape::Owned(t) | Shape::Strong(t) | Shape::Weak(t) | Shape::Ref(t) => Some(*t),
        Shape::Option(inner) => element_ref_tag(inner),
        _ => None,
    }
}

/// The tag of a **weak** reference leaf (optionally `Option`-wrapped) — its slot holds a
/// `u64` *control-block* offset, not a data offset. `None` for any non-weak shape. Lets
/// `move_out` hand a weak array back as a [`Moved::WeakList`] distinct from a data-ref
/// [`Moved::List`].
fn weak_element_tag(shape: &Shape) -> Option<EightCC> {
    match shape {
        Shape::Weak(t) => Some(*t),
        Shape::Option(inner) => weak_element_tag(inner),
        _ => None,
    }
}

/// The `(tag, kind)` of a cross-file `Foreign` leaf (optionally `Option`-wrapped) —
/// its slot is a 16-byte [`WidePtr`]. `None` for any non-foreign shape. Used to
/// drive the per-element foreign path in a `Vec` / array / tuple.
fn foreign_leaf(shape: &Shape) -> Option<(EightCC, OwnershipKind)> {
    match shape {
        Shape::Foreign { tag, kind } => Some((*tag, *kind)),
        Shape::Option(inner) => foreign_leaf(inner),
        _ => None,
    }
}

/// Whether an `Option<inner>` slot at `base` is `Some`. The null niche's **location
/// depends on the inner shape**: a `Foreign` slot is a 16-byte `WidePtr`
/// `{ file_id:u32 @0, type_index:u32 @4, offset:u64 @8 }` whose niche is the target
/// `offset` word at byte 8 — *not* the leading `file_id|type_index` word (which is
/// `0` for a present untyped SELF-file pointer, so testing it would misread a live
/// pointer as `None`). Every other offset-bearing inner — a block reference (`owned` /
/// `strong` / `weak` / `ref`) or a `Vec` descriptor (`data_off`) — uses the leading
/// `u64`.
fn option_present(data: &BStack, inner: &Shape, base: u64) -> io::Result<bool> {
    Ok(match inner {
        Shape::Foreign { .. } => !WidePtr::read_from_stack(data, base)?.is_null(),
        _ => read_u64_at(data, base)? != 0,
    })
}

/// Whether two type descriptors describe the **same layout** — equal in everything a
/// persisted instance depends on, EXCEPT the current *value* of a **mutable** class
/// variable. A mutable `#[bstack_static]` is updated in place (`set_class_value`), so
/// its persisted value legitimately differs from the compiled type's initial value;
/// a raw `existing == ty` would flag that as a schema change. Everything else — field
/// offsets / shapes / order, `rc`/`weak` mode, `ondisk_size`, and *const* class-var
/// values — must match.
fn layouts_match(a: &RttiType, b: &RttiType) -> bool {
    fn strip_shape(shape: &mut Shape) {
        match shape {
            Shape::Class {
                mutable,
                inner,
                value,
            } => {
                if *mutable {
                    *value = Box::default();
                }
                strip_shape(inner);
            }
            Shape::Option(inner) | Shape::Vec(inner) | Shape::Array { inner, .. } => {
                strip_shape(inner)
            }
            Shape::Tuple(items) => items.iter_mut().for_each(strip_shape),
            _ => {}
        }
    }
    fn strip(ty: &RttiType) -> RttiType {
        let mut t = ty.clone();
        match &mut t.body {
            RttiBody::Struct(fields) => fields.iter_mut().for_each(|f| strip_shape(&mut f.shape)),
            RttiBody::Enum(e) => e
                .variants
                .iter_mut()
                .for_each(|v| v.fields.iter_mut().for_each(|f| strip_shape(&mut f.shape))),
        }
        t
    }
    strip(a) == strip(b)
}

/// The interpret-budget-exhausted error (a corrupt schema/data pair, or a cycle).
fn budget_exceeded() -> io::Error {
    corrupt("[BSTACK0807] RTTI interpret budget exceeded (corrupt data or a cycle?)")
}

thread_local! {
    /// The current cross-file RTTI recursion depth. The interpreter is non-recursive
    /// *within* a file (a work-list), but `teardown` / `clone_value` recurse **natively**
    /// at each `Foreign` hop (through `teardown_foreign_in` / `clone_foreign_in`), each
    /// starting a fresh per-node budget — so the in-file cycle guard can't see across
    /// files. A foreign cycle (`A --owns--> B --owns--> A`, or a SELF back-edge) would
    /// drive unbounded native recursion → stack-overflow abort. [`DepthGuard`] bounds it.
    static RTTI_DEPTH: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// Native cross-file recursion cap. Each hop costs a few KB of native stack (a per-file
/// teardown/clone frame plus the two foreign helpers), so this stays well under the
/// smallest default thread stack (≈2 MiB) with wide margin — yet is far deeper than any
/// sane cross-file `Foreign` chain (the *in-file* walk is non-recursive and unbounded).
const MAX_RTTI_DEPTH: u32 = 100;

/// A scope guard bounding cross-file RTTI recursion (see [`RTTI_DEPTH`]). Created at the
/// top of `teardown` / `clone_value`; increments the depth, decrements on drop (so an
/// error/panic unwinds it cleanly), and refuses to enter past [`MAX_RTTI_DEPTH`].
struct DepthGuard;

impl DepthGuard {
    fn enter() -> io::Result<Self> {
        let depth = RTTI_DEPTH.with(|c| {
            let n = c.get() + 1;
            c.set(n);
            n
        });
        if depth > MAX_RTTI_DEPTH {
            RTTI_DEPTH.with(|c| c.set(c.get() - 1)); // undo: no guard is returned
            return Err(corrupt(
                "[BSTACK0807] RTTI cross-file recursion too deep (a foreign cycle?)",
            ));
        }
        Ok(DepthGuard)
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        RTTI_DEPTH.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// The element count of a vec data block, **validated against the block's own size**.
///
/// The `@0` word is the byte length; a corrupt/forged value must never drive a read or
/// free past the block. `byte_len` must fit the block's element region
/// (`data_size - header`) — otherwise the walk would materialize a petabyte-sized
/// allocation (abort) or, in teardown, read `u64`s from neighboring **live** blocks and
/// free ranges over them. The bound also keeps `base + i*stride` from wrapping. Returns
/// `byte_len / stride`.
fn checked_vec_len(byte_len: u64, data_size: u64, stride: u64) -> io::Result<u64> {
    let usable = data_size.saturating_sub(BYTEVEC_HEADER);
    if byte_len > usable {
        return Err(corrupt(format!(
            "[BSTACK0813] RTTI vector length ({byte_len} bytes) exceeds its data block \
             ({usable} usable bytes) — corrupt length word"
        )));
    }
    Ok(byte_len.checked_div(stride).unwrap_or(0))
}

fn clone_unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        "[BSTACK080F] RTTI clone: the foreign target names an invalid file id",
    )
}

/// Release one `weak` reference whose control block is at `ctrl_off`: decrement
/// `ctrl.weak`; the last weak handle (or phantom) frees the control block. The data
/// block is never touched by a weak drop.
/// Release one deferred `weak` reference (commit phase of teardown): decrement the
/// control block's weak count and free the control block if this was the last handle.
fn commit_weak_release<A: BStackRaiiAllocator>(alloc: &A, ctrl_off: u64) -> io::Result<()> {
    let data = alloc.stack();
    if refcount::fetch_sub(data, add_off(ctrl_off, CTRL_WEAK_OFFSET)?, 1)? == 1 {
        // SAFETY: last weak released — the control block is unreferenced.
        unsafe { alloc.free_many([BStackRange::new(ctrl_off, CONTROL_SIZE)])? };
    }
    Ok(())
}

/// The low-`width`-byte mask for comparing a stored discriminant against a variant's
/// (sign-extended) value.
fn disc_mask(width: u8) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Read a `width`-byte discriminant at `off`, zero-extended to `u64`.
fn read_disc(data: &BStack, off: u64, width: u8) -> io::Result<u64> {
    let w = width as usize;
    if w > 8 {
        // A discriminant fits in a `u64`; a wider width is a corrupt schema. Return
        // `Err` rather than index a `[u8; 8]` out of bounds (`disc_mask` already
        // tolerates `>= 8`). `decode_type` rejects such records on load, so this is
        // a defensive backstop.
        return Err(corrupt(
            "[BSTACK0816] RTTI enum discriminant width exceeds 8 bytes",
        ));
    }
    let mut b = [0u8; 8];
    data.get_into(off, &mut b[..w])?;
    Ok(u64::from_le_bytes(b))
}

fn class_error(msg: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("[BSTACK0812] RTTI class variable: {msg}"),
    )
}

/// Locate the value bytes of the `CLASS` field named `target` within a decoded record
/// body: `(offset within body, value length, mutable)`, or `None` if there is no such
/// (class) field. Walks the `TypeDesc` header + fields exactly as `decode_type` does,
/// stopping at the target instead of building an [`RttiType`].
fn class_value_slot(body: &[u8], target: &str) -> io::Result<Option<(usize, usize, bool)>> {
    let mut r = Reader::new(body);
    let flags = r.u8()?;
    let _disc_width = r.u8()?;
    let name_len = r.u16()? as usize;
    let count = r.u16()? as usize;
    let _disc_off = r.u16()?;
    let _payload_off = r.u16()?;
    let _ondisk_size = r.u64()?;
    let _ctrl_tag = r.eightcc()?; // control tag — same fixed-header slot
    let _name = r.string(name_len)?;
    r.align(8)?;
    // Only structs carry class variables; an enum's `count` is its variants.
    if flags & FLAG_ENUM != 0 {
        return Ok(None);
    }
    for _ in 0..count {
        let _offset = r.u32()?;
        let fname_len = r.u16()? as usize;
        let _shape_len = r.u16()? as usize;
        let fname = r.string(fname_len)?;
        r.align(4)?;
        if fname == target {
            return class_value_within_shape(&mut r);
        }
        // Skip this field's shape (decode advances the cursor, bounds-checked) + pad.
        let _ = Shape::decode(&mut r)?;
        r.align(4)?;
    }
    Ok(None)
}

/// If the shape at the cursor is a `CLASS` shape, consume its header and return
/// `(value offset within body, value length, mutable)` with the cursor left at the
/// value bytes; otherwise `None` (the named field is not a class variable).
fn class_value_within_shape(r: &mut Reader) -> io::Result<Option<(usize, usize, bool)>> {
    if r.u8()? != shape_tag::CLASS {
        return Ok(None);
    }
    let mutable = r.u8()? != 0;
    let _inner = Shape::decode(r)?;
    let value_len = r.u32()? as usize;
    // The value slot `[pos, pos + value_len)` must lie fully within this record's
    // body. Without this check a corrupt `value_len` flows into `set_class_value`,
    // which then writes `value_len` bytes at an offset past the record — tearing a
    // neighboring schema record from a safe call.
    if r.pos
        .checked_add(value_len)
        .is_none_or(|end| end > r.buf.len())
    {
        return Err(class_error("value length exceeds the record body"));
    }
    Ok(Some((r.pos, value_len, mutable)))
}

/// Pop the `n` values a container's children pushed, restoring declaration order.
/// Children are pushed onto `work` in forward order, so they execute (and land on
/// `results`) in reverse — this hands back `[c0, c1, …]`.
fn pop_n(results: &mut Vec<Value>, n: usize) -> io::Result<Vec<Value>> {
    let start = results
        .len()
        .checked_sub(n)
        .ok_or_else(|| corrupt("[BSTACK0809] RTTI interpret stack underflow"))?;
    let mut v = results.split_off(start);
    v.reverse();
    Ok(v)
}

/// Pop `names.len()` values and pair them with the field names, in order.
fn pop_named(results: &mut Vec<Value>, names: &[String]) -> io::Result<Vec<(String, Value)>> {
    let vals = pop_n(results, names.len())?;
    Ok(names.iter().cloned().zip(vals).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(s: &str) -> EightCC {
        EightCC::from_name(s)
    }

    fn sample_struct() -> RttiType {
        RttiType {
            tag: cc("Sample"),
            name: "my_sample_type".to_string(),
            rc: true,
            weak: false,
            ctrl_tag: None,
            ondisk_size: 64,
            body: RttiBody::Struct(
                vec![
                    RttiField {
                        name: "id".to_string(),
                        offset: 16,
                        shape: Shape::Pod { width: 8 },
                    },
                    RttiField {
                        name: "child".to_string(),
                        offset: 24,
                        shape: Shape::Owned(cc("Child")),
                    },
                    RttiField {
                        name: "names".to_string(),
                        offset: 32,
                        shape: Shape::Vec(Box::new(Shape::Pod { width: 1 })),
                    },
                    RttiField {
                        name: "kind".to_string(),
                        offset: 0,
                        shape: Shape::Class {
                            mutable: false,
                            inner: Box::new(Shape::Pod { width: 4 }),
                            value: vec![7u8, 0, 0, 0].into(),
                        },
                    },
                ]
                .into(),
            ),
        }
    }

    fn sample_enum() -> RttiType {
        RttiType {
            tag: cc("Expr"),
            name: "expr_node".to_string(),
            rc: false,
            weak: false,
            ctrl_tag: None,
            ondisk_size: 48,
            body: RttiBody::Enum(RttiEnum {
                disc_width: 2,
                disc_off: 16,
                payload_off: 24,
                variants: vec![
                    RttiVariant {
                        name: "Leaf".to_string(),
                        disc_value: 0,
                        fields: vec![RttiField {
                            name: "value".to_string(),
                            offset: 0,
                            shape: Shape::Pod { width: 8 },
                        }]
                        .into(),
                    },
                    RttiVariant {
                        name: "Node".to_string(),
                        disc_value: 300,
                        fields: vec![
                            RttiField {
                                name: "left".to_string(),
                                offset: 0,
                                shape: Shape::Owned(cc("Expr")),
                            },
                            RttiField {
                                name: "right".to_string(),
                                offset: 8,
                                shape: Shape::Owned(cc("Expr")),
                            },
                        ]
                        .into(),
                    },
                ]
                .into(),
            }),
        }
    }

    #[test]
    fn rtti_codec_struct_roundtrips() {
        let ty = sample_struct();
        let body = encode_type(&ty).unwrap();
        let back = decode_type(ty.tag, &body).unwrap();
        assert_eq!(ty, back);
    }

    #[test]
    fn rtti_codec_enum_roundtrips() {
        let ty = sample_enum();
        let body = encode_type(&ty).unwrap();
        let back = decode_type(ty.tag, &body).unwrap();
        assert_eq!(ty, back);
    }

    // Two types registered into the global slice via linkme, standing in for what
    // the `#[bstack_class]` macro will emit. `sync_compiled` must discover both.
    fn reg_pair_a() -> RttiType {
        RttiType {
            tag: cc("SyncRegA"),
            name: "sync_reg_a".to_string(),
            rc: false,
            weak: false,
            ctrl_tag: None,
            ondisk_size: 16,
            body: RttiBody::Struct(
                vec![RttiField {
                    name: "x".to_string(),
                    offset: 0,
                    shape: Shape::Pod { width: 8 },
                }]
                .into(),
            ),
        }
    }

    fn reg_pair_b() -> RttiType {
        RttiType {
            tag: cc("SyncRegB"),
            name: "sync_reg_b".to_string(),
            rc: false,
            weak: false,
            ctrl_tag: None,
            ondisk_size: 24,
            body: RttiBody::Struct(
                vec![RttiField {
                    name: "child".to_string(),
                    offset: 0,
                    shape: Shape::Owned(cc("SyncRegA")),
                }]
                .into(),
            ),
        }
    }

    #[distributed_slice(RTTI_TYPES)]
    static REG_A: RttiRegistration = RttiRegistration { build: reg_pair_a };
    #[distributed_slice(RTTI_TYPES)]
    static REG_B: RttiRegistration = RttiRegistration { build: reg_pair_b };

    #[test]
    fn rtti_sync_registers_compiled_types() {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_rtti_sync_{}.stack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        {
            let mut reg = RttiRegistry::open(&path).unwrap();
            // First sync appends at least our two registered types.
            let n = reg.sync_compiled().unwrap();
            assert!(n >= 2, "expected the two registered types, appended {n}");
            assert!(reg.ordinal_of(cc("SyncRegA")).is_some());
            assert!(reg.ordinal_of(cc("SyncRegB")).is_some());
            assert_eq!(
                reg.load_type(reg.ordinal_of(cc("SyncRegA")).unwrap())
                    .unwrap(),
                reg_pair_a()
            );
            // Idempotent: a second sync appends nothing.
            assert_eq!(reg.sync_compiled().unwrap(), 0);
        }

        // Reopen and re-sync: the on-disk types are recognized, nothing re-appended.
        {
            let mut reg = RttiRegistry::open(&path).unwrap();
            assert_eq!(reg.sync_compiled().unwrap(), 0);
            assert!(reg.ordinal_of(cc("SyncRegB")).is_some());
        }

        // The free `sync(path)` entry point is equivalent.
        {
            let reg = sync(&path).unwrap();
            assert!(reg.ordinal_of(cc("SyncRegA")).is_some());
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rtti_registry_append_load_and_reopen() {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_rtti_{}.stack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let a = sample_struct();
        let b = sample_enum();

        {
            let mut reg = RttiRegistry::open(&path).unwrap();
            assert!(reg.is_empty());
            assert_eq!(reg.append(&a).unwrap(), 0);
            assert_eq!(reg.append(&b).unwrap(), 1);
            // duplicate tag is rejected
            assert!(reg.append(&a).is_err());
            // read back live from the stack
            assert_eq!(reg.load_type(0).unwrap(), a);
            assert_eq!(reg.load_type(1).unwrap(), b);
            // typed pointer resolves through the ordinal
            let ord = reg.ordinal_of(b.tag).unwrap();
            assert_eq!(reg.resolve_ptr(typed_ptr(0, 4096, ord)), Some(1));
            assert_eq!(reg.resolve_ptr(WidePtr::from_raw(0, 0, 4096)), None); // untyped
        }

        // reopen: the scan rebuilds the same maps from disk
        {
            let reg = RttiRegistry::open(&path).unwrap();
            assert_eq!(reg.len(), 2);
            assert_eq!(reg.tag_of(0), Some(a.tag));
            assert_eq!(reg.load_type(0).unwrap(), a);
            assert_eq!(reg.load_type(1).unwrap(), b);
        }

        std::fs::remove_file(&path).ok();
    }

    // A persisted enum record with `disc_width > 8` must be rejected on
    // load, not later slice an 8-byte discriminant buffer out of bounds.
    #[test]
    fn decode_rejects_oversize_enum_discriminant_width() {
        let mut body = encode_type(&sample_enum()).unwrap();
        // Sanity: a well-formed enum record round-trips.
        assert!(decode_type(sample_enum().tag, &body).is_ok());
        // Header layout is `[flags u8][disc_width u8]…`; corrupt the width to 9.
        body[1] = 9;
        let err = decode_type(sample_enum().tag, &body).unwrap_err();
        assert!(err.to_string().contains("[BSTACK0816]"), "got: {err}");
    }

    // A `CLASS` field whose encoded `value_len` runs past the record
    // body must be rejected, so `set_class_value` never writes into a neighboring record.
    #[test]
    fn class_value_slot_rejects_value_len_past_body() {
        // Build a struct body with one mutable `CLASS` field "cv", exactly as the walker
        // parses it: header, then `offset/fname_len/shape_len/fname`, then the shape.
        fn body_with_value_len(value_len: u32, value: &[u8]) -> Vec<u8> {
            let mut w = Writer { buf: Vec::new() };
            w.u8(0); // flags: struct (not enum), not rc/weak
            w.u8(0); // disc_width
            w.u16(2); // name_len ("ty")
            w.u16(1); // count = 1 field
            w.u16(0); // disc_off
            w.u16(0); // payload_off
            w.u64(0); // ondisk_size
            w.eightcc(EightCC([0u8; 8])); // ctrl_tag
            w.bytes(b"ty");
            w.align(8);
            // field header
            w.u32(0); // offset
            w.u16(2); // fname_len ("cv")
            w.u16(0); // shape_len (unused for the target field)
            w.bytes(b"cv");
            w.align(4);
            // CLASS shape: tag, mutable, inner POD{width}, value_len, value bytes
            w.u8(shape_tag::CLASS);
            w.u8(1); // mutable
            w.u8(shape_tag::POD);
            w.u32(value.len() as u32);
            w.u32(value_len); // the (possibly corrupt) declared value length
            w.bytes(value);
            w.buf
        }

        // Well-formed: value_len matches the trailing bytes → located.
        let ok = body_with_value_len(5, &[0xC5; 5]);
        let (_, len, mutable) = class_value_slot(&ok, "cv").unwrap().unwrap();
        assert_eq!((len, mutable), (5, true));

        // Corrupt: value_len far exceeds the body → rejected, not returned as a slot.
        let bad = body_with_value_len(u32::MAX, &[0xC5; 5]);
        let err = class_value_slot(&bad, "cv").unwrap_err();
        assert!(err.to_string().contains("[BSTACK0812]"), "got: {err}");
    }

    // Encode-side lengths are written into fixed-width fields (u8 tuple
    // arity, u16 name/count, u32 value/body). A component that overflows its field must
    // be rejected on encode, not silently truncated into an unreadable record.
    #[test]
    fn encode_rejects_components_overflowing_their_length_fields() {
        fn ty_with(name: String, shape: Shape) -> RttiType {
            RttiType {
                tag: cc("Big"),
                name,
                rc: false,
                weak: false,
                ctrl_tag: None,
                ondisk_size: 8,
                body: RttiBody::Struct(
                    vec![RttiField {
                        name: "f".to_string(),
                        offset: 0,
                        shape,
                    }]
                    .into(),
                ),
            }
        }

        // Tuple arity is a `u8`: 256 elements overflow → rejected.
        let over = ty_with(
            "t".to_string(),
            Shape::Tuple(vec![Shape::Pod { width: 1 }; 256].into()),
        );
        assert!(
            encode_type(&over)
                .unwrap_err()
                .to_string()
                .contains("[BSTACK0817]"),
        );
        // 255 elements fit the `u8` arity → fine.
        let ok = ty_with(
            "t".to_string(),
            Shape::Tuple(vec![Shape::Pod { width: 1 }; 255].into()),
        );
        assert!(encode_type(&ok).is_ok());

        // A type name longer than the `u16` name-length field is rejected.
        let long = ty_with("x".repeat(70_000), Shape::Pod { width: 8 });
        assert!(
            encode_type(&long)
                .unwrap_err()
                .to_string()
                .contains("[BSTACK0817]"),
        );
    }

    // `Shape::decode` recurses one frame per nesting tag over untrusted
    // bytes; a deeply-nested corrupt record must be rejected (not overflow the stack).
    #[test]
    fn shape_decode_rejects_excessive_nesting() {
        // A legal nesting depth decodes fine: `Option^10<Pod>`.
        let mut ok = vec![shape_tag::OPTION; 10];
        ok.push(shape_tag::POD);
        ok.extend_from_slice(&8u32.to_le_bytes());
        assert!(Shape::decode(&mut Reader::new(&ok)).is_ok());

        // Past the bound the decode is refused before recursing further. The `Pod` leaf
        // is well-formed, so only the depth guard (not truncation) can reject this.
        let mut deep = vec![shape_tag::OPTION; MAX_SHAPE_DEPTH + 5];
        deep.push(shape_tag::POD);
        deep.extend_from_slice(&8u32.to_le_bytes());
        let err = Shape::decode(&mut Reader::new(&deep)).unwrap_err();
        assert!(err.to_string().contains("[BSTACK0818]"), "got: {err}");
    }
}
