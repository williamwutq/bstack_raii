//! The parsed, on-disk **schema vocabulary** and its little-endian wire codec — the
//! RTTI analog of the crate's `types` layer plus its serialization: the [`Shape`]
//! grammar and the [`RttiType`] descriptor, their record encode/decode, the typed
//! reads over the generic [`Reader`](crate::util::Reader) cursor, and the schema-side
//! `layouts_match` / `class_value_slot` helpers.

use std::io;

use crate::primitives::{EightCC, OwnershipKind};
use crate::util::{Reader, Writer, io_error};

use super::{class_error, too_large};

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

// -- RTTI-framed reads over the generic byte cursor --------------------------
// The cursor ([`Reader`]) is a pure, error-free primitive: a short read is a `None`,
// a failed alignment a `false`. These free wrappers add the little-endian typed reads
// and map that failure to the RTTI truncation error, keeping the domain vocabulary
// here at the caller rather than baked into `util`.

/// Map a byte-cursor underrun (a `None` from [`Reader::take`](crate::util::Reader::take))
/// to the RTTI truncation error.
fn need<T>(v: Option<T>) -> io::Result<T> {
    v.ok_or_else(|| io_error!("[BSTACK0804] truncated RTTI record"))
}

#[inline(always)]
fn u8(r: &mut Reader) -> io::Result<u8> {
    Ok(need(r.take(1))?[0])
}

#[inline(always)]
fn u16(r: &mut Reader) -> io::Result<u16> {
    Ok(u16::from_le_bytes(need(r.take(2))?.try_into().unwrap()))
}

#[inline(always)]
fn u32(r: &mut Reader) -> io::Result<u32> {
    Ok(u32::from_le_bytes(need(r.take(4))?.try_into().unwrap()))
}

#[inline(always)]
fn u64(r: &mut Reader) -> io::Result<u64> {
    Ok(u64::from_le_bytes(need(r.take(8))?.try_into().unwrap()))
}

#[inline(always)]
fn i64(r: &mut Reader) -> io::Result<i64> {
    Ok(i64::from_le_bytes(need(r.take(8))?.try_into().unwrap()))
}

#[inline(always)]
fn eightcc(r: &mut Reader) -> io::Result<EightCC> {
    Ok(EightCC(need(r.take(8))?.try_into().unwrap()))
}

/// Read an `n`-byte UTF-8 string, RTTI-framing both a short read and invalid UTF-8.
fn string(r: &mut Reader, n: usize) -> io::Result<String> {
    String::from_utf8(need(r.take(n))?.to_vec())
        .map_err(|_| io_error!("[BSTACK0802] RTTI name is not valid UTF-8"))
}

/// Advance `r` past zero-padding to the next `a`-byte boundary, RTTI-framing a boundary
/// that runs past the record. The bounds check lives in the generic
/// [`Reader::skip_pad`](crate::util::Reader::skip_pad) (which reports fit as a `bool`);
/// the domain error is applied here, at the caller.
fn align(r: &mut Reader, a: usize) -> io::Result<()> {
    if r.skip_pad(a) {
        Ok(())
    } else {
        Err(io_error!(
            InvalidData,
            "[BSTACK0804] truncated RTTI record (alignment)"
        ))
    }
}

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
            return Err(io_error!(
                InvalidData,
                "[BSTACK0818] RTTI shape nesting exceeds the maximum depth",
            ));
        }
        let tag = u8(r)?;
        Ok(match tag {
            t::POD => Shape::Pod { width: u32(r)? },
            t::OWNED => Shape::Owned(eightcc(r)?),
            t::STRONG => Shape::Strong(eightcc(r)?),
            t::WEAK => Shape::Weak(eightcc(r)?),
            t::REF => Shape::Ref(eightcc(r)?),
            t::EMBED => Shape::Embed(eightcc(r)?),
            t::FOREIGN => {
                let tag = eightcc(r)?;
                let kb = u8(r)?;
                let kind = OwnershipKind::from_u8(kb).ok_or_else(|| {
                    io_error!("[BSTACK0803] unknown RTTI foreign kind {:#04x}", kb)
                })?;
                Shape::Foreign { tag, kind }
            }
            t::OPTION => Shape::Option(Box::new(Shape::decode_at(r, depth + 1)?)),
            t::ARRAY => {
                let n = u32(r)?;
                Shape::Array {
                    n,
                    inner: Box::new(Shape::decode_at(r, depth + 1)?),
                }
            }
            t::VEC => Shape::Vec(Box::new(Shape::decode_at(r, depth + 1)?)),
            t::TUPLE => {
                let k = u8(r)? as usize;
                let items = (0..k)
                    .map(|_| Shape::decode_at(r, depth + 1))
                    .collect::<Result<Box<[_]>, _>>()?;
                Shape::Tuple(items)
            }
            t::CLASS => {
                let mutable = u8(r)? != 0;
                let inner = Box::new(Shape::decode_at(r, depth + 1)?);
                let value_len = u32(r)? as usize;
                let value = need(r.take(value_len))?.into();
                Shape::Class {
                    mutable,
                    inner,
                    value,
                }
            }
            other => {
                return Err(io_error!(
                    InvalidData,
                    "[BSTACK0803] unknown RTTI shape tag {:#04x}",
                    other
                ));
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
        let offset = u32(r)?;
        let name_len = u16(r)? as usize;
        let shape_len = u16(r)? as usize;
        let name = string(r, name_len)?;
        align(r, 4)?;
        let shape_start = r.pos;
        let shape = Shape::decode(r)?;
        if r.pos - shape_start != shape_len {
            return Err(io_error!(
                InvalidData,
                "[BSTACK0805] RTTI field shape length mismatch"
            ));
        }
        align(r, 4)?;
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
        align(r, 8)?;
        let disc_value = i64(r)?;
        let name_len = u16(r)? as usize;
        let field_count = u16(r)? as usize;
        let _pad = u32(r)?;
        let name = string(r, name_len)?;
        align(r, 8)?;
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
    let flags = u8(&mut r)?;
    let disc_width = u8(&mut r)?;
    let name_len = u16(&mut r)? as usize;
    let count = u16(&mut r)? as usize;
    let disc_off = u16(&mut r)?;
    let payload_off = u16(&mut r)?;
    let ondisk_size = u64(&mut r)?;
    let ctrl_tag_raw = eightcc(&mut r)?;
    let name = string(&mut r, name_len)?;
    align(&mut r, 8)?;
    let weak = flags & FLAG_WEAK != 0;

    let body = if flags & FLAG_ENUM != 0 {
        if disc_width > 8 {
            // A discriminant is read into a `u64`; reject a corrupt wider width on
            // load so no interpreter path later slices past an 8-byte buffer.
            return Err(io_error!(
                InvalidData,
                "[BSTACK0816] RTTI enum discriminant width exceeds 8 bytes",
            ));
        }
        if disc_width == 0 {
            // `disc_mask(0)` is 0 and a 0-byte read yields 0, so every variant
            // search would silently match the first variant; a corrupt record
            // must error, not mis-parse.
            return Err(io_error!(
                InvalidData,
                "[BSTACK0816] RTTI enum discriminant width is zero"
            ));
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
pub(in crate::rtti) fn frame_record(tag: EightCC, body: &[u8]) -> io::Result<(Vec<u8>, u32)> {
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

/// Whether two type descriptors describe the **same layout** — equal in everything a
/// persisted instance depends on, EXCEPT the current *value* of a **mutable** class
/// variable. A mutable `#[bstack_static]` is updated in place (`set_class_value`), so
/// its persisted value legitimately differs from the compiled type's initial value;
/// a raw `existing == ty` would flag that as a schema change. Everything else — field
/// offsets / shapes / order, `rc`/`weak` mode, `ondisk_size`, and *const* class-var
/// values — must match.
pub(in crate::rtti) fn layouts_match(a: &RttiType, b: &RttiType) -> bool {
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

/// Locate the value bytes of the `CLASS` field named `target` within a decoded record
/// body: `(offset within body, value length, mutable)`, or `None` if there is no such
/// (class) field. Walks the `TypeDesc` header + fields exactly as `decode_type` does,
/// stopping at the target instead of building an [`RttiType`].
pub(in crate::rtti) fn class_value_slot(
    body: &[u8],
    target: &str,
) -> io::Result<Option<(usize, usize, bool)>> {
    let mut r = Reader::new(body);
    let flags = u8(&mut r)?;
    let _disc_width = u8(&mut r)?;
    let name_len = u16(&mut r)? as usize;
    let count = u16(&mut r)? as usize;
    let _disc_off = u16(&mut r)?;
    let _payload_off = u16(&mut r)?;
    let _ondisk_size = u64(&mut r)?;
    let _ctrl_tag = eightcc(&mut r)?; // control tag — same fixed-header slot
    let _name = string(&mut r, name_len)?;
    align(&mut r, 8)?;
    // Only structs carry class variables; an enum's `count` is its variants.
    if flags & FLAG_ENUM != 0 {
        return Ok(None);
    }
    for _ in 0..count {
        let _offset = u32(&mut r)?;
        let fname_len = u16(&mut r)? as usize;
        let _shape_len = u16(&mut r)? as usize;
        let fname = string(&mut r, fname_len)?;
        align(&mut r, 4)?;
        if fname == target {
            return class_value_within_shape(&mut r);
        }
        // Skip this field's shape (decode advances the cursor, bounds-checked) + pad.
        let _ = Shape::decode(&mut r)?;
        align(&mut r, 4)?;
    }
    Ok(None)
}

/// If the shape at the cursor is a `CLASS` shape, consume its header and return
/// `(value offset within body, value length, mutable)` with the cursor left at the
/// value bytes; otherwise `None` (the named field is not a class variable).
fn class_value_within_shape(r: &mut Reader) -> io::Result<Option<(usize, usize, bool)>> {
    if u8(r)? != shape_tag::CLASS {
        return Ok(None);
    }
    let mutable = u8(r)? != 0;
    let _inner = Shape::decode(r)?;
    let value_len = u32(r)? as usize;
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

#[cfg(test)]
mod tests {
    use super::*;

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
