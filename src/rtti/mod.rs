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

use std::io;

use crate::primitives::WidePtr;
use crate::types::compiled::rc::CTRL_DATA_OFFSET;
use crate::util::io_error;

mod clone;
mod field;
mod r#move;
mod read;
mod registry;
mod schema;
mod teardown;
mod value;
mod walk;

pub use registry::{RTTI_TYPES, RttiRegistration, RttiRegistry, sync};
pub use schema::{
    RttiBody, RttiEnum, RttiField, RttiType, RttiVariant, Shape, decode_type, encode_type,
};
pub(in crate::rtti) use schema::{class_value_slot, frame_record, layouts_match};
pub(in crate::rtti) use value::Resolved;
pub use value::{AnyRef, ForeignPtr, Moved, Value, VecRef};
// `class_error` lives with the other shared helpers in `walk`; re-export it so
// `schema` / `registry` reach it as `super::class_error` like the rest.
pub(in crate::rtti) use walk::class_error;

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
pub(in crate::rtti) const RECORD_HEADER_LEN: u64 = 16;

// Maybe a DiskOffset or something could make our code more maintainable
// We can also make DiskOffset behave instead like NonZeroU64, due to the null niche
// requirement that is generally applied in this crate
/// Add two on-disk offsets/lengths, rejecting overflow. Every interpreter walk
/// (`read_value` / `teardown` / `clone_value`) chains additions off a **root**
/// offset that can be entirely attacker/caller-controlled (a forged pointer, or
/// — as here — a fuzzed argument); an unchecked `+` either panics under
/// `overflow-checks` or silently wraps to an unrelated in-bounds offset in a
/// release build. Reject cleanly instead.
pub(in crate::rtti) fn add_off(a: u64, b: u64) -> io::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| io_error!(InvalidData, "[BSTACK081A] RTTI offset arithmetic overflow"))
}

/// Multiply an on-disk element stride by an index, rejecting overflow — the
/// `mul` counterpart of [`add_off`] for `Array`/`Vec` element offsets.
pub(in crate::rtti) fn mul_off(a: u64, b: u64) -> io::Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| io_error!(InvalidData, "[BSTACK081A] RTTI offset arithmetic overflow"))
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
pub(in crate::rtti) fn too_large(what: &str, limit: &str) -> io::Error {
    io_error!(
        InvalidData,
        format!("[BSTACK0817] RTTI {what} exceeds the maximum encodable size ({limit})")
    )
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

// -- Parsed, in-memory schema (structure only) -----------------------------

/// The four ownership kinds an interpreted reference can carry — re-exported from
/// [the primitives](crate::primitives::OwnershipKind), the crate-wide vocabulary.
/// A `Foreign` leaf, a struct field, and a variant payload all classify with it.
pub use crate::primitives::OwnershipKind;

/// Bytes of a `VecDesc` (`data_off:u64` @0, `data_size:u64` @8) — the inline
/// descriptor of a persistent vector.
pub(in crate::rtti) const VECDESC_LEN: u64 = 16;
/// A byte-vec data block's header (`len:u64` @0, `cap:u64` @8, elements from 16).
pub(in crate::rtti) const BYTEVEC_HEADER: u64 = 16;
/// Bytes of a `WidePtr` on the wire.
pub(in crate::rtti) const FOREIGN_REPR_LEN: u64 = 16;
/// Offset of the `tag: EightCC` within a block's `BlockHeader` (`size: u64` @0).
pub(in crate::rtti) const HEADER_TAG_OFFSET: u64 = 8;
/// Bytes of an `(rc, weak)` control block (`XOnDiskRef`): a 16-byte header, then the
/// `strong`, `weak`, and data-back-pointer `u64`s. Fixed for every weakable type (the
/// control layout does not depend on `T`).
pub(in crate::rtti) const CONTROL_SIZE: u64 = CTRL_DATA_OFFSET + 8;

pub(in crate::rtti) fn unknown_tag() -> io::Error {
    io_error!(
        InvalidData,
        "[BSTACK080B] RTTI pointer/field references an unregistered type tag"
    )
}

#[cfg(test)]
mod tests {
    use linkme::distributed_slice;

    use super::*;
    use crate::primitives::EightCC;

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
}
