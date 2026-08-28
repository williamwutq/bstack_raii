//! The in-memory **registry** and link-time registration — the RTTI analog of
//! `io_core::registry`: the scanned schema stack every lookup resolves against, plus
//! the `linkme`-collected compiled-in type set and the [`sync`] entry point.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use bstack::BStack;
use linkme::distributed_slice;

use crate::primitives::{EightCC, WidePtr};
use crate::util::io_error;

use super::{
    RECORD_HEADER_LEN, RttiOrdinal, RttiType, class_error, class_value_slot, decode_type,
    encode_type, frame_record, layouts_match, unknown_tag,
};

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
                return Err(io_error!(
                    InvalidData,
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
            return Err(io_error!(
                InvalidData,
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
            return Err(io_error!(
                InvalidData,
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
                    return Err(io_error!(
                        InvalidData,
                        format!(
                            "[BSTACK0806] RTTI eightcc collision: '{}' and '{}' \
                         hash to one tag",
                            prev.name, ty.name
                        )
                    ));
                }
                // Same tag AND same name is still a collision when the layouts
                // differ — the tag ignores the module path, so `v1::Node` and
                // `v2::Node` arrive here as one name. Only a byte-identical
                // layout is genuinely "the same type registered twice".
                if !layouts_match(prev, &ty) {
                    return Err(io_error!(
                        InvalidData,
                        format!(
                            "[BSTACK0806] RTTI eightcc collision: two distinct types \
                         both named '{}' (same-named types in different modules?) \
                         share one tag",
                            ty.name
                        )
                    ));
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
                        return Err(io_error!(
                            InvalidData,
                            format!(
                                "[BSTACK0806] RTTI eightcc collision: on-disk '{}' vs \
                             compiled '{}' share one tag",
                                existing.name, ty.name
                            )
                        ));
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
                        return Err(io_error!(
                            InvalidData,
                            format!(
                                "[BSTACK0814] RTTI schema mismatch for '{}': the persisted \
                             layout differs from the compiled type (a field was added, \
                             removed, reordered, or resized). The on-disk data was \
                             written against the old layout.",
                                ty.name
                            )
                        ));
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
        let rec = self
            .records
            .get(ordinal as usize)
            .ok_or_else(|| io_error!(NotFound, "[BSTACK0801] RTTI ordinal out of range"))?;
        // NOTE: is rec.body_len bounded and will not result in dangerous allocations?
        // check similar patterns
        let mut body = vec![0u8; rec.body_len as usize];
        self.stack
            .get_into(rec.offset + RECORD_HEADER_LEN, &mut body)?;
        decode_type(rec.tag, &body)
    }

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
