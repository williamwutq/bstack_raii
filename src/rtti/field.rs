//! Path-addressed **field access** — reach one field by a `["outer", "inner", …]`
//! path and read, overwrite, or exchange it in place ([`get`](RttiRegistry::get) /
//! [`set`](RttiRegistry::set) / [`swap`](RttiRegistry::swap) /
//! [`swap_foreign`](RttiRegistry::swap_foreign)), plus the `resolve_field` navigator.

use bstack::BStack;

use crate::primitives::{EightCC, Offset, WidePtr};
use crate::registry::FileId;
use crate::types::compiled::rc::{CTRL_BACKPTR_OFFSET, CTRL_DATA_OFFSET};
use crate::util::{get_u64, read_u64};

use super::read::Op;
use super::walk::{disc_mask, read_disc, verify_data_block};
use super::{
    AnyRef, FOREIGN_REPR_LEN, ForeignPtr, HEADER_TAG_OFFSET, Resolved, RttiBody, RttiField,
    RttiOrdinal, RttiRegistry, Shape, Value, add_off, ordinal_type_index, unknown_tag,
};
use super::{RttiResult, rtti_err};

impl RttiRegistry {
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
    ) -> RttiResult<Resolved> {
        if path.is_empty() {
            return Err(rtti_err!(Set, "RTTI set: empty field path"));
        }
        let mut ord = ordinal;
        // Carry the descent offset as a branded `Offset` so each step's arithmetic is
        // overflow-checked. It stays *nullable*: a top-level class-variable path
        // (`block_off == 0`, the class var living in the schema, not the instance) never
        // dereferences `base`, so `0` is a legal sentinel here — a follow re-checks
        // non-null explicitly below before reading through it.
        let mut base = Offset::from_raw(block_off);
        for (i, seg) in path.iter().enumerate() {
            let ty = self.load_type(ord)?;
            // A struct's fields are block-relative; an enum's active variant's fields
            // are payload-relative.
            let (fields, field_base): (&[RttiField], Offset) = match &ty.body {
                RttiBody::Struct(f) => (f, base),
                RttiBody::Enum(e) => {
                    let raw = read_disc(
                        data,
                        base.checked_add(e.disc_off as u64)?.get(),
                        e.disc_width,
                    )?;
                    let mask = disc_mask(e.disc_width);
                    let variant = e
                        .variants
                        .iter()
                        .find(|v| (v.disc_value as u64) & mask == raw)
                        .ok_or_else(|| {
                            rtti_err!(Set, "RTTI set: no variant for discriminant {}", raw)
                        })?;
                    (&variant.fields, base.checked_add(e.payload_off as u64)?)
                }
            };
            let field = fields
                .iter()
                .find(|f| &f.name == seg)
                .ok_or_else(|| rtti_err!(Set, "RTTI set: no field named `{}`", seg))?;
            let field_off = field_base.checked_add(field.offset as u64)?;

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
                    offset: field_off.get(),
                    shape: field.shape.clone(),
                });
            }
            // Descend into a block reference for the next segment.
            match &field.shape {
                Shape::Owned(tag) | Shape::Strong(tag) | Shape::Ref(tag) => {
                    // The stored offset points at the child block; a null is a dangling
                    // path. Check non-null here (the deref happens next iteration).
                    let child = Offset::from_raw(read_u64(data, field_off.get())?);
                    if child.is_null() {
                        return Err(rtti_err!(Set, "RTTI set: null reference at `{}`", seg));
                    }
                    base = child;
                    ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                }
                Shape::Embed(tag) => {
                    ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                    base = field_off;
                }
                _ => {
                    return Err(rtti_err!(
                        Set,
                        "RTTI set: cannot descend through non-block field `{}`",
                        seg
                    ));
                }
            }
        }
        unreachable!("the last segment returns inside the loop")
    }

    /// Validate that `block_off` names a live block of type `ordinal` — the precondition
    /// the **safe** [`swap`](Self::swap) / [`swap_foreign`](Self::swap_foreign) need but
    /// that [`resolve_field`](Self::resolve_field) does not itself check. Those two do a
    /// raw write at `block_off + field.offset` and hand the displaced slot back as an
    /// *owning* [`AnyRef`] / [`ForeignPtr`]; with a wrong or null `block_off` the write
    /// lands at an arbitrary in-file location and the handed-back reference is fabricated
    /// over arbitrary bytes, whose later safe `bstack_drop` would free storage the caller
    /// does not own. (`set` guards the same base with its `unsafe` contract instead — it
    /// has a *further* obligation validation can't cover: a POD image may overwrite
    /// invariant-bearing bytes even at a correct base.) Unlike [`verify_data_block`], a
    /// null base is rejected here: a live instance never sits at offset 0.
    fn verify_block_of(&self, data: &BStack, block_off: u64, ordinal: RttiOrdinal) -> RttiResult<()> {
        let off = Offset::from_raw(block_off);
        if off.is_null() {
            return Err(rtti_err!(
                Mutator,
                "RTTI mutator: block offset 0 does not name a live block"
            ));
        }
        let tag = self.load_type(ordinal)?.tag;
        verify_data_block(data, off, tag)
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
    ) -> RttiResult<Value> {
        match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => {
                self.run_read(data, Op::Shape { shape, offset })
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
    ) -> RttiResult<()> {
        let (offset, shape) = match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => (offset, shape),
            // A class variable is schema-side; write it in place there.
            Resolved::Class { tag, name } => return self.set_class_value(tag, &name, value),
        };
        match shape {
            Shape::Pod { width } => {
                if value.len() != width as usize {
                    return Err(rtti_err!(
                        Set,
                        "RTTI set: field is {} bytes, got {}",
                        width,
                        value.len()
                    ));
                }
            }
            // A `ref` is a bare `u64` target offset (a non-owning alias).
            Shape::Ref(t) => {
                if value.len() != 8 {
                    return Err(rtti_err!(
                        Set,
                        "RTTI set: a `ref` field is an 8-byte offset, got {}",
                        value.len()
                    ));
                }
                // Validate the offset names a live block of the ref's type — an
                // unchecked offset would let a later path descend into an arbitrary
                // in-file location.
                let target = get_u64(value);
                verify_data_block(data, Offset::from_raw(target), t)?;
            }
            _ => {
                return Err(rtti_err!(
                    Set,
                    "RTTI set: field is not POD / `ref` / class variable; an owning \
                     reference is `swap`ped, not set"
                ));
            }
        }
        data.set(offset, value).map_err(Into::into)
    }

    /// **Swap** the **owning** reference field named by `path` to point at `new`,
    /// returning the previous target as an [`AnyRef`] (`None` if it was null). A pointer
    /// exchange: the field takes ownership of `new`, and the old reference is handed back
    /// for the caller to reuse or tear down — no refcount changes (ownership moves, it is
    /// not duplicated).
    ///
    /// `new` is **validated** against the on-disk header before it is installed: a live
    /// block of the field's type must sit at its offset (for a `weak` field, at the
    /// control block's forward data pointer). This keeps a fabricated [`AnyRef`] from
    /// pointing an owning slot at an arbitrary location — rejected with `[BSTACK0815]`.
    ///
    /// `new`'s [`tag`](AnyRef::tag) **must equal the field's declared type** (an
    /// eightcc mismatch is rejected), and the target must be an **owning** in-file
    /// reference (`owned` / `strong` / `weak`, optionally `Option`-wrapped). For a `weak`
    /// field, `new` and the returned old reference are the target's **control-block**
    /// [`AnyRef`] (exactly what [`move_out`](Self::move_out) hands back). A POD field or
    /// a container is rejected; a **`ref`** field is a non-owning alias (swapping it
    /// would hand back a tear-down-able alias to a block it does not own) — repoint it
    /// with [`set`](Self::set) instead; a cross-file `foreign` field uses
    /// [`swap_foreign`](Self::swap_foreign) instead (an [`AnyRef`] can't name its file).
    pub fn swap(
        &self,
        data: &BStack,
        ordinal: RttiOrdinal,
        block_off: u64,
        path: &[&str],
        new: AnyRef,
    ) -> RttiResult<Option<AnyRef>> {
        // `block_off` is caller-supplied; validate it names a live block of `ordinal`
        // before resolving, so the raw write and the owning-`AnyRef` handback below
        // cannot be aimed at an arbitrary in-file location (see `verify_block_of`).
        self.verify_block_of(data, block_off, ordinal)?;
        let (offset, mut shape) = match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => (offset, shape),
            Resolved::Class { .. } => {
                return Err(rtti_err!(
                    Swap,
                    "RTTI swap: a class variable is a value, not a reference — use `set`"
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
            // `owned`/`strong` hold a data offset; `weak` holds a control offset — both
            // are a single `u64` slot exchanged the same way (no refcount change). Each
            // owns something the swap transfers out, so the displaced `AnyRef` is the
            // caller's to keep or tear down.
            Shape::Owned(t) | Shape::Strong(t) => (t, false),
            Shape::Weak(t) => (t, true),
            // A `ref` owns nothing — it aliases a block some *other* slot owns. Swapping
            // it would hand the displaced offset back as an owning `AnyRef`, which a
            // caller could tear down and double-free the block its real owner still
            // holds. The reference kind is edge metadata (the field's `Shape`), not a
            // fact about the block, so an isolated `AnyRef` cannot carry it — which is
            // exactly why `swap`, an ownership transfer, does not accept a `ref`. Repoint
            // one in place with `set` (same validation, returns nothing); a typed,
            // non-owning handle to the old target is the static `replace_<field>`'s
            // `BStackRef` return, which the untyped RTTI API cannot reproduce.
            Shape::Ref(_) => {
                return Err(rtti_err!(
                    Swap,
                    "RTTI swap: a `ref` field is a non-owning alias — repoint it with \
                     `set`; `swap` exchanges an owning reference (`owned` / `strong` / \
                     `weak`)"
                ));
            }
            Shape::Foreign { .. } => {
                return Err(rtti_err!(
                    Swap,
                    "RTTI swap: a `foreign` reference names a cross-file target — use \
                     `swap_foreign`"
                ));
            }
            _ => {
                return Err(rtti_err!(
                    Swap,
                    "RTTI swap: field is not a swappable reference"
                ));
            }
        };
        if new.tag() != tag {
            return Err(rtti_err!(
                Swap,
                "RTTI swap: eightcc mismatch: `new` is not the field's type"
            ));
        }
        // The offset being installed — bound once as an `Offset` so the null-niche test
        // and the validation arithmetic below share one branded value.
        let new_off = Offset::from_raw(new.offset());
        // A non-nullable slot must never hold the `0` niche: the generated walks
        // treat non-nullable as proof of non-null, so installing a null here would
        // persist a handle over offset 0 and derail every later read / teardown.
        if new_off.is_null() && !nullable {
            return Err(rtti_err!(
                Mutator,
                "RTTI mutator: a null reference cannot be installed into a non-nullable field"
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
            if let Some(new_off) = new_off.to_non_null() {
                // (1) Direct: the control block's header tag. Enforced whenever the
                // target type's schema records a control tag (every `weak` type does);
                // if unresolvable, fall back to the forward-pointer check below.
                let ctrl_tag = self
                    .ordinal_of(tag)
                    .and_then(|ord| self.load_type(ord).ok())
                    .and_then(|t| t.ctrl_tag);
                if let Some(ctrl_tag) = ctrl_tag {
                    let mut hdr = [0u8; 8];
                    data.get_into(new_off.checked_add(HEADER_TAG_OFFSET)?.as_u64(), &mut hdr)?;
                    if EightCC(hdr) != ctrl_tag {
                        return Err(rtti_err!(
                            Mutator,
                            "RTTI mutator: offset {} does not hold a live control block of \
                             the target type (its header tag is not the type's control tag)",
                            new_off.as_u64()
                        ));
                    }
                }
                // (2) Forward data pointer + backpointer round-trip.
                let data_ptr = read_u64(data, new_off.checked_add(CTRL_DATA_OFFSET)?.as_u64())?;
                verify_data_block(data, Offset::from_raw(data_ptr), tag)?;
                let backptr = read_u64(data, add_off(data_ptr, CTRL_BACKPTR_OFFSET)?)?;
                if backptr != new_off.as_u64() {
                    return Err(rtti_err!(
                        Mutator,
                        "RTTI mutator: offset {} is not the target's control block \
                         (its backpointer names {backptr})",
                        new_off.as_u64()
                    ));
                }
            }
        } else {
            verify_data_block(data, new_off, tag)?;
        }
        // Atomic exchange: install the new offset and take the displaced one in one
        // locked step, so concurrent callers each get the distinct old target they
        // displaced — never both hand back an owning `AnyRef` to the same block.
        let old_bytes = data.swap(offset, new_off.get().to_le_bytes())?;
        let old = u64::from_le_bytes(old_bytes[..8].try_into().unwrap());
        // SAFETY: `block_off` was validated (`verify_block_of`) to name a live block of
        // `ordinal`, so `old` was displaced from that block's own field slot, which held
        // a live target of the field's declared (schema-resolved) tag.
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
    ) -> RttiResult<Option<ForeignPtr>> {
        // `block_off` is caller-supplied; validate it names a live block of `ordinal`
        // before resolving, so the raw 16-byte write and the owning-`ForeignPtr`
        // handback below cannot be aimed at an arbitrary in-file location.
        self.verify_block_of(data, block_off, ordinal)?;
        let (offset, mut shape) = match self.resolve_field(data, ordinal, block_off, path)? {
            Resolved::Instance { offset, shape } => (offset, shape),
            Resolved::Class { .. } => {
                return Err(rtti_err!(
                    Swap,
                    "RTTI swap: a class variable is a value, not a reference — use `set`"
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
                return Err(rtti_err!(
                    Swap,
                    "RTTI swap: field is not a `foreign` reference — use `swap` for \
                     in-file references"
                ));
            }
        };
        if new.tag != tag {
            return Err(rtti_err!(
                Swap,
                "RTTI swap: eightcc mismatch: `new` is not the field's foreign target type"
            ));
        }
        // As `swap`: a non-nullable slot must never hold the null niche.
        if new.offset == 0 && !nullable {
            return Err(rtti_err!(
                Mutator,
                "RTTI mutator: a null foreign reference cannot be installed into a \
                 non-nullable field"
            ));
        }
        // Validate the new target names a live block of the field's type in its own
        // file before installing the raw pointer — an unchecked `(file_id, offset)`
        // would let a later cross-file teardown free an arbitrary range in that file.
        if new.offset != 0 {
            let fid = FileId::from_u64(new.file_id)
                .ok_or_else(|| rtti_err!(Swap, "RTTI swap: invalid foreign file id in `new`"))?;
            if fid.is_self() {
                verify_data_block(data, Offset::from_raw(new.offset), new.tag)?;
            } else {
                crate::registry::with_host(fid, |h| {
                    verify_data_block(h.stack(), Offset::from_raw(new.offset), new.tag)
                })
                .ok_or_else(|| {
                    rtti_err!(
                        Swap,
                        "RTTI swap: the new target's file is not attached — cannot \
                         validate the pointer"
                    )
                })??;
            }
        }
        // Build the new 16-byte `WidePtr { file_id:u32, type_index:u32, offset:u64 }`
        // (type_index = the target's ordinal + 1, per `typed_ptr`).
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        let mut b = [0u8; FOREIGN_REPR_LEN as usize];
        b[0..4].copy_from_slice(&(new.file_id as u32).to_le_bytes());
        b[4..8].copy_from_slice(&ordinal_type_index(ord).to_le_bytes());
        b[8..16].copy_from_slice(&new.offset.to_le_bytes());
        // Atomic exchange of the whole 16-byte pointer, taking the old one this
        // caller displaced.
        let old_repr = data.swap(offset, b)?;
        let __wp = WidePtr::decode(&old_repr);
        Ok((!__wp.is_null()).then_some(ForeignPtr {
            tag,
            kind,
            file_id: __wp.file_id(),
            offset: __wp.offset().get(),
        }))
    }
}
