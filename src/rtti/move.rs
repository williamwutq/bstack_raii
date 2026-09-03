//! The **move interpreter** — the RTTI `bstack_move!`: disassemble a block into its
//! owned parts (a [`SmallStringMap`]`<`[`Moved`]`>`), freeing only the shell and handing
//! each owned reference back to the caller.

use std::collections::HashMap;

use bstack::{BStack, BStackRange};

use crate::BStackRaiiAllocator;
use crate::io_core::refcount;
use crate::primitives::{NonNullOffset, WidePtr};
use crate::types::compiled::rc::CTRL_WEAK_OFFSET;
use crate::util::{SmallStringMap, read_u64};

use super::walk::strong_counter_slot;
use super::{
    AnyRef, CONTROL_SIZE, FOREIGN_REPR_LEN, ForeignPtr, Moved, RttiBody, RttiField, RttiOrdinal,
    RttiRegistry, RttiType, Shape, VecRef, add_off, mul_off, unknown_tag,
};
use super::{RttiResult, rtti_err};

impl RttiRegistry {
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
    ) -> RttiResult<SmallStringMap<Moved>> {
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
        let (strong_slot, ctrl_off): (Option<NonNullOffset>, Option<NonNullOffset>) = {
            let root = &cache[&ordinal];
            if root.rc {
                // The strong counter slot (inline `rc`, or the control block's `strong`
                // via the data backptr) — branded once, with `ctrl` for the weak release
                // below. `block_off` is the caller-owned live root, hence non-null.
                let (strong_slot, ctrl) =
                    strong_counter_slot(data, root.weak, NonNullOffset::from_field(block_off)?)?;
                // Atomic try-unwrap, exactly as `BStackRc::try_move`: claim sole
                // ownership by CAS `strong: 1 -> 0`, so a concurrent clone/upgrade
                // either beats the move (the CAS fails cleanly) or is refused by
                // the zero count for the whole field walk — never both succeeding.
                if !refcount::cas(data, strong_slot, 1, 0)? {
                    let strong = read_u64(data, strong_slot.as_u64())?;
                    return Err(rtti_err!(
                        SharedMove,
                        "RTTI move_out of a shared reference-counted block \
                         (strong count {}); only the sole owner may disassemble it",
                        strong
                    ));
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
                // keeps its sole owner. `strong_slot` was branded non-null at the CAS, so
                // the restore is unconditional — no re-check that could silently skip it.
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
            && refcount::fetch_sub(data, ctrl_off.checked_add(CTRL_WEAK_OFFSET)?, 1)? == 1
        {
            to_free.push(BStackRange::new(ctrl_off.as_u64(), CONTROL_SIZE));
        }
        // Route through the WAL (or the allocator's atomic bulk free) like the
        // static `bstack_move!` shell teardown, so a crash after the fields moved
        // out but before these frees commit is reclaimed on the next open, not
        // leaked permanently.
        // SAFETY: the shell is the caller-owned root (its fields already moved out);
        // the control block, if included, has no remaining references; both live in
        // this file.
        if let Err(e) = unsafe { crate::io_core::commit_home_frees(alloc, to_free) } {
            // SAFETY: `materialized` are this call's own embed copies.
            let _ = unsafe { alloc.free_many(std::mem::take(&mut materialized)) };
            return Err(e.into());
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
    ) -> RttiResult<SmallStringMap<Moved>> {
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
                    let (variant, payload_base) = e.resolve_variant(data, block_off)?;
                    (variant.fields.clone(), payload_base)
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
                return Err(rtti_err!(
                    Malformed,
                    "RTTI record has two fields named '{}'",
                    f.name
                ));
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
    ) -> RttiResult<Moved> {
        Ok(match shape {
            Shape::Pod { width } => {
                // Untrusted width: bound against the stack before allocating.
                if *width as u64 > data.len()?.saturating_sub(off) {
                    return Err(rtti_err!(
                        Malformed,
                        "RTTI POD width runs past the end of the data stack",
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
                let child = read_u64(data, off)?;
                Moved::Ref((child != 0).then(|| unsafe { AnyRef::new(*tag, child) }))
            }
            Shape::Weak(tag) => {
                let ctrl = read_u64(data, off)?;
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
                let data_off = read_u64(data, off)?; // VecDesc.data_off @0
                if data_off == 0 {
                    Moved::Vec(None)
                } else {
                    Moved::Vec(Some(VecRef {
                        data_off,
                        data_size: read_u64(data, add_off(off, 8)?)?,
                        elem: (**inner).clone(),
                    }))
                }
            }
            Shape::Array { n, inner } => {
                if let Some((tag, kind)) = inner.foreign_leaf() {
                    // A foreign array: each element is a 16-byte `WidePtr` inline
                    // in the shell; hand every cross-file pointer back to the caller.
                    let mut list = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        let __wp = WidePtr::read_from_stack(
                            data,
                            add_off(off, mul_off(i, FOREIGN_REPR_LEN)?)?,
                        )?;
                        let (file_id, offset) = (__wp.file_id(), __wp.offset().get());
                        list.push(ForeignPtr {
                            tag,
                            kind,
                            file_id,
                            offset,
                        });
                    }
                    Moved::ForeignList(list.into())
                } else if let Some(tag) = inner.weak_element_tag() {
                    // A weak array (`[#[bstack_weak] T; N]`, opt): each element is a
                    // `u64` **control-block** offset at `off + i*8`. Kept distinct from a
                    // data-ref list (`Moved::WeakList`, the array analog of `Moved::Weak`)
                    // so a control offset is never handed back as if it named a `T`.
                    let mut list = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        let e = read_u64(data, add_off(off, mul_off(i, 8)?)?)?;
                        list.push((e != 0).then(|| unsafe { AnyRef::new(tag, e) }));
                    }
                    Moved::WeakList(list.into())
                } else if let Some(tag) = inner.element_ref_tag() {
                    // A flat data-reference array (`owned` / `strong` / `ref`, opt): each
                    // element is a `u64` **data** offset at `off + i*8`.
                    let mut list = Vec::new(); // no capacity hint: `n` is untrusted
                    for i in 0..*n as u64 {
                        let e = read_u64(data, add_off(off, mul_off(i, 8)?)?)?;
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
                } else if matches!(&**inner, Shape::Array { .. }) && inner.has_reference() {
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
                } else if inner.has_reference() {
                    // Any other reference-bearing element we can't flatten one-per-`u64`.
                    return Err(rtti_err!(
                        Unsupported,
                        "RTTI move_out of an array whose element is a vector (or \
                         other reference-bearing container that is neither a flat reference, \
                         an `#[embed]`, nor a nested array) is not yet supported"
                    ));
                } else {
                    // A POD array (nested or not): the whole inline run of bytes.
                    // Untrusted `n`: bound the run against the stack before allocating,
                    // as the scalar `Pod` arm does — a forged length must fail cleanly,
                    // not size a multi-GiB `vec` before the `get_into` would reject it.
                    let total = mul_off(*n as u64, self.shape_stride(inner, cache)?)?;
                    if total > data.len()?.saturating_sub(off) {
                        return Err(rtti_err!(
                            Malformed,
                            "RTTI POD array runs past the end of the data stack",
                        ));
                    }
                    let mut buf = vec![0u8; total as usize];
                    data.get_into(off, &mut buf)?;
                    Moved::Pod(buf.into())
                }
            }
            Shape::Tuple(items) => {
                if items.iter().any(Shape::has_reference) {
                    // A tuple carrying any reference member — a same-file `owned` /
                    // `strong` / `weak` / `vec`, or a cross-file `foreign` — is moved
                    // member-by-member: a POD member by value, a reference member as its
                    // own `Moved` holding the transferred `AnyRef`, at cumulative element
                    // offsets. A whole-tuple POD copy here would bury an owned pointer in
                    // opaque bytes, orphaning its block (never handed back, never freed).
                    let mut parts = Vec::with_capacity(items.len());
                    let mut eo = off;
                    for it in items {
                        parts.push(self.move_field(alloc, data, it, eo, cache, materialized)?);
                        eo = add_off(eo, self.shape_stride(it, cache)?)?;
                    }
                    Moved::Tuple(parts.into())
                } else {
                    // A pure-POD aggregate: its inline bytes (sum of element strides),
                    // bounded against the stack before allocating (element strides are
                    // schema-derived and untrusted).
                    let mut total = 0u64;
                    for it in items {
                        total = add_off(total, self.shape_stride(it, cache)?)?;
                    }
                    if total > data.len()?.saturating_sub(off) {
                        return Err(rtti_err!(
                            Malformed,
                            "RTTI POD tuple runs past the end of the data stack",
                        ));
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
}
