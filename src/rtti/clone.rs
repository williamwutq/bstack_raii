//! The **deep-clone interpreter** — the non-recursive clone walk: `owned` / `embed`
//! sub-structure byte-copied into fresh blocks and repointed, shared (`strong` / `weak`)
//! targets refcount-bumped, WAL-integrated for crash-safe reclamation. The RTTI analog
//! of `io_core::clone`.

use std::collections::HashMap;

use bstack::{BStack, BStackRange};

use crate::BStackRaiiAllocator;
use crate::io_core::{WalTxn, alloc_logged, refcount};
use crate::primitives::{EightCC, NonNullOffset, Offset, OwnershipKind, WidePtr};
use crate::registry::{FileId, ForeignHostAllocator};
use crate::types::compiled::rc::{
    CTRL_BACKPTR_OFFSET, CTRL_STRONG_OFFSET, CTRL_WEAK_OFFSET, RC_REFCOUNT_OFFSET,
};
use crate::util::read_u64;

use super::walk::{DepthGuard, checked_vec_len};
use super::{
    BYTEVEC_HEADER, FOREIGN_REPR_LEN, RttiBody, RttiOrdinal, RttiRegistry, RttiType, Shape,
    add_off, mul_off, unknown_tag,
};
use super::{RttiResult, rtti_err};

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
    bumps: Vec<NonNullOffset>,
    /// Every freshly allocated range, so a failed clone frees its orphans.
    allocated: Vec<BStackRange>,
    /// The in-flight intention-first WAL transaction: when the allocator
    /// names a WAL anchor, each `alloc_copy` block is logged `Pending` before it is
    /// used, so a **crash** mid-clone is reclaimed by [`wal::finish`](crate::io_core::wal::finish)
    /// on the next open (the in-process error path already frees `allocated`). `None`
    /// when the allocator opts out of reclamation or nothing has been allocated yet.
    wal: Option<WalTxn>,
}

impl RttiRegistry {
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
    ) -> RttiResult<()> {
        if matches!(kind, OwnershipKind::Ref) {
            return Ok(()); // aliased — the copied slot is correct
        }
        let __wp = WidePtr::read_from_stack(home_data, src_off)?;
        let Some(src_target) = NonNullOffset::new(__wp.offset()) else {
            return Ok(()); // null — copied as 0
        };
        let new_off = NonNullOffset::from_field(new_off)?;
        if __wp.is_self() {
            self.clone_foreign_in(home, home_data, tag, kind, src_target, new_off)
        } else {
            let fid = FileId::from_u64(__wp.file_id()).ok_or_else(|| {
                rtti_err!(
                    ForeignFile,
                    "RTTI clone: the foreign target names an invalid file id"
                )
            })?;
            let host = crate::registry::host_arc(fid).ok_or_else(|| {
                rtti_err!(
                    ForeignFile,
                    "RTTI clone: the foreign target's file is detached / not attached"
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
        src_target: NonNullOffset,
        new_off: NonNullOffset,
    ) -> RttiResult<()> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        let tstack = target.stack();
        match kind {
            OwnershipKind::Ref => Ok(()),
            OwnershipKind::Owned => {
                // SAFETY: `src_target` came from the source's validated foreign slot.
                let new_target = unsafe { self.clone_value(target, ord, src_target.as_u64())? };
                // Repoint only the address word of the copied WidePtr.
                home_data
                    .set(
                        new_off.checked_add(FOREIGN_REPR_LEN - 8)?.as_u64(),
                        new_target.to_le_bytes(),
                    )
                    .map_err(Into::into)
            }
            OwnershipKind::Strong => {
                // `src_target` is proven non-null, so `strong`'s inline counter offset
                // stays branded through `checked_add`; only the `(rc, weak)` control
                // offset — read from disk — needs re-validating against null.
                let slot = if self.load_type(ord)?.weak {
                    let ctrl = read_u64(
                        tstack,
                        src_target.checked_add(CTRL_BACKPTR_OFFSET)?.as_u64(),
                    )?;
                    NonNullOffset::from_field(add_off(ctrl, CTRL_STRONG_OFFSET)?)?
                } else {
                    src_target.checked_add(RC_REFCOUNT_OFFSET)?
                };
                refcount::fetch_add(tstack, slot, 1)?;
                Ok(())
            }
            OwnershipKind::Weak => {
                refcount::fetch_add(tstack, src_target.checked_add(CTRL_WEAK_OFFSET)?, 1)?;
                Ok(())
            }
        }
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
    ) -> RttiResult<u64> {
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
                    w.set_idle(alloc)?;
                }
                Ok(new_root)
            }
            Err(e) => {
                // Reclaim the orphaned partial clone (leak-free error path). With a WAL
                // transaction in flight, abandon it — `finish_at_locked` frees exactly
                // the still-`Pending` `Alloc`s (== `st.allocated`) and marks the block
                // idle, the same path a crash takes. Otherwise free the ranges directly.
                if let Some(w) = &st.wal {
                    let _ = w.finish(alloc);
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
    ) -> RttiResult<u64> {
        let mut work: Vec<CloneOp> = vec![CloneOp::Block {
            src_off: root_src,
            ord: ordinal,
        }];
        let mut budget: u64 = 4_000_000;

        while let Some(op) = work.pop() {
            budget = budget.checked_sub(1).ok_or_else(|| {
                rtti_err!(
                    Budget,
                    "RTTI clone budget exceeded (corrupt data or a cycle?)"
                )
            })?;
            match op {
                CloneOp::Block { src_off, ord } => {
                    self.ensure_type(ord, st)?;
                    let size = st.cache[&ord].ondisk_size;
                    let new_off =
                        self.alloc_copy(alloc, NonNullOffset::from_field(src_off)?, size, st)?;
                    st.map.insert(src_off, new_off);
                    // Walk the fields at matching source / destination offsets.
                    let ty = &st.cache[&ord];
                    self.push_clone_fields(&mut work, data, ty, src_off, new_off)?;
                }

                CloneOp::Inline {
                    src_base,
                    new_base,
                    ord,
                } => {
                    self.ensure_type(ord, st)?;
                    let ty = &st.cache[&ord];
                    self.push_clone_fields(&mut work, data, ty, src_base, new_base)?;
                }

                CloneOp::Field {
                    shape,
                    src_off,
                    new_off,
                } => match shape {
                    // Copied verbatim: inline bytes, a schema value, or a `ref` alias.
                    Shape::Pod { .. } | Shape::Class { .. } | Shape::Ref(_) => {}
                    Shape::Owned(tag) => {
                        let child = read_u64(data, src_off)?;
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
                        if let Some(child) =
                            Offset::from_raw(read_u64(data, src_off)?).to_non_null()
                        {
                            let off = self.strong_bump_off(data, tag, child, st)?;
                            st.bumps.push(off);
                        }
                    }
                    Shape::Weak(_) => {
                        if let Some(ctrl) = Offset::from_raw(read_u64(data, src_off)?).to_non_null()
                        {
                            st.bumps.push(ctrl.checked_add(CTRL_WEAK_OFFSET)?);
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
                        if inner.option_present(data, src_off)? {
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
                            rtti_err!(
                                Budget,
                                "RTTI clone budget exceeded (corrupt data or a cycle?)"
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
                        let src_data = read_u64(data, src_off)?; // VecDesc.data_off
                        if src_data != 0 {
                            let data_size = read_u64(data, add_off(src_off, 8)?)?;
                            let new_data = self.alloc_copy(
                                alloc,
                                NonNullOffset::from_field(src_data)?,
                                data_size,
                                st,
                            )?;
                            // Repoint the (freshly-copied) descriptor's data pointer;
                            // its size word was copied verbatim.
                            data.set(new_off, new_data.to_le_bytes())?;
                            // `@0` is the byte length; count is `byte_len / stride`
                            // (8 per `u64` offset, 16 per `WidePtr`).
                            let stride = self.shape_stride(&inner, &mut st.cache)?;
                            let byte_len = read_u64(data, src_data)?;
                            let len = checked_vec_len(byte_len, data_size, stride)?;
                            let sbase = add_off(src_data, BYTEVEC_HEADER)?;
                            let nbase = add_off(new_data, BYTEVEC_HEADER)?;
                            match &*inner {
                                Shape::Owned(tag) => {
                                    let ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                                    for i in 0..len {
                                        let delta = mul_off(i, stride)?;
                                        let e = read_u64(data, add_off(sbase, delta)?)?;
                                        if e != 0 {
                                            st.patches.push((add_off(nbase, delta)?, e));
                                            work.push(CloneOp::Block { src_off: e, ord });
                                        }
                                    }
                                }
                                Shape::Strong(tag) => {
                                    for i in 0..len {
                                        let e =
                                            read_u64(data, add_off(sbase, mul_off(i, stride)?)?)?;
                                        if let Some(e) = Offset::from_raw(e).to_non_null() {
                                            let off = self.strong_bump_off(data, *tag, e, st)?;
                                            st.bumps.push(off);
                                        }
                                    }
                                }
                                Shape::Weak(_) => {
                                    for i in 0..len {
                                        let c =
                                            read_u64(data, add_off(sbase, mul_off(i, stride)?)?)?;
                                        if let Some(c) = Offset::from_raw(c).to_non_null() {
                                            st.bumps.push(c.checked_add(CTRL_WEAK_OFFSET)?);
                                        }
                                    }
                                }
                                // A vector of `Foreign` pointers: the data block (and
                                // its reprs) was byte-copied above; deep-copy each
                                // `owned` target across the boundary, bump `strong` /
                                // `weak` — the per-element mirror of the scalar path.
                                other if other.foreign_leaf().is_some() => {
                                    let (tag, kind) = other.foreign_leaf().unwrap();
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
                .ok_or_else(|| rtti_err!(Clone, "RTTI clone: an owned child was not cloned"))?;
            data.set(new_slot, new_child.to_le_bytes())?;
        }
        // Then bump every shared target's refcount (over-count-safe, never under).
        for &off in &st.bumps {
            refcount::fetch_add(data, off, 1)?;
        }

        st.map
            .get(&root_src)
            .copied()
            .ok_or_else(|| rtti_err!(Clone, "RTTI clone: the root was not cloned"))
    }

    /// Allocate a `size`-byte block and byte-copy `[src_off, src_off+size)` into it,
    /// recording it in `st.allocated`. Returns the new block's start offset. Source and
    /// destination are both in `alloc`'s own stack (`alloc.stack()`): the returned range
    /// is an offset in that file, so the copy must land there — the coupling is derived
    /// here rather than threaded, so a mismatched `(alloc, data)` pair can't be formed.
    ///
    /// `src_off` is a **block start**, taken [`NonNullOffset`] so a `0` (a misread or
    /// forged source) is rejected up front rather than silently copying the file header.
    fn alloc_copy<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        src_off: NonNullOffset,
        size: u64,
        st: &mut CloneState,
    ) -> RttiResult<u64> {
        let data = alloc.stack();
        // Allocate + WAL-log intention-first (shared with the compiled clone path).
        let range = alloc_logged(alloc, &mut st.wal, &mut st.allocated, size)?;
        data.copy(src_off.as_u64(), range.start(), size)?;
        Ok(range.start())
    }

    /// The counter offset to bump when cloning a `strong` reference to `data_child`
    /// of type `tag`: an `rc` block's inline refcount, or an `(rc, weak)` block's
    /// `ctrl.strong` (reached via the data block's `ctrl` back-pointer). Strong only —
    /// a strong clone never adds a phantom weak.
    fn strong_bump_off(
        &self,
        data: &BStack,
        tag: EightCC,
        data_child: NonNullOffset,
        st: &mut CloneState,
    ) -> RttiResult<NonNullOffset> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        self.ensure_type(ord, st)?;
        if st.cache[&ord].weak {
            // `data_child` is proven non-null, so the back-pointer address stays branded
            // through `checked_add`; the `ctrl` read *from disk* is the one value that can
            // be null (a corrupt back-pointer), so it re-validates via `from_field`.
            let ctrl = read_u64(data, data_child.checked_add(CTRL_BACKPTR_OFFSET)?.as_u64())?;
            Ok(NonNullOffset::from_field(add_off(
                ctrl,
                CTRL_STRONG_OFFSET,
            )?)?)
        } else {
            Ok(data_child.checked_add(RC_REFCOUNT_OFFSET)?)
        }
    }

    /// Push a [`CloneOp::Field`] for every field of `ty` — a struct's fields, or the
    /// active enum variant's fields (selected by the **source** discriminant) — pairing
    /// each field's source offset (`src_base + f.offset`) with its destination
    /// (`new_base + f.offset`). Shared by the `Block` and `Inline` arms of the clone
    /// walk, whose only difference is that `Block` first allocates and byte-copies the
    /// block before walking it.
    fn push_clone_fields(
        &self,
        work: &mut Vec<CloneOp>,
        data: &BStack,
        ty: &RttiType,
        src_base: u64,
        new_base: u64,
    ) -> RttiResult<()> {
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
                // The active variant is chosen by the source discriminant; its payload
                // sits at the same `payload_off` in the fresh block.
                let (variant, src_payload) = e.resolve_variant(data, src_base)?;
                let new_payload = add_off(new_base, e.payload_off as u64)?;
                for f in &variant.fields {
                    work.push(CloneOp::Field {
                        shape: f.shape.clone(),
                        src_off: add_off(src_payload, f.offset as u64)?,
                        new_off: add_off(new_payload, f.offset as u64)?,
                    });
                }
            }
        }
        Ok(())
    }

    /// Load + cache a type descriptor if not already present.
    fn ensure_type(&self, ord: RttiOrdinal, st: &mut CloneState) -> RttiResult<()> {
        if let std::collections::hash_map::Entry::Vacant(e) = st.cache.entry(ord) {
            e.insert(self.load_type(ord)?);
        }
        Ok(())
    }
}
