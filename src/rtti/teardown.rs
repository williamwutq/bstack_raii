//! The **free interpreter** — the non-recursive teardown walk that reclaims a
//! structure's `owned` / `embed` / `strong` / `weak` / `ref` / `vec` / array / tuple /
//! option storage, refcount decrements and all. The RTTI analog of `io_core::teardown`.

use std::collections::HashMap;

use bstack::BStackRange;

use crate::BStackRaiiAllocator;
use crate::io_core::refcount;
use crate::primitives::{EightCC, NonNullOffset, Offset, OwnershipKind, WidePtr};
use crate::registry::{FileId, ForeignHostAllocator};
use crate::util::read_u64;

use super::walk::{DepthGuard, checked_vec_len, commit_weak_release, strong_counter_slot};
use super::{
    BYTEVEC_HEADER, RttiBody, RttiOrdinal, RttiRegistry, RttiType, Shape, add_off, mul_off,
    unknown_tag,
};
use super::{RttiResult, rtti_err};

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

impl RttiRegistry {
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
    ) -> RttiResult<()> {
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
        let mut strong_releases: Vec<(EightCC, NonNullOffset)> = Vec::new(); // (tag, data offset)
        let mut weak_releases: Vec<NonNullOffset> = Vec::new(); // control offsets
        let mut foreign_releases: Vec<(EightCC, OwnershipKind, u64, u64)> = Vec::new();
        let mut budget: u64 = 4_000_000;

        while let Some(op) = work.pop() {
            budget = budget.checked_sub(1).ok_or_else(|| {
                rtti_err!(
                    Budget,
                    "RTTI teardown budget exceeded (corrupt data or a cycle?)"
                )
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
                            let (variant, payload_base) = e.resolve_variant(data, block_off)?;
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
                        let child = read_u64(data, offset)?;
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
                        // The slot holds the target's data offset; a null is an absent
                        // reference (nothing to release). Refine once here.
                        if let Some(data_off) =
                            Offset::from_raw(read_u64(data, offset)?).to_non_null()
                        {
                            strong_releases.push((tag, data_off));
                        }
                    }
                    Shape::Weak(_) => {
                        // A weak field's slot holds the *control* offset directly; a null
                        // is an absent handle. Refine once here.
                        if let Some(ctrl_off) =
                            Offset::from_raw(read_u64(data, offset)?).to_non_null()
                        {
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
                        if inner.option_present(data, offset)? {
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
                            rtti_err!(
                                Budget,
                                "RTTI teardown budget exceeded (corrupt data or a cycle?)"
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
                        let data_off = read_u64(data, offset)?; // VecDesc.data_off @0
                        if data_off != 0 {
                            let data_size = read_u64(data, add_off(offset, 8)?)?; // .data_size @8
                            // A vector of owning/shared elements releases each element
                            // from the data block's element area too. The `@0` word is
                            // the byte length, so the count is `byte_len / stride`
                            // (stride = 8 for a `u64` offset, 16 for a `WidePtr`).
                            let base = add_off(data_off, BYTEVEC_HEADER)?;
                            let stride = self.shape_stride(&inner, &mut cache)?;
                            let byte_len = read_u64(data, data_off)?;
                            let len = checked_vec_len(byte_len, data_size, stride)?;
                            // Peel any `Option` wrapper: an `Option<owned/strong/weak>`
                            // element shares the bare leaf's nullable slot and must be
                            // released, not fall through as inert POD (which would leak).
                            let elem = inner.peel_option();
                            // Charge for all elements up front — `len` comes off the
                            // (untrusted) descriptor, and the element ops are pushed
                            // eagerly, so a huge count must fail cleanly here rather than
                            // grow `work` past the budget before the per-pop check trips
                            // (the same guard the `Array` arm and the read walk apply).
                            budget = budget.checked_sub(len).ok_or_else(|| {
                                rtti_err!(
                                    Budget,
                                    "RTTI teardown budget exceeded (corrupt data or a cycle?)"
                                )
                            })?;
                            match elem {
                                Shape::Owned(tag) => {
                                    let ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                                    for i in 0..len {
                                        let e =
                                            read_u64(data, add_off(base, mul_off(i, stride)?)?)?;
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
                                            read_u64(data, add_off(base, mul_off(i, stride)?)?)?;
                                        if let Some(e) = Offset::from_raw(e).to_non_null() {
                                            strong_releases.push((*tag, e));
                                        }
                                    }
                                }
                                Shape::Weak(_) => {
                                    for i in 0..len {
                                        let c =
                                            read_u64(data, add_off(base, mul_off(i, stride)?)?)?;
                                        if let Some(c) = Offset::from_raw(c).to_non_null() {
                                            weak_releases.push(c);
                                        }
                                    }
                                }
                                // A vector of `Foreign` pointers: each element is a
                                // 16-byte `WidePtr`; its target is torn down in its
                                // own file in the commit phase (a null offset is a no-op).
                                other if other.foreign_leaf().is_some() => {
                                    let (tag, kind) = other.foreign_leaf().unwrap();
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
        unsafe { crate::io_core::commit_home_frees(alloc, to_free) }.map_err(Into::into)
    }

    /// Release one `strong` reference to an `ord`-typed block at `data_off` in the file
    /// `alloc` owns: decrement its strong count and, only if this was the last owner,
    /// tear the data subtree down (its own transaction) and release the phantom weak,
    /// freeing the control block if no real weak handles remain. The type's `weak` flag
    /// selects the inline-refcount vs separate-control-block path. Shared by the
    /// deferred commit ([`commit_strong_release`](Self::commit_strong_release)) and the
    /// resolved-foreign path ([`teardown_foreign_in`](Self::teardown_foreign_in)) — the
    /// same last-owner ladder, differing only in which file's allocator is passed.
    ///
    /// `data_off` must be the data offset of a live `ord`-typed block held by the owning
    /// slot being released — the precondition of [`teardown`](Self::teardown), whose
    /// ownership transfers with the slot; both callers uphold it by construction.
    fn release_strong<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        ord: RttiOrdinal,
        data_off: NonNullOffset,
    ) -> RttiResult<()> {
        let data = alloc.stack();
        // Inline `rc` refcount, or the `(rc, weak)` control block's `strong` (with its
        // `ctrl` offset returned for the phantom-weak release below).
        let (strong_slot, ctrl) = strong_counter_slot(data, self.load_type(ord)?.weak, data_off)?;
        if refcount::fetch_sub(data, strong_slot, 1)? == 1 {
            // SAFETY: last strong owner (the fetch_sub hit zero); `data_off` is the
            // caller's slot-derived block offset.
            unsafe { self.teardown(alloc, ord, data_off.as_u64())? };
            // For `(rc, weak)`, release the phantom weak too — frees the control block
            // if it was the last handle.
            if let Some(ctrl) = ctrl {
                commit_weak_release(alloc, ctrl)?;
            }
        }
        Ok(())
    }

    /// Release one deferred `strong` reference (commit phase of [`teardown`](Self::teardown))
    /// in its home file: resolve `tag` to its ordinal and run the shared last-owner
    /// ladder ([`release_strong`](Self::release_strong)).
    fn commit_strong_release<A: BStackRaiiAllocator>(
        &self,
        alloc: &A,
        tag: EightCC,
        data_off: NonNullOffset,
    ) -> RttiResult<()> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        self.release_strong(alloc, ord, data_off)
    }

    /// Tear down a `Foreign` reference's target **in the target's own file**. `SELF`
    /// (`file_id == 0`) resolves against `home`; a registered file is reached through
    /// its [`BStackRaiiHost`](crate::registry::BStackRaiiHost) — a detached / unknown file
    /// leaks (the design permits it) rather than erroring. `offset == 0` (null) is a
    /// no-op.
    fn teardown_foreign<A: BStackRaiiAllocator>(
        &self,
        home: &A,
        tag: EightCC,
        kind: OwnershipKind,
        file_id: u64,
        offset: u64,
    ) -> RttiResult<()> {
        let Some(offset) = Offset::from_raw(offset).to_non_null() else {
            return Ok(()); // null target — nothing to free
        };
        if matches!(kind, OwnershipKind::Ref) {
            return Ok(()); // a non-owning alias
        }
        if file_id == 0 {
            self.teardown_foreign_in(home, tag, kind, offset)
        } else {
            let Some(fid) = FileId::from_u64(file_id) else {
                return Ok(());
            };
            match crate::registry::host_arc(fid) {
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
        offset: NonNullOffset,
    ) -> RttiResult<()> {
        let ord = self.ordinal_of(tag).ok_or_else(unknown_tag)?;
        match kind {
            OwnershipKind::Ref => Ok(()),
            // SAFETY: `offset` came from the owning foreign slot being torn down;
            // ownership of the target transfers with the slot.
            OwnershipKind::Owned => unsafe { self.teardown(target, ord, offset.as_u64()) },
            // SAFETY: `offset` is the target's data offset, from the owning foreign
            // slot being torn down; ownership transfers with the slot.
            OwnershipKind::Strong => self.release_strong(target, ord, offset),
            // A weak foreign's `offset` is the control offset (proven non-null) — the
            // same last-weak release as the deferred in-file path.
            OwnershipKind::Weak => commit_weak_release(target, offset),
        }
    }
}
