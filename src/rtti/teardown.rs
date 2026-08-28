//! The **free interpreter** — the non-recursive teardown walk that reclaims a
//! structure's `owned` / `embed` / `strong` / `weak` / `ref` / `vec` / array / tuple /
//! option storage, refcount decrements and all. The RTTI analog of `io_core::teardown`.

use std::collections::HashMap;
use std::io;

use bstack::BStackRange;

use crate::BStackRaiiAllocator;
use crate::io_core::refcount;
use crate::primitives::{EightCC, NonNullOffset, OwnershipKind, WidePtr};
use crate::registry::{FileId, ForeignHostAllocator};
use crate::types::compiled::rc::{
    CTRL_BACKPTR_OFFSET, CTRL_STRONG_OFFSET, CTRL_WEAK_OFFSET, RC_REFCOUNT_OFFSET,
};
use crate::util::{io_error, read_u64};

use super::walk::{
    DepthGuard, checked_vec_len, commit_weak_release, disc_mask, foreign_leaf, option_present,
    read_disc,
};
use super::{
    BYTEVEC_HEADER, CONTROL_SIZE, RttiBody, RttiOrdinal, RttiRegistry, RttiType, Shape, add_off,
    mul_off, unknown_tag,
};

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
                io_error!(
                    InvalidData,
                    "[BSTACK0807] RTTI teardown budget exceeded (corrupt data or a cycle?)"
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
                                    io_error!(
                                        InvalidData,
                                        "[BSTACK0808] no RTTI variant for discriminant {}",
                                        raw
                                    )
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
                        let data_off = read_u64(data, offset)?;
                        if data_off != 0 {
                            strong_releases.push((tag, data_off));
                        }
                    }
                    Shape::Weak(_) => {
                        // A weak field's slot holds the *control* offset directly.
                        let ctrl_off = read_u64(data, offset)?;
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
                            io_error!("[BSTACK0807] RTTI teardown budget exceeded (corrupt data or a cycle?)")
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
                            match &*inner {
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
                                        if e != 0 {
                                            strong_releases.push((*tag, e));
                                        }
                                    }
                                }
                                Shape::Weak(_) => {
                                    for i in 0..len {
                                        let c =
                                            read_u64(data, add_off(base, mul_off(i, stride)?)?)?;
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
        unsafe { crate::io_core::commit_home_frees(alloc, to_free) }
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
            let ctrl_off = read_u64(data, add_off(data_off, CTRL_BACKPTR_OFFSET)?)?;
            if refcount::fetch_sub(
                data,
                NonNullOffset::from_field(add_off(ctrl_off, CTRL_STRONG_OFFSET)?)?,
                1,
            )? == 1
            {
                // SAFETY: this caller was the last strong owner (the fetch_sub hit
                // zero), and `data_off` came from the owning slot being released.
                unsafe { self.teardown(alloc, ord, data_off)? };
                if refcount::fetch_sub(
                    data,
                    NonNullOffset::from_field(add_off(ctrl_off, CTRL_WEAK_OFFSET)?)?,
                    1,
                )? == 1
                {
                    // SAFETY: last weak released — the control block is unreferenced.
                    unsafe { alloc.free_many([BStackRange::new(ctrl_off, CONTROL_SIZE)])? };
                }
            }
        } else if refcount::fetch_sub(
            data,
            NonNullOffset::from_field(add_off(data_off, RC_REFCOUNT_OFFSET)?)?,
            1,
        )? == 1
        {
            // SAFETY: as above — sole owner, slot-derived offset.
            unsafe { self.teardown(alloc, ord, data_off)? };
        }
        Ok(())
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
                    let ctrl = read_u64(data, add_off(offset, CTRL_BACKPTR_OFFSET)?)?;
                    if refcount::fetch_sub(
                        data,
                        NonNullOffset::from_field(add_off(ctrl, CTRL_STRONG_OFFSET)?)?,
                        1,
                    )? == 1
                    {
                        // SAFETY: last strong owner; slot-derived offset.
                        unsafe { self.teardown(target, ord, offset)? };
                        if refcount::fetch_sub(
                            data,
                            NonNullOffset::from_field(add_off(ctrl, CTRL_WEAK_OFFSET)?)?,
                            1,
                        )? == 1
                        {
                            // SAFETY: last weak released — control block unreferenced.
                            unsafe { target.free_many([BStackRange::new(ctrl, CONTROL_SIZE)])? };
                        }
                    }
                } else if refcount::fetch_sub(
                    data,
                    NonNullOffset::from_field(add_off(offset, RC_REFCOUNT_OFFSET)?)?,
                    1,
                )? == 1
                {
                    // SAFETY: last strong owner; slot-derived offset.
                    unsafe { self.teardown(target, ord, offset)? };
                }
                Ok(())
            }
            OwnershipKind::Weak => {
                // A weak foreign's offset is the control offset.
                if refcount::fetch_sub(
                    data,
                    NonNullOffset::from_field(add_off(offset, CTRL_WEAK_OFFSET)?)?,
                    1,
                )? == 1
                {
                    // SAFETY: last weak released — control block unreferenced.
                    unsafe { target.free_many([BStackRange::new(offset, CONTROL_SIZE)])? };
                }
                Ok(())
            }
        }
    }
}
