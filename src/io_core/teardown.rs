//! Disk-level recursive destruction, fully decoupled from Rust's `Drop`.
//!
//! [`BStackDrop`] is implemented by every `#[bstack_block]` type (frees the
//! block and recurses into its owned children) and by the small child-handle
//! types in [`crate::handle`]. It takes `self` (a *without-allocator* handle)
//! plus an explicit allocator, so it is generic over all handle-like types.

use core::cell::RefCell;
use std::io;

use bstack::{BStackGenOp, BStackOwnedSlice, BStackRange};

use crate::BStackRaiiAllocator;
use crate::registry::FileId;
use crate::types::drop::BStackDrop;
use crate::io_core::wal::{WalEntry, WalLog, WalStatus, finish_at_locked, persist_at, wal_lock_for};

/// A collected teardown transaction: the installing allocator's **stack identity**
/// (scoping the sink to its file) plus the `(file, range)` frees gathered so far.
type TeardownSink = (usize, Vec<(FileId, BStackRange)>);

thread_local! {
    /// While a WAL-backed teardown is in progress, the collector that
    /// [`dealloc_range`] funnels every subtree slice into *instead of* freeing it
    /// eagerly. The root driver ([`wal_teardown`]) installs it; the generated
    /// recursion and nested handle `bstack_drop`s see it transparently (they all
    /// go through `dealloc_range`), so no allocator/sink parameter has to be
    /// threaded through the whole teardown.
    ///
    /// Each entry is `(file_id, range)`: the [`FileId`] the slice lives in — the
    /// tearing allocator's [`wal_file_id`](BStackRaiiAllocator::wal_file_id), which is
    /// [`SELF`](FileId::SELF) for the home file and the foreign id when a
    /// `Foreign<T>` subtree is being torn down through a `ForeignHostAllocator`. The
    /// WAL commits each entry against its own file so recovery reclaims it there.
    ///
    /// The sink is **keyed by the installing allocator's stack identity** (the first
    /// element), so [`dealloc_range`] only collects frees that belong to that file. A
    /// nested `bstack_drop` against a *different* file's ordinary allocator (whose
    /// `wal_file_id` is also `SELF`, but for *its* file) is **not** collected — tagging
    /// it `SELF` would misdirect it into the installer's file — it falls through to an
    /// eager free in its own file. (A genuine cross-file `Foreign` free carries a
    /// non-`SELF` id and is still collected, then routed by the registry on recovery.)
    static TEARDOWN_SINK: RefCell<Option<TeardownSink>> = const { RefCell::new(None) };
}

/// The address of an allocator's backing `BStack` — a stable per-file identity used to
/// scope the teardown sink to the file that installed it (see [`TEARDOWN_SINK`]).
fn stack_addr<A: BStackRaiiAllocator>(allocator: &A) -> usize {
    allocator.stack() as *const _ as usize
}

thread_local! {
    /// Current nesting depth of the *generated* teardown recursion —
    /// `OwnedRef::bstack_drop` re-entering `__bstack_drop_children` in-file, and
    /// the `foreign_drop_*` helpers re-entering it across files. The RTTI
    /// interpreter bounds the same two recursions (its in-file walk by a node
    /// budget, its cross-file hops by `DepthGuard`); this is the static
    /// counterpart, so an owned cycle returns `Err` instead of exhausting the
    /// native stack (an abort).
    static TEARDOWN_DEPTH: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// Generous bound on [`TEARDOWN_DEPTH`]: legitimate ownership nesting is bounded
/// by the depth of the *type* graph (each level is a distinct `#[bstack_owned]`
/// field or collection hop), which real programs keep in the tens; a chain past
/// this is a cycle or corruption. Kept well under typical native stack limits so
/// the failure is an `io::Error`, not a stack-overflow abort.
const MAX_TEARDOWN_DEPTH: u32 = 500;

/// Scope guard for [`TEARDOWN_DEPTH`] — increments on entry (refusing past
/// [`MAX_TEARDOWN_DEPTH`]), decrements on drop (so error unwinds restore it).
pub(crate) struct TeardownDepthGuard;

impl TeardownDepthGuard {
    pub(crate) fn enter() -> io::Result<Self> {
        let depth = TEARDOWN_DEPTH.with(|c| {
            let n = c.get() + 1;
            c.set(n);
            n
        });
        if depth > MAX_TEARDOWN_DEPTH {
            TEARDOWN_DEPTH.with(|c| c.set(c.get() - 1)); // undo: no guard returned
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "teardown recursion too deep (an owned-reference cycle?)",
            ));
        }
        Ok(TeardownDepthGuard)
    }
}

impl Drop for TeardownDepthGuard {
    fn drop(&mut self) {
        TEARDOWN_DEPTH.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Tear down `handle` (a whole owned subtree) as one crash-atomic batch of frees,
/// so a crash mid-teardown is completed — not leaked — by `finish` on the next
/// open. This is what [`BStackOwned::bstack_drop`](crate::BStackOwned) runs, so
/// **every owned teardown is automatically WAL-backed** when the allocator names
/// an anchor ([`BStackRaiiAllocator::wal_anchor`] returns `Some`); an allocator that
/// returns `None` falls straight through to a plain [`BStackDrop::bstack_drop`]
/// and behaves exactly as before (mid-teardown crash ⇒ orphan leak).
///
/// While the sink is installed, every [`dealloc_range`] in the (ordinary,
/// generic) teardown recursion *collects* its slice rather than freeing it;
/// afterwards the whole set commits as one `Dealloc` transaction and is executed
/// via [`finish`](crate::io_core::wal::finish)'s completion path (the same path crash
/// recovery takes). Nested owned frees (e.g. a collection freeing its values
/// through `BStackOwned::bstack_drop`) see the sink already set and just collect,
/// so exactly one transaction wraps the outermost teardown.
pub fn wal_teardown<A: BStackRaiiAllocator, T: BStackDrop>(
    handle: T,
    allocator: &A,
) -> io::Result<()> {
    // Nested teardown: an outer driver already owns the sink; frees already
    // collect, so just recurse (exactly one transaction wraps the whole subtree).
    if TEARDOWN_SINK.with(|s| s.borrow().is_some()) {
        return handle.bstack_drop(allocator);
    }
    // No anchor → the allocator opts out of reclamation: plain teardown.
    if allocator.wal_anchor().is_none() {
        return handle.bstack_drop(allocator);
    }
    TEARDOWN_SINK.with(|s| *s.borrow_mut() = Some((stack_addr(allocator), Vec::new())));
    // Clear the sink on *every* exit from here on — including an unwind out of
    // `bstack_drop`. A bare `.take()` after the call is skipped on panic, leaving the
    // sink `Some`; the next top-level teardown on this thread would then hit the
    // `is_some()` guard above, misdetect a *nested* call, and silently funnel every free
    // into the stale (never-committed) sink — leaking the whole subtree while returning
    // `Ok` (issue F3). The guard restores `None` even if a caught panic unwinds through.
    struct SinkGuard;
    impl Drop for SinkGuard {
        fn drop(&mut self) {
            TEARDOWN_SINK.with(|s| *s.borrow_mut() = None);
        }
    }
    let _sink_guard = SinkGuard;
    let result = handle.bstack_drop(allocator);
    let slices = TEARDOWN_SINK
        .with(|s| s.borrow_mut().take())
        .map(|(_, v)| v)
        .unwrap_or_default();
    // A mid-walk error means the teardown did not finish, so the frees it *did*
    // collect must NOT be committed: freeing a partial set while the still-linked
    // parent points at those ranges is an observable torn structure a retry then
    // double-frees. Discard the collected frees — nothing is freed, the structure
    // is intact, and the caller can retry (the crate's "corruption degrades to a
    // reclaimable leak, never a torn structure" baseline). Only the collect-and-
    // commit *frees* are undone this way; refcount decrements a `strong` child's
    // teardown already applied are not collected here and remain a retry hazard.
    if result.is_err() {
        return result;
    }
    // Bulk-capable allocator, same-file subtree: `dealloc_bulk` is itself atomic and
    // self-recovering, so free the whole subtree as one atomic batch and **skip the
    // WAL** — wrapping an already-atomic bulk free in the WAL is redundant and
    // unsound (the allocator's recovery direction is opaque, so a WAL retry could
    // double-free). A crash mid-bulk is reclaimed by the allocator's own recovery.
    // A cross-file (mixed `FileId`) teardown still routes through the WAL so its
    // foreign frees are replayed via the registry on recovery.
    if allocator.atomic_bulk() && slices.iter().all(|(fid, _)| *fid == FileId::SELF) {
        // SAFETY: the sink's ranges were each collected from an owned handle's
        // own teardown; the deferral changes when they are freed, not what.
        unsafe { allocator.free_many(slices.into_iter().map(|(_, r)| r))? };
    } else {
        wal_free_all(allocator, slices)?;
    }
    result
}

/// Commit `slices` as one committed `Dealloc` transaction and execute the frees.
/// A crash mid-free leaves a `Complete` WAL that `finish` rolls forward on reopen.
///
/// The whole staging→commit→finish critical section runs under the file's WAL
/// lock, so concurrent teardowns on the same file serialize here (they collect
/// their subtrees independently first — that part stays concurrent) rather than
/// racing the single shared anchor slot.
fn wal_free_all<A: BStackRaiiAllocator>(
    allocator: &A,
    slices: Vec<(FileId, BStackRange)>,
) -> io::Result<()> {
    if slices.is_empty() {
        return Ok(());
    }
    let lock = wal_lock_for(allocator);
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());

    let mut log = WalLog::with_capacity(slices.len());
    for (fid, s) in &slices {
        // `fid == SELF` ⇒ a local free (this file); a foreign id ⇒ reclaimed in that
        // file through the registry on `finish` (see `wal::free_recorded`).
        log.append(WalEntry::dealloc_in(WalStatus::Pending, *fid, *s));
    }
    // Stage the transaction `Pending`, then commit it by flipping `txn_status` to
    // `Complete` in one atomic `inplace_gen` — the single commit point (a crash
    // before it abandons; after it, `finish` rolls the frees forward). With the
    // sink now cleared, `finish_at_locked` executes the frees and marks the
    // persistent WAL block idle for reuse.
    let wal_range = persist_at(allocator, &log, WalStatus::Pending)?;
    let flip = [WalStatus::Complete as u8];
    let mut done = false;
    allocator.stack().inplace_gen(|_feedback| {
        if done {
            None
        } else {
            done = true;
            // SAFETY: `flip` outlives the call; the `txn_status` byte follows the
            // u64 magic at offset 8 in `WalHeader`.
            let data: &[u8] = unsafe { core::mem::transmute::<&[u8], _>(&flip[..]) };
            Some(BStackGenOp::Write {
                offset: wal_range.start() + 8,
                data,
            })
        }
    })?;
    finish_at_locked(allocator)?;
    Ok(())
}

/// Commit a batch of **home-file** owned-range frees for crash recovery, routed
/// exactly as [`wal_teardown`]: a bulk-capable allocator frees them atomically
/// through its own self-recovering machinery; otherwise they go through the WAL,
/// so a crash mid-free is rolled forward on the next open rather than leaking
/// permanently.
///
/// This is the sink the RTTI interpreter uses so its collected frees no longer
/// bypass the WAL their static counterparts (`wal_teardown` / `bstack_move!`) go
/// through. All ranges must live in `allocator`'s own file
/// (tagged [`FileId::SELF`]); cross-file releases are handled separately by the
/// interpreter (`teardown_foreign`).
///
/// # Safety
/// Each range must be a live allocation owned by `allocator` that no other live
/// handle will also free (as for [`dealloc_range`] / [`free_many`]).
pub(crate) unsafe fn commit_home_frees<A: BStackRaiiAllocator>(
    allocator: &A,
    ranges: Vec<BStackRange>,
) -> io::Result<()> {
    if ranges.is_empty() {
        return Ok(());
    }
    if allocator.atomic_bulk() {
        // SAFETY: forwarded from the caller's contract.
        unsafe { allocator.free_many(ranges) }
    } else {
        let slices = ranges.into_iter().map(|r| (FileId::SELF, r)).collect();
        wal_free_all(allocator, slices)
    }
}

/// Free a block by range: recurse into its owned children, then dealloc the shell.
///
/// The single implementation of "tear down a `T` block", shared by every affine
/// owner ([`BStackOwned`](crate::BStackOwned), the `handle::*Ref` tokens, the
/// block-element vectors). It replaces the per-type `impl BStackDrop for <handle>`
/// bodies that used to live on each `Copy` block handle — the handle is now a pure
/// view, so this is reachable only through a non-`Copy` owner.
///
/// # Safety
/// `range` must be a live `T` block owned by `allocator` that no other live owner
/// will also free.
pub(crate) unsafe fn drop_block<T: crate::block::BStackBlock, A: BStackRaiiAllocator>(
    range: BStackRange,
    allocator: &A,
) -> io::Result<()> {
    T::__bstack_drop_children(range, allocator)?;
    unsafe { dealloc_range(allocator, range) }
}

/// Free a raw block range by reconstructing an owned slice and delegating to the
/// allocator. The central sink the generated `bstack_drop` code funnels through,
/// since ranges carry no allocator of their own.
///
/// # Safety
/// `range` must be a live allocation owned by `allocator` that no other live
/// handle will also free.
pub unsafe fn dealloc_range<A: BStackRaiiAllocator>(
    allocator: &A,
    range: BStackRange,
) -> io::Result<()> {
    // Inside a WAL-backed teardown, defer the free: collect the slice so the whole
    // subtree commits (and frees) as one crash-atomic transaction — but ONLY if it
    // belongs to the file that installed the sink. Otherwise a nested `bstack_drop`
    // against a different file's ordinary allocator would be tagged `SELF` and freed in
    // the installer's file (a cross-file misdirected free); such a free must go eagerly
    // to its own file instead.
    let deferred = TEARDOWN_SINK.with(|s| match s.borrow_mut().as_mut() {
        Some((installer, sink)) => {
            // A genuine cross-file `Foreign` free (through a `ForeignHostAllocator`)
            // carries a non-`SELF` id and is routed to that file by the registry on
            // recovery — collect it. A same-file free (matching stack identity) is the
            // ordinary case. Anything else is a *different* file's free: don't collect.
            let foreign = allocator.wal_file_id() != FileId::SELF;
            if foreign || stack_addr(allocator) == *installer {
                sink.push((allocator.wal_file_id(), range));
                true
            } else {
                false
            }
        }
        None => false,
    });
    if deferred {
        return Ok(());
    }
    let owned: BStackOwnedSlice<'_, A> =
        unsafe { BStackOwnedSlice::from_raw_range(allocator, range) };
    allocator.dealloc(owned).map_err(|e| e.source)
}
