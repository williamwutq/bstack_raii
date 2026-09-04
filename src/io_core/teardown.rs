//! Disk-level recursive destruction, fully decoupled from Rust's `Drop`.
//!
//! [`BStackDrop`] is implemented by every `#[bstack_block]` type (frees the
//! block and recurses into its owned children) and by the small child-handle
//! drop cores in [`crate::types::compiled::owned`] / [`crate::types::compiled::rc`]
//! (`OwnedRef`, `StrongRef`, `StrongWeakRef`, `WeakRef`). It takes `self` (a
//! *without-allocator* handle) plus an explicit allocator, so it is generic over
//! all handle-like types.

use core::cell::RefCell;
use std::io;

use bstack::{BStack, BStackOwnedSlice, BStackRange};

use crate::BStackRaiiAllocator;
use crate::registry::FileId;
use crate::types::traits::BStackDrop;
use crate::util::io_error;

/// A collected teardown transaction: a raw pointer to the installing allocator's
/// [`BStack`] (its identity, scoping the sink to that file — compared via `BStack`'s
/// own pointer-identity [`PartialEq`]) plus the `(file, range)` frees gathered so
/// far. A raw pointer, not `&BStack`, because it must outlive the borrow across the
/// nested `dealloc_range` calls; sound because the installing allocator is borrowed
/// for the whole enclosing [`wal_teardown`]. Thread-local, so the `!Send` pointer is
/// fine (unlike the process-global [`registry`](crate::registry), whose stack keys
/// must stay `usize`).
type TeardownSink = (*const BStack, Vec<(FileId, BStackRange)>);

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

/// A zero-sized handle over [`TEARDOWN_SINK`] that centralises its whole lifecycle —
/// install, collect, take, and clear-on-exit — in one place. Previously each of
/// these was a raw `TEARDOWN_SINK.with(|s| …borrow_mut()…)` scattered across
/// [`wal_teardown`] and [`dealloc_range`], which is where the F3 clear-on-panic bug
/// lived (see [`SinkGuard`]).
struct Sink;

impl Sink {
    /// Whether a teardown sink is currently installed on this thread (a nested
    /// teardown, whose frees already collect into the outer driver's sink).
    fn is_active() -> bool {
        TEARDOWN_SINK.with(|s| s.borrow().is_some())
    }

    /// Install a fresh sink keyed by `stack` (the installing allocator's [`BStack`]
    /// identity), returning the [`SinkGuard`] that clears it on scope exit.
    fn install(stack: *const BStack) -> SinkGuard {
        TEARDOWN_SINK.with(|s| *s.borrow_mut() = Some((stack, Vec::new())));
        SinkGuard
    }

    /// Collect `range` into the installed sink **iff** it belongs to the file that
    /// installed it — deferring the free to the batched commit — and report whether
    /// it did. Returns `false` (i.e. "free it eagerly") when no sink is installed, or
    /// when the range belongs to a *different* file's ordinary allocator (tagging it
    /// `SELF` would misdirect it into the installer's file). A genuine cross-file
    /// `Foreign` free carries a non-`SELF` id and *is* collected, then routed by the
    /// registry on recovery.
    fn collect<A: BStackRaiiAllocator>(allocator: &A, range: BStackRange) -> bool {
        TEARDOWN_SINK.with(|s| match s.borrow_mut().as_mut() {
            Some((installer, sink)) => {
                let foreign = allocator.wal_file_id() != FileId::SELF;
                // SAFETY: `*installer` points at the `BStack` of the allocator that
                // installed the sink, which is borrowed for the whole enclosing
                // `wal_teardown` — so it is live here. The `==` is `BStack`'s own
                // pointer-identity `PartialEq`.
                if foreign || allocator.stack() == unsafe { &**installer } {
                    sink.push((allocator.wal_file_id(), range));
                    true
                } else {
                    false
                }
            }
            None => false,
        })
    }

    /// The sink-aware leaf free: the one dealloc path that consults the teardown
    /// sink. If a sink is installed and `range` belongs to its file, [`collect`]
    /// defers the slice into the batched commit; otherwise the range is freed
    /// eagerly in its own file by reconstructing an owned slice for the allocator.
    /// This is what distinguishes it from a bulk dealloc, which never touches the
    /// sink — hence it hangs off [`Sink`].
    ///
    /// [`collect`]: Sink::collect
    ///
    /// # Safety
    /// `range` must be a live allocation owned by `allocator` that no other live
    /// handle will also free.
    unsafe fn dealloc<A: BStackRaiiAllocator>(allocator: &A, range: BStackRange) -> io::Result<()> {
        // Inside a WAL-backed teardown, defer the free: collect the slice so the
        // whole subtree commits (and frees) as one crash-atomic transaction — but
        // ONLY if it belongs to the file that installed the sink (see [`collect`]).
        // Otherwise it falls through to an eager free in its own file.
        if Self::collect(allocator, range) {
            return Ok(());
        }
        let owned: BStackOwnedSlice<'_, A> =
            unsafe { BStackOwnedSlice::from_raw_range(allocator, range) };
        allocator.dealloc(owned).map_err(|e| e.source)
    }
}

/// RAII guard for an installed [`Sink`]: its `Drop` clears the thread-local on
/// **every** exit — including an unwind out of `bstack_drop`. A bare `.take()` after
/// the teardown call is skipped on panic, leaving the sink `Some`; the next
/// top-level teardown on this thread would then hit [`Sink::is_active`], misdetect a
/// *nested* call, and silently funnel every free into the stale (never-committed)
/// sink — leaking the whole subtree while returning `Ok` (issue F3). Making the clear
/// structural (owned by this guard) rules that out even if a caught panic unwinds
/// through.
struct SinkGuard;

impl SinkGuard {
    /// Take the collected `(file, range)` frees, leaving the sink empty. The guard's
    /// `Drop` still runs afterwards (an idempotent second clear).
    fn take(&self) -> Vec<(FileId, BStackRange)> {
        TEARDOWN_SINK
            .with(|s| s.borrow_mut().take())
            .map(|(_, v)| v)
            .unwrap_or_default()
    }
}

impl Drop for SinkGuard {
    fn drop(&mut self) {
        TEARDOWN_SINK.with(|s| *s.borrow_mut() = None);
    }
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
            return Err(io_error!(
                "teardown recursion too deep (an owned-reference cycle?)"
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
    if Sink::is_active() {
        return handle.bstack_drop(allocator);
    }
    // No anchor → the allocator opts out of reclamation: plain teardown.
    if allocator.wal_anchor().is_none() {
        return handle.bstack_drop(allocator);
    }
    // The guard clears the sink on *every* exit from here on (including an unwind),
    // ruling out the F3 stale-sink hazard structurally (see [`SinkGuard`]).
    let sink_guard = Sink::install(core::ptr::from_ref(allocator.stack()));
    let result = handle.bstack_drop(allocator);
    let slices = sink_guard.take();
    // A mid-walk error means the teardown did not finish, so the frees it *did*
    // collect must NOT be committed: freeing a partial set while the still-linked
    // parent points at those ranges is an observable torn structure a retry then
    // double-frees. Discard the collected frees — nothing is freed, the structure
    // is intact, and the caller can retry (the crate's "corruption degrades to a
    // reclaimable leak, never a torn structure" baseline). Only the collect-and-
    // commit *frees* are undone this way; refcount decrements a `strong` child's
    // teardown already applied are not collected here and remain a retry hazard.
    result?;
    // Commit the collected subtree frees — bulk-or-WAL dispatch lives in
    // [`wal::commit_frees`], shared with the RTTI interpreter's `commit_home_frees`.
    crate::io_core::commit_frees(allocator, slices)
}

/// Free a raw block range by reconstructing an owned slice and delegating to the
/// allocator. The central sink the generated `bstack_drop` code funnels through,
/// since ranges carry no allocator of their own — a thin public entry over the
/// sink-aware `Sink::dealloc` (kept a free function because generated
/// `bstack_drop` code and every collection call it by this name).
///
/// # Safety
/// `range` must be a live allocation owned by `allocator` that no other live
/// handle will also free.
pub unsafe fn dealloc_range<A: BStackRaiiAllocator>(
    allocator: &A,
    range: BStackRange,
) -> io::Result<()> {
    unsafe { Sink::dealloc(allocator, range) }
}
