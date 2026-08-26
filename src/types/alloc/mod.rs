//! The crate's **allocator capability**: [`BStackRaiiAllocator`], the bound every
//! `bstack_raii` operation is generic over, and (in [`host`]) its object-safe,
//! cross-file projection.
//!
//! These are *semantic types* — the vocabulary the whole layer is written against —
//! not I/O mechanism, so they live under [`crate::types`] rather than in a `mechanism`
//! module.

pub mod host;

use std::io;

use bstack::{BStackOwnedSliceAllocator, BStackRange};

use crate::registry::FileId;

/// The allocator every `bstack_raii` operation is bound on: a
/// [`BStackOwnedSliceAllocator`] the layer can soundly build owning handles on top
/// of. It is the crate-wide allocator capability — constructors, `try_clone_in`,
/// `bstack_drop`, and every stdlib collection require it.
///
/// Beyond its supertrait it carries two things the layer relies on: the **null
/// niche** at payload offset 0 (a hard safety requirement, see below) and an
/// **optional** WAL anchor slot. The anchor is what lets `try_clone_in` /
/// `bstack_drop` reclaim orphaned allocations on the next open **automatically**
/// (they read [`wal_anchor`](Self::wal_anchor) directly); `None` (the default)
/// means "no reclamation" — those ops behave exactly as before, minus the
/// crash-orphan cleanup. Every bstack-provided allocator implements this trait; a
/// custom allocator that upholds the null niche adds a one-line
/// `unsafe impl BStackRaiiAllocator for MyAlloc {}` (defaulting to `None`, or
/// returning `Some(slot)` if it reserves a stable slot).
///
/// The WAL machinery it feeds lives in [`crate::io_core::wal`]; the trait itself is
/// the crate's front-door allocator bound, and lives here among the semantic types.
///
/// # Safety
///
/// An implementor asserts **both** of the following:
///
/// 1. **Null niche.** The allocator **never** hands out a live slice whose
///    `start()` is `0`. `bstack_raii` reserves payload offset 0 as its universal
///    null sentinel — a `0` offset means "none" everywhere in the layer (an absent
///    handle / [`Option`] niche, a dead weak reference, "no WAL block", …). An
///    allocator that could return offset 0 is **unsound** with this crate: a real
///    allocation would be indistinguishable from null. (Every bstack allocator
///    satisfies this: each keeps a reserved region at payload offset 0 that it
///    never allocates from.)
///
/// 2. **WAL anchor (only when returning `Some(off)`).** `[off, off + 8)` is a
///    stable, persistent 8-byte region the allocator **never** hands out via
///    `alloc` and **never** uses for its own metadata, and that survives across
///    open/close. `bstack_raii` stores the current WAL block's offset there
///    (`0` = none). Returning `None` asserts nothing beyond (1).
pub unsafe trait BStackRaiiAllocator: BStackOwnedSliceAllocator {
    fn wal_anchor(&self) -> Option<u64> {
        None
    }

    /// The [`FileId`](crate::registry::FileId) whose file this allocator's frees
    /// belong to, used to **tag WAL teardown entries**. A normal file-owning
    /// allocator represents *its own* file, so it returns [`FileId::SELF`](crate::registry::FileId::SELF)
    /// (`0`) — its WAL entries mean "this file". The cross-file teardown adapter
    /// [`ForeignHostAllocator`](crate::registry::ForeignHostAllocator) overrides this
    /// with the foreign file's id, so a free collected while tearing down a foreign
    /// subtree is recorded against — and, on recovery, reclaimed in — *that* file
    /// (see [`crate::io_core::wal`]'s `free_recorded`). Callers other than the
    /// teardown WAL have no reason to read this.
    fn wal_file_id(&self) -> FileId {
        FileId::SELF
    }

    /// Allocate one block per length in `sizes`, returning their ranges in order.
    ///
    /// The default is a **sequential** fallback: each `alloc` is individually
    /// crash-atomic, but the set is not (a crash mid-sequence orphans the blocks
    /// done so far — a leak the WAL layer reclaims, never a torn structure). It
    /// unwinds already-allocated blocks on any failure, so a partial allocation
    /// never leaks *within* the call.
    ///
    /// A bulk-capable allocator ([`bstack::BStackBulkAllocator`]) **overrides** this
    /// to route through the atomic [`alloc_bulk`](bstack::BStackBulkAllocator::alloc_bulk),
    /// so the whole set becomes one crash-atomic operation recovered by the
    /// allocator's own machinery. Ordinary trait dispatch picks the override at
    /// monomorphization, so compound ops generic over `A` get the fast path for free.
    fn alloc_many(&self, sizes: &[u64]) -> io::Result<Vec<BStackRange>> {
        crate::bulk::seq_alloc_many(self, sizes)
    }

    /// Free every range in `ranges`. The default is a **sequential** fallback
    /// (each `dealloc` individually atomic); a bulk-capable allocator overrides it
    /// to route through the atomic [`dealloc_bulk`](bstack::BStackBulkAllocator::dealloc_bulk).
    ///
    /// On a partial failure the fallback **frees every range it can** (rather than
    /// stopping at the first error and leaving an unknown suffix allocated) and
    /// returns a [`FreeManyError`](crate::FreeManyError) (wrapped in the
    /// [`io::Error`]) naming every range whose free did not cleanly complete — a
    /// superset of those still allocated, so nothing is silently leaked.
    /// Downcast the error's source to `FreeManyError` to recover
    /// them.
    ///
    /// # Safety
    /// Each range must be a live allocation owned by `self` that no other live
    /// handle will also free (as for [`crate::io_core::teardown`]'s `dealloc_range`,
    /// whose obligation this method carries range-by-range). `BStackRange::new` is a
    /// safe constructor, so nothing gates the argument but this contract; the safe
    /// ways to free remain [`BStackDrop`](crate::BStackDrop) / [`AutoDrop`](crate::AutoDrop).
    unsafe fn free_many(
        &self,
        ranges: impl IntoIterator<Item = BStackRange>,
    ) -> io::Result<()> {
        crate::bulk::seq_free_many(self, ranges)
    }

    /// Whether this allocator provides **atomic, self-recovering** bulk
    /// alloc/free — i.e. it implements [`bstack::BStackBulkAllocator`] and
    /// overrides [`alloc_many`](Self::alloc_many) / [`free_many`](Self::free_many)
    /// to route through it. Default `false`.
    ///
    /// When `true`, a compound op whose blocks all live in **this** file can free
    /// (or allocate) them as one atomic `dealloc_bulk` / `alloc_bulk` and **skip the
    /// WAL** entirely: the WAL exists to emulate atomic batch alloc/free for
    /// allocators that lack it, and wrapping an already-atomic bulk op in it is both
    /// redundant and unsound (the allocator's crash-recovery direction is opaque, so
    /// a WAL retry on reopen could double-free). A crash mid-bulk is left to the
    /// allocator's own recovery — consistent, leak-at-worst, exactly the guarantee
    /// the WAL would have provided. Cross-file (mixed [`FileId`](crate::registry::FileId))
    /// batches fall back to the WAL for its registry routing.
    fn atomic_bulk(&self) -> bool {
        false
    }
}

/// A thread-shareable [`BStackRaiiAllocator`] — the bound a file's live host must
/// satisfy to be stored in (and resolved from) the [registry](crate::registry) across
/// threads.
///
/// Purely a convenience alias (`BStackRaiiAllocator + Send + Sync`, blanket-impl'd)
/// so call sites don't repeat the `+ Send + Sync` every time. It is **not** what the
/// registry stores: `BStackRaiiAllocator` is not object-safe (`BStackAllocator: Sized`,
/// plus the GAT `Allocated<'a>` and `alloc -> Self::Allocated<'_>`), so there is no
/// `dyn SyncBStackRaiiAllocator`. [`BStackRaiiHost`](host::BStackRaiiHost) is its
/// object-safe projection, and what actually goes behind the `Arc<dyn …>`.
pub trait SyncBStackRaiiAllocator: BStackRaiiAllocator + Send + Sync {}
impl<A: BStackRaiiAllocator + Send + Sync> SyncBStackRaiiAllocator for A {}
