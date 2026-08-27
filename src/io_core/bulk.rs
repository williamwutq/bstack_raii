//! Sequential fallbacks for [`BStackRaiiAllocator::alloc_many`] /
//! [`BStackRaiiAllocator::free_many`].
//!
//! ## How bulk dispatch works
//!
//! bstack provides atomic [`BStackBulkAllocator::alloc_bulk`] /
//! [`dealloc_bulk`](bstack::BStackBulkAllocator::dealloc_bulk), but only some
//! allocators implement it, and **all** of `bstack_raii` is generic over
//! `A: BStackRaiiAllocator`. On stable Rust a generic function cannot dispatch on
//! whether its concrete `A` *also* implements `BStackBulkAllocator` (that needs the
//! nightly `specialization` feature; autoref specialization collapses to the
//! fallback in generic code), so we don't try. Instead `alloc_many` / `free_many`
//! are **provided methods on [`BStackRaiiAllocator`]**: the default bodies are these
//! sequential helpers, and each bulk-capable allocator *overrides* them to call
//! `alloc_bulk` / `dealloc_bulk` (see the impls in [`crate::io_core::wal`]). Ordinary trait
//! dispatch then picks the override at monomorphization — the fast path flows
//! through generic code, and non-bulk allocators (e.g. `FirstFit`) keep the
//! sequential path — with no bound leaking onto the public API.
//!
//! Atomicity of the fallback:
//!
//! * each individual `alloc` / `dealloc` **is** crash-atomic (allocator contract);
//! * the *set* is not — a crash mid-sequence orphans the blocks done so far (a
//!   leak, never a torn structure), which the WAL layer reclaims.

use std::error::Error;
use std::{fmt, io};

use bstack::{BStackBulkAllocator, BStackOwnedSlice, BStackRange};

use crate::BStackRaiiAllocator;
use crate::io_core::dealloc_range;

/// The error a partial [`free_many`](BStackRaiiAllocator::free_many) returns: the
/// first underlying failure, plus **every range whose free did not cleanly
/// complete**.
///
/// The sequential fallback frees every range it can and collects the ones whose
/// `dealloc` returned an error, rather than stopping at the first error and
/// leaving an unknown suffix allocated. [`unfreed`](Self::unfreed)
/// is a **superset** of the ranges still allocated: every range genuinely still
/// live is named (so the caller never silently leaks one), but a single
/// `dealloc` is not atomic against a mid-operation I/O fault, so a named range
/// may in fact have been reclaimed before a follow-up step failed. Treat the set
/// as "needs checking": account for these ranges, and verify liveness before
/// re-freeing (a blind retry of the whole set could double-free a
/// freed-but-reported one).
///
/// It is wrapped in the returned [`io::Error`] as the source, so a caller that
/// only propagates keeps the plain error and a caller that wants to recover
/// downcasts:
///
/// ```ignore
/// if let Some(fme) = err.get_ref().and_then(|e| e.downcast_ref::<FreeManyError>()) {
///     for r in fme.unfreed() { /* retry / account for r */ }
/// }
/// ```
#[derive(Debug)]
pub struct FreeManyError {
    source: io::Error,
    unfreed: Vec<BStackRange>,
}

impl FreeManyError {
    /// Build from the first underlying error and the ranges left unfreed.
    pub(crate) fn from_parts(source: io::Error, unfreed: Vec<BStackRange>) -> Self {
        Self { source, unfreed }
    }

    /// The ranges that were **not** freed and are still allocated.
    pub fn unfreed(&self) -> &[BStackRange] {
        &self.unfreed
    }
}

impl fmt::Display for FreeManyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "free_many freed all it could; {} range(s) left unfreed: {}",
            self.unfreed.len(),
            self.source
        )
    }
}

impl Error for FreeManyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

impl crate::handback::HandBack for FreeManyError {
    fn io(&self) -> &io::Error {
        &self.source
    }
}

impl From<FreeManyError> for io::Error {
    fn from(e: FreeManyError) -> io::Error {
        io::Error::new(e.source.kind(), e)
    }
}

/// Sequential fallback for [`BStackRaiiAllocator::alloc_many`]: allocate one block
/// per entry in `sizes`, in order. On any failure the blocks already allocated are
/// freed (reverse order) before the error is returned, so a partial allocation
/// never leaks within the call.
pub(crate) fn seq_alloc_many<A: BStackRaiiAllocator>(
    allocator: &A,
    sizes: &[u64],
) -> io::Result<Vec<BStackRange>> {
    let mut out = Vec::with_capacity(sizes.len());
    for &size in sizes {
        match allocator.alloc(size) {
            Ok(slice) => out.push(slice.as_range()),
            Err(e) => {
                for r in out.into_iter().rev() {
                    // SAFETY: our own fresh, unshared allocations.
                    let _ = unsafe { dealloc_range(allocator, r) };
                }
                return Err(e);
            }
        }
    }
    Ok(out)
}

/// Sequential fallback for [`BStackRaiiAllocator::free_many`]: free every range in
/// turn, **continuing past a failure** and collecting the ranges it could not
/// free rather than stopping at the first error. On any failure
/// it returns a [`FreeManyError`] (wrapped in the [`io::Error`]) naming every
/// unfreed range, so the caller can retry exactly those without double-freeing
/// the ones already reclaimed.
pub(crate) fn seq_free_many<A: BStackRaiiAllocator>(
    allocator: &A,
    ranges: impl IntoIterator<Item = BStackRange>,
) -> io::Result<()> {
    let mut first_err: Option<io::Error> = None;
    let mut unfreed: Vec<BStackRange> = Vec::new();
    for r in ranges {
        // SAFETY: the caller's contract, as for `dealloc_range`.
        if let Err(e) = unsafe { dealloc_range(allocator, r) } {
            if first_err.is_none() {
                first_err = Some(e);
            }
            unfreed.push(r);
        }
    }
    match first_err {
        None => Ok(()),
        Some(source) => Err(FreeManyError { source, unfreed }.into()),
    }
}

/// The bulk override shared by every [`BStackBulkAllocator`]: allocate all `sizes`
/// as one atomic [`alloc_bulk`](BStackBulkAllocator::alloc_bulk) and hand back their
/// ranges. Either all blocks are allocated or none is (and the store is unchanged);
/// a crash mid-op is reclaimed by the allocator's own recovery, so this needs no WAL.
/// The `atomic_bulk`-path counterpart of [`seq_alloc_many`], for a
/// [`BStackBulkAllocator`]; wired in by `BStackRaiiAllocator`'s bulk overrides.
pub(crate) fn bulk_alloc_many<A>(allocator: &A, sizes: &[u64]) -> io::Result<Vec<BStackRange>>
where
    A: BStackRaiiAllocator + BStackBulkAllocator,
{
    let slices = allocator.alloc_bulk(sizes)?;
    Ok(slices.into_iter().map(|s| s.as_range()).collect())
}

/// The bulk override shared by every [`BStackBulkAllocator`]: free all `ranges` as
/// one atomic [`dealloc_bulk`](BStackBulkAllocator::dealloc_bulk). Reconstructs an
/// owned slice per range (as [`crate::io_core::teardown::dealloc_range`] does) and
/// frees them together; on failure the error's `source` is surfaced (the un-freed
/// handles it carries back are dropped — they are non-RAII, so dropping does not
/// double-free). The `atomic_bulk`-path counterpart of [`seq_free_many`].
pub(crate) fn bulk_free_many<A>(
    allocator: &A,
    ranges: impl IntoIterator<Item = BStackRange>,
) -> io::Result<()>
where
    A: BStackRaiiAllocator + BStackBulkAllocator,
{
    let ranges: Vec<BStackRange> = ranges.into_iter().collect();
    let handles = ranges
        .iter()
        // SAFETY: each range is a live allocation owned by `allocator` that no other
        // live handle will also free (the `free_many` contract).
        .map(|&r| unsafe { BStackOwnedSlice::from_raw_range(allocator, r) })
        .collect::<Vec<_>>();
    // `dealloc_bulk` is atomic (all-or-nothing), so on failure every range is
    // still allocated — report them all as unfreed.
    allocator
        .dealloc_bulk(handles)
        .map_err(|e| FreeManyError::from_parts(e.source, ranges).into())
}
