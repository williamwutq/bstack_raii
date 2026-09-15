//! [`PeriodicCoalesceAllocator`]: a throttled [`coalesce_after_op`](BStackRaiiAllocator::coalesce_after_op)
//! wrapper for [`bstack::SegregatedBStackAllocator`].

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use bstack::{
    BStack, BStackAllocError, BStackAllocator, BStackBulkAllocError, BStackBulkAllocator,
    BStackOwnedSlice, BStackRange, SegregatedBStackAllocator,
};

use crate::primitives::NonNullOffset;
use crate::registry::FileId;
use crate::types::alloc::BStackRaiiAllocator;

/// Wraps a [`SegregatedBStackAllocator`], throttling the automatic post-teardown
/// coalesce that [`BStackRaiiAllocator::coalesce_after_op`] drives.
///
/// `SegregatedBStackAllocator::coalesce` strides the *whole* arena on every call
/// (see its doc) — cheap relative to the I/O it's saving when there's real
/// fragmentation to merge, but pure overhead on a call that finds nothing (which,
/// for a workload of many differently-sized live objects, is empirically the
/// common case — most single-block frees never land next to another already-free
/// block). This wrapper counts calls with an atomic counter (wrapping on overflow
/// — only the residue mod `interval` matters) and only forwards every
/// `interval`-th one to the real `coalesce`, trading a bounded amount of extra
/// fragmentation for far fewer full-arena scans.
///
/// Every other operation forwards straight through to the wrapped allocator —
/// this changes only the coalesce cadence, nothing else about `bstack_raii`'s
/// behavior on `Segregated`.
pub struct PeriodicCoalesceAllocator {
    inner: SegregatedBStackAllocator,
    interval: u64,
    counter: AtomicU64,
}

impl PeriodicCoalesceAllocator {
    /// Wrap `inner`; the automatic hook actually coalesces every `interval`-th
    /// unit of work it's called for (`interval == 0` is treated as `1` — coalesce
    /// on every call, the same cadence as not wrapping at all).
    #[must_use]
    pub fn new(inner: SegregatedBStackAllocator, interval: u64) -> Self {
        Self {
            inner,
            interval: interval.max(1),
            counter: AtomicU64::new(0),
        }
    }

    /// The wrapped allocator.
    #[must_use]
    pub fn inner(&self) -> &SegregatedBStackAllocator {
        &self.inner
    }

    /// Unwrap, discarding the call counter.
    #[must_use]
    pub fn into_inner(self) -> SegregatedBStackAllocator {
        self.inner
    }
}

impl BStackAllocator for PeriodicCoalesceAllocator {
    type Error = io::Error;
    type Allocated<'a> = BStackOwnedSlice<'a, Self>;

    fn stack(&self) -> &BStack {
        self.inner.stack()
    }

    fn into_stack(self) -> BStack {
        self.inner.into_stack()
    }

    fn alloc(&self, len: u64) -> io::Result<BStackOwnedSlice<'_, Self>> {
        let r = self.inner.alloc(len)?.as_range();
        // SAFETY: `r` is a fresh live allocation from `inner`, which this wrapper
        // forwards every free back to; rebinding it to `self` keeps that routing.
        Ok(unsafe { BStackOwnedSlice::from_raw_range(self, r) })
    }

    fn realloc<'a>(
        &'a self,
        handle: BStackOwnedSlice<'a, Self>,
        new_len: u64,
    ) -> Result<BStackOwnedSlice<'a, Self>, BStackAllocError<'a, Self>> {
        let old = handle.as_range();
        // SAFETY: `old` is the live allocation `handle` named, owned by `inner`.
        let inner_h = unsafe { BStackOwnedSlice::from_raw_range(&self.inner, old) };
        match self.inner.realloc(inner_h, new_len) {
            Ok(s) => {
                let r = s.as_range();
                Ok(unsafe { BStackOwnedSlice::from_raw_range(self, r) })
            }
            Err(e) => Err(match e.handle {
                Some(h) => BStackAllocError::with_handle(e.source, unsafe {
                    BStackOwnedSlice::from_raw_range(self, h.as_range())
                }),
                None => BStackAllocError::lost(e.source),
            }),
        }
    }

    fn dealloc<'a>(
        &'a self,
        handle: BStackOwnedSlice<'a, Self>,
    ) -> Result<(), BStackAllocError<'a, Self>> {
        let r = handle.as_range();
        // SAFETY: `r` is the live allocation `handle` named, owned by `inner`.
        let inner_h = unsafe { BStackOwnedSlice::from_raw_range(&self.inner, r) };
        self.inner.dealloc(inner_h).map_err(|e| match e.handle {
            Some(h) => BStackAllocError::with_handle(e.source, unsafe {
                BStackOwnedSlice::from_raw_range(self, h.as_range())
            }),
            None => BStackAllocError::lost(e.source),
        })
    }
}

impl BStackBulkAllocator for PeriodicCoalesceAllocator {
    fn alloc_bulk(
        &self,
        lengths: impl AsRef<[u64]>,
    ) -> io::Result<Vec<BStackOwnedSlice<'_, Self>>> {
        let slices = self.inner.alloc_bulk(lengths)?;
        Ok(slices
            .into_iter()
            .map(|s| {
                let r = s.as_range();
                // SAFETY: as `alloc`, above.
                unsafe { BStackOwnedSlice::from_raw_range(self, r) }
            })
            .collect())
    }

    fn dealloc_bulk<'a>(
        &'a self,
        handles: impl IntoIterator<Item = BStackOwnedSlice<'a, Self>>,
    ) -> Result<(), BStackBulkAllocError<'a, Self>> {
        let ranges: Vec<BStackRange> = handles.into_iter().map(|h| h.as_range()).collect();
        let inner_handles = ranges
            .iter()
            // SAFETY: as `dealloc`, above.
            .map(|&r| unsafe { BStackOwnedSlice::from_raw_range(&self.inner, r) })
            .collect::<Vec<_>>();
        self.inner.dealloc_bulk(inner_handles).map_err(|e| {
            let handles = e
                .handles
                .into_iter()
                .map(|h| {
                    let r = h.as_range();
                    // SAFETY: as `dealloc`, above — these are the ranges the inner
                    // call reports as still owned by the caller.
                    unsafe { BStackOwnedSlice::from_raw_range(self, r) }
                })
                .collect();
            BStackBulkAllocError::with_handles(e.source, handles)
        })
    }
}

// SAFETY: forwards every allocation to `SegregatedBStackAllocator`, whose own
// `BStackRaiiAllocator` impl already asserts the null niche and a stable WAL
// anchor; this wrapper only observes/forwards ranges and adds a throttled
// coalesce, so it upholds the same contract.
unsafe impl BStackRaiiAllocator for PeriodicCoalesceAllocator {
    fn wal_anchor(&self) -> Option<NonNullOffset> {
        self.inner.wal_anchor()
    }

    fn wal_file_id(&self) -> FileId {
        self.inner.wal_file_id()
    }

    fn alloc_many(&self, sizes: &[u64]) -> io::Result<Vec<BStackRange>> {
        crate::io_core::bulk::bulk_alloc_many(self, sizes)
    }

    unsafe fn free_many(&self, ranges: impl IntoIterator<Item = BStackRange>) -> io::Result<()> {
        crate::io_core::bulk::bulk_free_many(self, ranges)
    }

    fn atomic_bulk(&self) -> bool {
        true
    }

    fn coalesce_after_op(&self) -> io::Result<()> {
        // Wrapping is fine: only `n % interval` matters, and `interval` divides
        // cleanly often enough in practice that a rare skipped/early beat right at
        // the u64 wraparound is not worth guarding against.
        let n = self.counter.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        if n % self.interval == 0 {
            self.inner.coalesce()?;
        }
        Ok(())
    }
}
