//! A `BStackRaiiAllocator` that records every range it frees.
//!
//! FUZZ.md's **O5** compares the two teardown implementations by the *set of ranges
//! each one frees*. `DebugCheckingAllocator` cannot serve: it panics on a double
//! free rather than reporting, so a differential run stops at the first divergence
//! instead of enumerating it. This wrapper is the reporting counterpart.
use std::cell::RefCell;
use std::io;

use bstack::{BStack, BStackAllocError, BStackAllocator, BStackOwnedSlice, BStackRange};
use bstack_raii::{BStackRaiiAllocator, STD_WAL_ANCHOR};

pub struct Recorder {
    inner: bstack::FirstFitBStackAllocator,
    freed: RefCell<Vec<BStackRange>>,
    allocated: RefCell<Vec<BStackRange>>,
}

impl Recorder {
    pub fn new(stack: BStack) -> io::Result<Self> {
        Ok(Recorder {
            inner: bstack::FirstFitBStackAllocator::new(stack)?,
            freed: RefCell::new(Vec::new()),
            allocated: RefCell::new(Vec::new()),
        })
    }

    /// Every range freed so far, sorted — the O5 comparison key.
    pub fn freed(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self
            .freed
            .borrow()
            .iter()
            .map(|r| (r.start(), r.len()))
            .collect();
        v.sort_unstable();
        v
    }

    /// Every range allocated so far, sorted.
    pub fn allocated(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self
            .allocated
            .borrow()
            .iter()
            .map(|r| (r.start(), r.len()))
            .collect();
        v.sort_unstable();
        v
    }

    /// Ranges allocated but never freed — a leak set, independent of file length
    /// (so it works on backends where O1's exact-length check does not).
    pub fn live(&self) -> Vec<(u64, u64)> {
        let freed = self.freed();
        self.allocated()
            .into_iter()
            .filter(|a| !freed.contains(a))
            .collect()
    }

    /// A mark into the allocation *log*. Pair with [`allocated_since`].
    ///
    /// Set difference on `(offset, len)` pairs is not a sound way to ask "what was
    /// allocated after this point": a later allocation can reuse the exact slot of an
    /// earlier freed one, and the duplicate pair is then silently cancelled out.
    /// The log is ordered, so take a suffix instead.
    pub fn mark(&self) -> usize {
        self.allocated.borrow().len()
    }

    /// Everything allocated after `mark`, sorted.
    pub fn allocated_since(&self, mark: usize) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self.allocated.borrow()[mark..]
            .iter()
            .map(|r| (r.start(), r.len()))
            .collect();
        v.sort_unstable();
        v
    }

    /// A mark into the *free* log, for [`freed_since`].
    pub fn free_mark(&self) -> usize {
        self.freed.borrow().len()
    }

    /// Everything freed after `mark`, sorted.
    pub fn freed_since(&self, mark: usize) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self.freed.borrow()[mark..]
            .iter()
            .map(|r| (r.start(), r.len()))
            .collect();
        v.sort_unstable();
        v
    }

    pub fn reset_log(&self) {
        self.freed.borrow_mut().clear();
        self.allocated.borrow_mut().clear();
    }
}

impl BStackAllocator for Recorder {
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
        self.allocated.borrow_mut().push(r);
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
                self.freed.borrow_mut().push(old);
                self.allocated.borrow_mut().push(r);
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
        match self.inner.dealloc(inner_h) {
            Ok(()) => {
                self.freed.borrow_mut().push(r);
                Ok(())
            }
            Err(e) => Err(match e.handle {
                Some(h) => BStackAllocError::with_handle(e.source, unsafe {
                    BStackOwnedSlice::from_raw_range(self, h.as_range())
                }),
                None => BStackAllocError::lost(e.source),
            }),
        }
    }
}

// SAFETY: forwards every allocation to `FirstFitBStackAllocator`, which upholds the
// null niche at payload offset 0 and reserves the `[8, 16)` WAL anchor slot; this
// wrapper only observes the ranges passing through.
unsafe impl BStackRaiiAllocator for Recorder {
    fn wal_anchor(&self) -> Option<u64> {
        Some(STD_WAL_ANCHOR)
    }
}
