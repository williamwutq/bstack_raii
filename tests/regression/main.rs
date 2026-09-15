//! The crate's regression suite — one binary over all former
//! per-finding test files, grouped by theme (see each submodule).

// Shared with `o5_differential`'s `mod recorder`: a `BStackRaiiAllocator` wrapper
// that records every range its `dealloc` genuinely completes, for tests that need
// non-destructive ground truth about what was actually freed (re-probing an
// original offset via `dealloc_range` is unsound once two adjacent frees coalesce).
#[path = "../o5/recorder.rs"]
#[allow(dead_code)]
mod recorder;

mod atomicity;
mod collections;
mod errorpaths;
mod foreign;
mod identity;
mod rtti;
mod schema;
