//! FUZZ.md **H3** — the fault-injection / crash-consistency layer (**O6**) over
//! H2's `Shared` (`rc, weak`) lifecycle. Requires `--features fault-injection`
//! (and a debug build, which `cargo test` already is) — see `bstack::fault`.
//!
//! The oracle here is **not** leak-freedom: a fault landing between the two
//! non-WAL-protected phases of a strong/weak release (`strong_release_ctrl` in
//! `src/handle.rs` frees the data block, then separately frees the control
//! block once weak also hits zero) is expected to be able to leak the control
//! block — that two-phase release isn't wrapped in a WAL transaction, so
//! there's nothing for `bstack_raii::finish` to recover here, and "may leak,
//! must never corrupt" is the crate's own stated baseline (see FUZZ.md's O2).
//! What this sweep actually checks (**O6's real content**): a fault at *any*
//! point during teardown of a `try_clone`d + `downgrade`d `Shared` must never
//! corrupt the allocator's bookkeeping — `DebugCheckingAllocator` panics
//! in-line on any overlap or double-free, so surviving the whole sweep with no
//! panic *is* the assertion — and the file must still be usable afterward: a
//! fresh, fault-free `Shared` lifecycle post-recovery must succeed cleanly.
//!
//! Deliberately excludes `GhostTree`: `tests/model_fuzz.rs::op_sequence_ghost_tree`
//! already documents a `(rc,weak)`+`GhostTree` corruption-adjacent leak bug with
//! *no* fault injection involved; mixing that in here would confound the signal
//! this test is after (fault-induced outcomes specifically).
#![cfg(feature = "fault-injection")]

#[path = "hypercube/common.rs"]
mod common;
#[path = "hypercube/fixtures.rs"]
mod fixtures;

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bstack::BStackAllocator;
use bstack::fault::FaultPolicy;
use bstack_raii::TryClone;
use common::TempStack;
use fixtures::Shared;

/// Fails the `target`-th fault-eligible op once, then goes quiet — one
/// fault per run, landing at a different point in the teardown each sweep.
struct FailAtSeq {
    seen: AtomicU64,
    target: u64,
}

impl FaultPolicy for FailAtSeq {
    fn next_fault(&self, _op: &'static str, _seq: u64) -> Option<io::Error> {
        if self.seen.fetch_add(1, Ordering::SeqCst) == self.target {
            Some(io::Error::other("injected fault"))
        } else {
            None
        }
    }
}

#[test]
fn shared_teardown_fault_sweep_never_corrupts() {
    // Generous upper bound: comfortably covers every settable op in a full
    // strong+strong+weak teardown (each drop is at most a couple of `set`s /
    // `fetch_sub`s) — targets past the real count are simply never hit, so
    // `next_fault` never fires and the run is equivalent to a clean one.
    for target in 0..24u64 {
        let temp = TempStack::new();
        let alloc = temp.debug_checking_allocator();
        let stack = alloc.stack();

        // Build: two strong refs + one weak, so teardown exercises the full
        // strong-then-weak two-phase release strong_release_ctrl performs.
        let rc0 = Shared::new(&alloc, 1).unwrap();
        let rc1 = rc0.try_clone().unwrap();
        let w = rc0.downgrade().unwrap();

        stack.set_fault_policy(Some(Arc::new(FailAtSeq {
            seen: AtomicU64::new(0),
            target,
        })));
        // `AutoDrop::drop` (src/teardown.rs) swallows the `io::Result` by the
        // same contract as Rust's own `Drop` — a fault here must never panic,
        // only at worst silently leak. Order matches a real scope-exit drop.
        drop(rc1);
        drop(rc0);
        drop(w);
        stack.set_fault_policy(None);

        // O6: "reopen" (a fresh allocator handle to the same file) + finish
        // any WAL-recorded transaction, then prove the file is still sound —
        // not just "didn't panic during teardown" but "still fully usable" —
        // by running one more clean, unfaulted lifecycle through to teardown.
        drop(alloc);
        let alloc2 = temp.debug_checking_allocator();
        bstack_raii::finish(&alloc2).unwrap();

        let rc = Shared::new(&alloc2, 2).unwrap();
        let clone = rc.try_clone().unwrap();
        let weak = rc.downgrade().unwrap();
        drop(clone);
        drop(rc);
        assert!(
            weak.upgrade().unwrap().is_none(),
            "post-fault recovery left a Shared object in an inconsistent \
             refcount state (upgrade should see strong == 0) at target {target}"
        );
        drop(weak);
    }
}
