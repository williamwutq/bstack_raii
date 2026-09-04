//! Operation: **refcount** — shared (`rc` / `rc, weak`) semantics, exercised through
//! the public handle API (exact-count assertions that peek at the control block stay
//! as unit tests in `src/tests.rs`).

use bstack::BStackAllocator;
use bstack_raii::{BStackBlock, TryClone};

use crate::common::TempStack;
use crate::fixtures::{Shared, rc_sink};

#[test]
fn strong_clone_keeps_alive_until_last_drop() {
    let tmp = TempStack::new();
    let a = tmp.allocator();
    let stack = a.stack();

    let rc = Shared::new(&a, 5).unwrap(); // strong = 1
    let rc2 = rc.try_clone().unwrap(); // strong = 2 (shares, not deep-copied)
    drop(rc); // strong = 1, still alive
    assert_eq!(rc2.handle().get_v(stack).unwrap(), 5);
    drop(rc2); // strong = 0, freed
}

#[test]
fn weak_upgrade_reflects_strong_liveness() {
    let tmp = TempStack::new();
    let a = tmp.allocator();

    let rc = Shared::new(&a, 7).unwrap();
    let w = rc.downgrade().unwrap();
    assert!(w.upgrade().unwrap().is_some()); // strong alive → upgradeable
    drop(rc); // last strong gone
    assert!(w.upgrade().unwrap().is_none()); // no longer upgradeable
    drop(w); // releases the last weak (frees the control block)
}

#[test]
fn shared_container_clone_shares() {
    let tmp = TempStack::new();
    let a = tmp.allocator();
    let stack = a.stack();

    // An `rc` block is duplicated by a refcount bump, not a deep copy.
    let rc = rc_sink(&a).unwrap();
    let rc2 = rc.try_clone().unwrap();
    let off1 = rc.handle().range().start();
    let off2 = rc2.handle().range().start();
    assert_eq!(off1, off2); // same block (shared)
    drop(rc);
    assert_eq!(rc2.handle().get_pod(stack).unwrap(), 7); // still alive
    drop(rc2);
}

// Refcount teardown under DebugCheckingAllocator (a live double-free oracle): the strong
// path frees the data (and, for `rc, weak`, the control only once the last weak is also
// gone); the weak path frees the control when last. A control block freed by BOTH the
// last strong and the last weak — the classic rc/weak double-free — panics here, where a
// plain FirstFit run's swallowed teardown error would hide it.
#[test]
fn dbg_refcount_teardown_no_double_free() {
    let tmp = TempStack::new();
    let a = tmp.debug_checking_allocator();

    // Strong share, drop both: the block is freed exactly once at strong 0.
    let rc = Shared::new(&a, 5).unwrap();
    let rc2 = rc.try_clone().unwrap();
    drop(rc);
    drop(rc2);

    // rc + weak: last strong frees the data and keeps the control (a weak is alive);
    // the last weak then frees the control. Neither block may be freed twice.
    let rc = Shared::new(&a, 7).unwrap();
    let w = rc.downgrade().unwrap();
    drop(rc);
    drop(w);

    // A shared block carrying its own children, freed at strong 0.
    let s = rc_sink(&a).unwrap();
    let s2 = s.try_clone().unwrap();
    drop(s);
    drop(s2);
}
