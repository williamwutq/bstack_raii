//! Operation: **deep clone** — `try_clone_in` produces an independent copy (distinct
//! child blocks), tearing the clone down never disturbs the original, and the clone
//! itself reclaims cleanly.

use bstack::BStackAllocator;
use bstack_raii::{BStackBlock, BStackDrop, TryCloneIn};

use crate::common::{TempStack, assert_teardown_reclaims};
use crate::fixtures::*;

#[test]
fn block_sink_clone_is_deep_and_independent() {
    let tmp = TempStack::new();
    let a = tmp.allocator();
    let stack = a.stack();

    let h = block_sink(&a).unwrap();
    let c = h.try_clone_in(&a).unwrap();

    // Same values …
    assert_eq!(
        c.handle().get_owned(stack).unwrap().get_v(stack).unwrap(),
        10
    );
    // … but a distinct child block (deep copy, not an alias).
    let orig = h.handle().get_owned(stack).unwrap().range().start();
    let clone = c.handle().get_owned(stack).unwrap().range().start();
    assert_ne!(orig, clone);

    // Dropping the clone leaves the original fully intact.
    c.bstack_drop(&a).unwrap();
    assert_eq!(
        h.handle().get_owned(stack).unwrap().get_v(stack).unwrap(),
        10
    );
    h.bstack_drop(&a).unwrap();
}

#[test]
fn block_sink_clone_reclaims() {
    let tmp = TempStack::new();
    let a = tmp.allocator();
    let src = block_sink(&a).unwrap();
    assert_teardown_reclaims(&a, || src.try_clone_in(&a).unwrap());
    src.bstack_drop(&a).unwrap();
}

// The richest interaction fixture (owned / strong / embed + vec / array + nested arrays
// + vec-of-arr + arr-of-vec + strong vec/arr) under DebugCheckingAllocator, which PANICS
// in-line on a double-free / overlapping free. This is a strictly stronger check than the
// FirstFit tests above: teardown SWALLOWS dealloc errors, so on plain FirstFit a
// double-free (an aliasing clone that shares a leaf with the original) returns an Err
// that never reaches `bstack_drop().unwrap()` — only this oracle surfaces it. Clone, then
// tear down BOTH copies: any leaf the clone failed to deep-copy double-frees here.
#[test]
fn block_sink_clone_and_teardown_no_double_free() {
    let tmp = TempStack::new();
    let a = tmp.debug_checking_allocator();
    let h = block_sink(&a).unwrap();
    let c = h.try_clone_in(&a).unwrap();
    c.bstack_drop(&a).unwrap();
    h.bstack_drop(&a).unwrap();
}

// The same double-free oracle applied to the OTHER codegen paths (embed, foreign-SELF,
// enum, whole-value-mut), each with its own drop/clone token stream — an aliasing clone
// or a teardown that visits a child twice panics in-line here where a FirstFit run would
// swallow it.
#[test]
fn dbg_sinks_clone_and_teardown_no_double_free() {
    let tmp = TempStack::new();
    let a = tmp.debug_checking_allocator();

    let c1 = embed_sink(&a).unwrap();
    let c1c = c1.try_clone_in(&a).unwrap();
    c1c.bstack_drop(&a).unwrap();
    c1.bstack_drop(&a).unwrap();

    let cf = foreign_self_sink(&a).unwrap();
    let cfc = cf.try_clone_in(&a).unwrap();
    cfc.bstack_drop(&a).unwrap();
    cf.bstack_drop(&a).unwrap();

    let ce = enum_owned(&a).unwrap();
    let cec = ce.try_clone_in(&a).unwrap();
    cec.bstack_drop(&a).unwrap();
    ce.bstack_drop(&a).unwrap();

    let cm = mut_sink(&a).unwrap();
    let cmc = cm.try_clone_in(&a).unwrap();
    cmc.bstack_drop(&a).unwrap();
    cm.bstack_drop(&a).unwrap();
}

#[test]
fn embed_sink_clone_is_independent() {
    let tmp = TempStack::new();
    let a = tmp.allocator();
    let stack = a.stack();

    let h = embed_sink(&a).unwrap();
    let c = h.try_clone_in(&a).unwrap();
    // The embedded child's own owned grandchild is deep-copied, not aliased.
    let orig = h
        .handle()
        .get_child()
        .unwrap()
        .get_owned(stack)
        .unwrap()
        .range()
        .start();
    let clone = c
        .handle()
        .get_child()
        .unwrap()
        .get_owned(stack)
        .unwrap()
        .range()
        .start();
    assert_ne!(orig, clone);
    c.bstack_drop(&a).unwrap();
    assert_eq!(
        h.handle()
            .get_child()
            .unwrap()
            .get_owned(stack)
            .unwrap()
            .get_v(stack)
            .unwrap(),
        10
    );
    h.bstack_drop(&a).unwrap();
}

#[test]
fn enum_owned_clone_is_deep() {
    let tmp = TempStack::new();
    let a = tmp.allocator();
    let stack = a.stack();

    let e = enum_owned(&a).unwrap();
    let c = e.try_clone_in(&a).unwrap();
    match c.handle().read(&a).unwrap() {
        EnumSinkView::Owned(leaf) => assert_eq!(leaf.get_v(stack).unwrap(), 42),
        _ => panic!("expected Owned"),
    }
    c.bstack_drop(&a).unwrap();
    // Original still readable after the clone is gone.
    match e.handle().read(&a).unwrap() {
        EnumSinkView::Owned(leaf) => assert_eq!(leaf.get_v(stack).unwrap(), 42),
        _ => panic!("expected Owned"),
    }
    e.bstack_drop(&a).unwrap();
}
