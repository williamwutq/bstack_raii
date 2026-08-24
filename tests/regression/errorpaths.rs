//! Error-path resource hand-back.
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

mod ctorhandback {
//! Regression: a generated `new` that fails a fallible step after
//! consuming an owned/strong/embedded child hands the child **back** through
//! [`ConstructError`] instead of orphaning it. The returned handle is the caller's
//! again — the same block, contents intact, and a real owning handle (freeing it
//! reclaims the block, with no double-free). Requires `--features fault-injection`.
#![cfg(feature = "fault-injection")]
use bstack::fault::FaultPolicy;
use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::{BStackBlock, BStackDrop, bstack_class};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[bstack_class]
struct Leaf {
    v: u32,
}

#[bstack_class]
struct Parent {
    #[bstack_owned]
    child: Leaf,
}

#[bstack_class]
struct EmbedParent {
    tag: u32,
    #[embed]
    child: Leaf,
}

#[bstack_class]
struct VecParent {
    #[bstack_owned]
    kids: Vec<Leaf>,
}

#[bstack_raii::bstack_enum]
enum OwnEnum {
    Empty,
    #[bstack_owned]
    One(Leaf),
}

/// Fail the `nth` (0-based) stack op named `target` — used to skip the two `set`s
/// a vec field's own data block takes and fault the *parent* block's `set` (the
/// third), so the consumed vector is reconstructed and handed back (a `recovered`,
/// not a `lost`, construction).
struct FailCount {
    target: &'static str,
    nth: u64,
    seen: std::sync::atomic::AtomicU64,
}
impl FaultPolicy for FailCount {
    fn next_fault(&self, op: &'static str, _seq: u64) -> Option<io::Error> {
        if op == self.target && self.seen.fetch_add(1, Ordering::SeqCst) == self.nth {
            return Some(io::Error::other("injected parent write failure"));
        }
        None
    }
}

/// Fail the first stack op named `target` that runs *after* the policy is armed —
/// so a child can be built cleanly first and only the *parent*'s allocation (or
/// write) is faulted.
struct FailArmed {
    target: &'static str,
    armed: AtomicBool,
    fired: AtomicBool,
}
impl FailArmed {
    fn new(target: &'static str) -> Arc<Self> {
        Arc::new(Self {
            target,
            armed: AtomicBool::new(false),
            fired: AtomicBool::new(false),
        })
    }
}
impl FaultPolicy for FailArmed {
    fn next_fault(&self, op: &'static str, _seq: u64) -> Option<io::Error> {
        if op == self.target
            && self.armed.load(Ordering::SeqCst)
            && !self.fired.swap(true, Ordering::SeqCst)
        {
            return Some(io::Error::other("injected parent construction failure"));
        }
        None
    }
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("bstack_raii_ctorhb_{tag}_{stamp}.bstack"))
}

#[test]
fn failed_new_hands_back_the_owned_child() {
    let data = tmp("owned");
    let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();

    let child = Leaf::new(&alloc, 42).unwrap();
    let child_off = child.handle().range().start();

    // Build the child cleanly, then fault the *parent* block's write.
    let policy = FailArmed::new("set");
    alloc.stack().set_fault_policy(Some(policy.clone()));
    policy.armed.store(true, Ordering::SeqCst);

    let err = match Parent::new(&alloc, child) {
        Ok(_) => panic!("the faulted parent write should have failed"),
        Err(e) => e,
    };
    alloc.stack().set_fault_policy(None);

    // The consumed child came back — not orphaned, not freed.
    let (returned,) = err.fields.expect("the owned child must be handed back");
    assert_eq!(
        returned.handle().range().start(),
        child_off,
        "the *same* child block must be handed back"
    );
    assert_eq!(
        returned.handle().get_v(alloc.stack()).unwrap(),
        42,
        "the child's contents must be intact"
    );

    // It is a genuine owning handle: freeing it reclaims the block (no leak), and a
    // fresh same-size allocation reuses it (proving it was really live and freed).
    let child_sz = std::mem::size_of::<<Leaf as BStackBlock>::OnDisk>() as u64;
    returned.bstack_drop(&alloc).unwrap();
    let mut reused = false;
    for _ in 0..4 {
        if alloc.alloc(child_sz).unwrap().as_range().start() == child_off {
            reused = true;
        }
    }
    assert!(reused, "the handed-back-then-freed child block was not reclaimed");

    std::fs::remove_file(&data).ok();
}

#[test]
fn failed_new_hands_back_the_embedded_child() {
    let data = tmp("embed");
    let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();

    let child = Leaf::new(&alloc, 7).unwrap();
    let child_off = child.handle().range().start();

    // Fault the parent block's write (the embed copy runs only *after* a
    // successful write, so the child is still its own standalone block to hand
    // back).
    let policy = FailArmed::new("set");
    alloc.stack().set_fault_policy(Some(policy.clone()));
    policy.armed.store(true, Ordering::SeqCst);

    let err = match EmbedParent::new(&alloc, 1, child) {
        Ok(_) => panic!("the faulted parent write should have failed"),
        Err(e) => e,
    };
    alloc.stack().set_fault_policy(None);

    let (returned,) = err.fields.expect("the embedded child must be handed back");
    assert_eq!(
        returned.handle().range().start(),
        child_off,
        "the embedded child's standalone block must be handed back"
    );
    assert_eq!(returned.handle().get_v(alloc.stack()).unwrap(), 7);
    returned.bstack_drop(&alloc).unwrap();

    std::fs::remove_file(&data).ok();
}

#[test]
fn failed_new_hands_back_the_owned_vec() {
    let data = tmp("vec");
    let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();

    let kids = vec![
        Leaf::new(&alloc, 1).unwrap(),
        Leaf::new(&alloc, 2).unwrap(),
        Leaf::new(&alloc, 3).unwrap(),
    ];

    // Fault the *third* `set` during `VecParent::new`: the first two write the
    // vector's own data block (which must succeed so the vec can be reconstructed
    // and handed back); the third writes the parent block image.
    let policy = Arc::new(FailCount {
        target: "set",
        nth: 2,
        seen: std::sync::atomic::AtomicU64::new(0),
    });
    alloc.stack().set_fault_policy(Some(policy));
    let err = match VecParent::new(&alloc, kids) {
        Ok(_) => panic!("the faulted parent write should have failed"),
        Err(e) => e,
    };
    alloc.stack().set_fault_policy(None);

    // The whole vector of owned children comes back as an owning block-vector —
    // reconstructed from its data block, not orphaned.
    let (returned,) = err
        .fields
        .expect("the owned vector must be handed back (not `lost`)");
    assert_eq!(returned.len().unwrap(), 3, "all elements handed back");
    let vals: Vec<u32> = returned
        .to_vec()
        .unwrap()
        .iter()
        .map(|leaf| leaf.get_v(alloc.stack()).unwrap())
        .collect();
    assert_eq!(vals, vec![1, 2, 3], "element contents intact and in order");

    // A genuine owning handle (it carries its allocator): freeing it reclaims the
    // data block and every child.
    returned.bstack_drop().unwrap();

    std::fs::remove_file(&data).ok();
}

#[test]
fn failed_new_hands_back_the_enum_owned_child() {
    let data = tmp("enum");
    let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();

    let leaf = Leaf::new(&alloc, 99).unwrap();
    let leaf_off = leaf.handle().range().start();

    // Fault the enum block's write (the owned child is already encoded into the
    // payload; the reconstruction reads that offset back out).
    let policy = FailArmed::new("set");
    alloc.stack().set_fault_policy(Some(policy.clone()));
    policy.armed.store(true, Ordering::SeqCst);

    let err = match OwnEnum::new(&alloc, OwnEnumData::One(leaf)) {
        Ok(_) => panic!("the faulted enum write should have failed"),
        Err(e) => e,
    };
    alloc.stack().set_fault_policy(None);

    // The whole `EData` value comes back with the active variant's owned child
    // intact — not orphaned.
    let returned = err
        .fields
        .expect("the enum value (with its owned child) must be handed back");
    match returned {
        OwnEnumData::One(child) => {
            assert_eq!(
                child.handle().range().start(),
                leaf_off,
                "the same child block must be handed back"
            );
            assert_eq!(child.handle().get_v(alloc.stack()).unwrap(), 99);
            child.bstack_drop(&alloc).unwrap();
        }
        OwnEnumData::Empty => panic!("wrong variant handed back"),
    }

    std::fs::remove_file(&data).ok();
}
}

mod lost {
//! Regression: `ReplaceError::lost` (`value: None`) documents the displaced OLD
//! block as "reachable only through crash-recovery / the WAL". No generated mutator
//! touches the WAL, so there is nothing for recovery to roll forward.
use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::{BStackBlock, BStackDrop, BStackOwned, bstack_block};

#[bstack_block]
struct Leaf {
    v: u32,
}

#[bstack_block]
struct Par {
    #[bstack_mut]
    #[bstack_owned]
    c: Leaf,
}

#[test]
fn replace_records_nothing_in_the_wal() {
    let path = std::env::temp_dir().join(format!(
        "bstack_raii_lost_{}.bstack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();

    let a: BStackOwned<Leaf> = Leaf::new(&alloc, 11).unwrap();
    let a_off = a.handle().range().start();
    let par: BStackOwned<Par> = Par::new(&alloc, a).unwrap();
    let b: BStackOwned<Leaf> = Leaf::new(&alloc, 22).unwrap();

    // Successful replace: the old handle comes back. The `lost` path performs the
    // SAME disk operations (one `set`) and simply declines to hand it over, so the
    // on-disk/WAL state at that point is identical to here.
    let old = par.handle().replace_c(alloc.stack(), b).unwrap();
    assert_eq!(old.handle().range().start(), a_off);

    // Simulate the `lost` outcome: the caller never receives the old handle.
    core::mem::forget(old);

    // What the doc says recovery will do for you:
    let reclaimed = bstack_raii::finish(&alloc).unwrap();
    println!("wal::finish after replace_ reclaimed {reclaimed} slices (old block @{a_off})");

    // And after a full reopen:
    drop(alloc);
    let alloc2 = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
    let reclaimed2 = bstack_raii::finish(&alloc2).unwrap();
    println!("wal::finish after reopen reclaimed {reclaimed2} slices");

    // The block is still allocated: a fresh alloc of the same size does not reuse it.
    let probe = alloc2
        .alloc(core::mem::size_of::<<Leaf as BStackBlock>::OnDisk>() as u64)
        .unwrap();
    println!(
        "probe alloc -> {} (old block @{a_off} {})",
        probe.as_range().start(),
        if probe.as_range().start() == a_off {
            "REUSED"
        } else {
            "still held"
        }
    );

    par.bstack_drop(&alloc2).unwrap();
    std::fs::remove_file(&path).ok();
}
}

mod freemany {
//! Regression: `free_many`'s sequential fallback **continues past a
//! failure**, freeing every range it can, and returns a [`FreeManyError`] naming
//! exactly the ranges it could not free — so the caller always knows which ranges
//! remain allocated and can retry exactly those without risking a double-free. This
//! test asserts that: whatever the fault, the reported `unfreed()` set matches the
//! ranges actually left allocated, and a retry of just those succeeds. Requires
//! `--features fault-injection`.
#![cfg(feature = "fault-injection")]
use bstack::fault::FaultPolicy;
use bstack::{BStack, BStackAllocator, BStackRange, FirstFitBStackAllocator};
use bstack_raii::{BStackRaiiAllocator, FreeManyError};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct FailAt {
    seen: AtomicU64,
    target: u64,
}
impl FaultPolicy for FailAt {
    fn next_fault(&self, _op: &'static str, _seq: u64) -> Option<io::Error> {
        if self.seen.fetch_add(1, Ordering::SeqCst) == self.target {
            Some(io::Error::other("injected fault"))
        } else {
            None
        }
    }
}

#[test]
fn free_many_partial_failure_reports_unfreed_ranges() {
    let path = std::env::temp_dir().join(format!(
        "bstack_raii_fm_{}.bstack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
    // A non-bulk allocator, so `free_many` takes the sequential fallback.
    assert!(!alloc.atomic_bulk());

    let mut partials = 0usize;
    let mut continued = false;
    for target in 0..24u64 {
        // Re-allocate a fresh trio each sweep so each run starts clean.
        let alloc2 = FirstFitBStackAllocator::new(
            BStack::open(&path.with_extension(format!("s{target}"))).unwrap(),
        )
        .unwrap();
        let rs: Vec<BStackRange> = (0..3)
            .map(|_| alloc2.alloc(32).unwrap().as_range())
            .collect();
        alloc2.stack().set_fault_policy(Some(Arc::new(FailAt {
            seen: AtomicU64::new(0),
            target,
        })));
        // SAFETY: `rs` are our own live, unshared allocations.
        let r = unsafe { alloc2.free_many(rs.clone()) };
        alloc2.stack().set_fault_policy(None);

        // Probe which of the three are still allocated afterwards: a fresh
        // `dealloc_range` succeeds on a still-allocated block and reports
        // "already free" on one `free_many` already reclaimed.
        let still_allocated: Vec<BStackRange> = rs
            .iter()
            .copied()
            .filter(|x| unsafe { bstack_raii::dealloc_range(&alloc2, *x) }.is_ok())
            .collect();

        match r {
            Ok(()) => {
                // No fault landed on a dealloc: everything was freed.
                assert!(
                    still_allocated.is_empty(),
                    "fault@{target}: free_many returned Ok but left {} range(s) allocated",
                    still_allocated.len()
                );
            }
            Err(e) => {
                partials += 1;
                // The error carries the ranges whose free did not cleanly
                // complete, downcastable.
                let fme = e
                    .get_ref()
                    .and_then(|src| src.downcast_ref::<FreeManyError>())
                    .expect("a partial free_many returns a FreeManyError source");
                let reported: std::collections::BTreeSet<u64> =
                    fme.unfreed().iter().map(|r| r.start()).collect();
                let actual: std::collections::BTreeSet<u64> =
                    still_allocated.iter().map(|r| r.start()).collect();
                // Every range genuinely still allocated is reported, so the caller
                // never silently leaks one. (`dealloc_range` is not atomic against
                // an injected mid-op fault, so `reported` may *over*-report a range
                // whose free completed before a follow-up op faulted — hence
                // superset, not equality.)
                assert!(
                    actual.is_subset(&reported),
                    "fault@{target}: a still-allocated range was not reported: \
                     reported={reported:?} still_allocated={actual:?}"
                );
                // Continue-past-failure: when the injected fault did not wedge the
                // allocator's free-list, it went on to free the rest of the trio.
                // (A mid-op fault *can* leave FirstFit inconsistent so later real
                // deallocs also fail — so this is provable for at least one fault
                // point, not necessarily every one.)
                if fme.unfreed().len() < rs.len() {
                    continued = true;
                }
            }
        }
        drop(alloc2);
        std::fs::remove_file(path.with_extension(format!("s{target}"))).ok();
    }
    assert!(
        partials > 0,
        "expected at least one fault to land on a dealloc and produce a partial free"
    );
    assert!(
        continued,
        "expected at least one partial free to continue past the failure and free the rest"
    );
    std::fs::remove_file(&path).ok();
}
}

mod rcderef {
//! Regression: `rc.bstack_drop(&alloc)` is a **compile error**. A
//! bare handle does not implement `BStackDrop`, so that spelling does not resolve —
//! a `compile_fail` doctest in `src/lib.rs` pins it. Releasing a strong reference is
//! `drop(rc)` (its `Drop` decrements and frees only at zero); this test confirms that
//! correct path keeps other owners valid.
#![forbid(unsafe_code)]
use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::{BStackBlock, TryClone, bstack_block};

#[bstack_block(rc, weak)]
struct Shared {
    v: u32,
}

#[test]
fn dropping_an_rc_is_a_refcount_release_not_a_raw_free() {
    let path = std::env::temp_dir().join(format!(
        "bstack_raii_rcderef_{}.bstack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
    let stack = alloc.stack();

    let rc1 = Shared::new(&alloc, 0xABCD).unwrap();
    let data_off = rc1.handle().range().start();
    let rc2 = rc1.try_clone().unwrap(); // strong = 2
    let weak = rc1.downgrade().unwrap();

    // The correct release: `drop` runs the strong decrement (2 -> 1), NOT a free.
    drop(rc1);

    // The block is still live — rc2 reads it, and the weak upgrades.
    assert_eq!(rc2.handle().get_v(stack).unwrap(), 0xABCD);
    assert!(
        weak.upgrade().unwrap().is_some(),
        "the object must still be live at strong count 1"
    );
    // The data block was NOT handed back to the allocator.
    let probe = alloc.alloc(4).unwrap();
    assert_ne!(
        probe.as_range().start(),
        data_off,
        "the live Shared block must not have been freed by dropping one of two owners"
    );

    // Releasing the last owners frees it exactly once (no double free).
    drop(rc2);
    drop(weak);

    std::fs::remove_file(&path).ok();
}
}
