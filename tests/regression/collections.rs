//! Collection iterator / container findings.
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

mod iterinvalidate {
//! Regression: a stdlib collection iterator caches its backing
//! block location at construction. Each `next()` re-reads a cheap invariant (the
//! backing pointer, or `len`/`root`) and returns `Err(InvalidData)` when it changed,
//! so mutating the collection mid-iteration is detected and yields a clean, fail-fast
//! error instead of reading freed storage — which, since each item is an
//! owning-capable `T` handle, would otherwise be a way to free arbitrary offsets.
//! These tests mutate an in-flight iterator and assert exactly that clean error.
#![allow(dead_code)]
use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::{BStackBTreeMap, BStackDeque, BStackDrop, BStackLinkedList, bstack_block};

#[bstack_block]
struct Leaf {
    v: u32,
}

type A = FirstFitBStackAllocator;

fn tmp(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "bstack_raii_iterinv_{tag}_{}.bstack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn deque_iterator_detects_ring_growth() {
    let path = tmp("deque");
    let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

    let d = BStackDeque::<Leaf>::new(&alloc).unwrap();
    for i in 0..4 {
        let l = Leaf::new(&alloc, i).unwrap();
        d.handle().push_back(&alloc, l).unwrap();
    }

    let mut it = d.handle().iter(alloc.stack()).unwrap();
    it.next().expect("first element").expect("no error yet");

    // Force a ring growth: the old ring (the iterator's snapshot) is freed.
    for i in 0..64 {
        let l = Leaf::new(&alloc, 100 + i).unwrap();
        d.handle().push_back(&alloc, l).unwrap();
    }

    let r = it.next().expect("iterator still yields (an error, not None)");
    assert!(
        r.is_err(),
        "deque iterator read the freed ring instead of failing fast"
    );

    let _ = d.into_inner();
    std::fs::remove_file(&path).ok();
}

#[test]
fn deque_iterator_detects_pop() {
    // A `pop` mutates the ring *in place* — it advances `head` / decrements `len`
    // and hands out (frees) the front element block, but does NOT move the ring or
    // clear the vacated slot. A `data`/`cap`-only check misses it and reads a stale
    // offset to a freed block (UAF); the iterator must compare `head`/`len` too and
    // fail fast.
    let path = tmp("deque_pop");
    let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

    let d = BStackDeque::<Leaf>::new(&alloc).unwrap();
    for i in 0..4 {
        let l = Leaf::new(&alloc, i).unwrap();
        d.handle().push_back(&alloc, l).unwrap();
    }

    let mut it = d.handle().iter(alloc.stack()).unwrap();
    it.next().expect("first element").expect("no error yet");

    // Pop the front: frees that element block and advances head/len, same ring.
    let popped = d.handle().pop_front(&alloc).unwrap().expect("an element");
    popped.bstack_drop(&alloc).unwrap();

    let r = it.next().expect("iterator still yields (an error, not None)");
    assert!(
        r.is_err(),
        "deque iterator read a stale/freed element slot after an in-place pop \
         instead of failing fast"
    );

    let _ = d.into_inner();
    std::fs::remove_file(&path).ok();
}

#[test]
fn list_iterator_detects_pop() {
    let path = tmp("list");
    let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

    let l = BStackLinkedList::<Leaf>::new(&alloc).unwrap();
    for i in 0..4 {
        let x = Leaf::new(&alloc, i).unwrap();
        l.handle().push_back(&alloc, x).unwrap();
    }

    let mut it = l.handle().iter(alloc.stack()).unwrap();
    it.next().expect("first element").expect("no error yet");

    // Pop a node — its block is freed; the iterator's cached `cur` may name it.
    l.handle().pop_back(&alloc).unwrap();

    let r = it.next().expect("iterator still yields (an error, not None)");
    assert!(
        r.is_err(),
        "list iterator did not detect the pop-during-iteration"
    );

    let _ = l.into_inner();
    std::fs::remove_file(&path).ok();
}

#[test]
fn btreemap_iterator_detects_insert() {
    let path = tmp("btree");
    let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

    let m = BStackBTreeMap::<u32, Leaf>::new(&alloc).unwrap();
    for i in 0..8u32 {
        let v = Leaf::new(&alloc, i).unwrap();
        m.handle().insert(&alloc, i, v).unwrap();
    }

    let mut it = m.handle().iter(alloc.stack()).unwrap();
    it.next().expect("first entry").expect("no error yet");

    // A path-copying insert frees the old root-to-leaf path and swaps the root — the
    // iterator's cached frames now name freed nodes.
    for i in 100..140u32 {
        let v = Leaf::new(&alloc, i).unwrap();
        m.handle().insert(&alloc, i, v).unwrap();
    }

    let r = it.next().expect("iterator still yields (an error, not None)");
    assert!(
        r.is_err(),
        "b-tree iterator decoded freed nodes instead of failing fast"
    );

    let _ = m.into_inner();
    std::fs::remove_file(&path).ok();
}
}

mod collections {
//! Harness: a drop/clone differential across every stdlib collection that
//! holds **owning** element blocks.
//!
//! For each: build it from three known child blocks, deep-clone it, drop the
//! ORIGINAL, then check both halves of the contract at once —
//!   (a) every one of the original's children is freed (no leak), and
//!   (b) the clone still reads back its three values (so the clone never aliased
//!       the original's children, which the drop just reclaimed).
#![allow(dead_code)]
use bstack::{BStack, BStackAllocator, BStackRange, FirstFitBStackAllocator};
use bstack_raii::{
    BStackBTreeMap, BStackBinaryHeap, BStackBlock, BStackDeque, BStackDrop, BStackHashMap,
    BStackLinkedList, BStackOwned, TryCloneIn, bstack_block,
};

#[bstack_block]
struct Leaf {
    v: u32,
}

type A = FirstFitBStackAllocator;

fn leaf_sz() -> u64 {
    core::mem::size_of::<<Leaf as BStackBlock>::OnDisk>() as u64
}

/// A range that still frees cleanly was NOT freed by the teardown.
fn still_allocated(a: &A, off: u64) -> bool {
    unsafe { bstack_raii::dealloc_range(a, BStackRange::new(off, leaf_sz())) }.is_ok()
}

fn tmp(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "bstack_raii_coll_{tag}_{}.bstack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn check<C, B, V>(name: &str, build: B, read_values: V)
where
    C: TryCloneIn + BStackBlock,
    B: Fn(&A, &mut Vec<u64>) -> BStackOwned<C>,
    V: Fn(&A, &C) -> Vec<u32>,
{
    let path = tmp(name);
    let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

    let mut child_offs = Vec::new();
    let orig = build(&alloc, &mut child_offs);
    let clone = orig.try_clone_in(&alloc).unwrap();

    let before = read_values(&alloc, clone.handle());
    orig.bstack_drop(&alloc).unwrap();
    // Probe for leaks FIRST — the scribble below would re-occupy the freed slots.
    let leaked: Vec<u64> = child_offs
        .iter()
        .copied()
        .filter(|&o| still_allocated(&alloc, o))
        .collect();
    // Scribble over whatever the drop reclaimed, so an aliasing clone cannot read
    // stale-but-intact bytes and appear healthy.
    let mut scribble = Vec::new();
    for _ in 0..8 {
        let mut sl = alloc.alloc(leaf_sz()).unwrap();
        sl.write(vec![0xFFu8; leaf_sz() as usize]).unwrap();
        scribble.push(sl.as_range());
    }
    let after = read_values(&alloc, clone.handle());
    for r in scribble {
        let _ = unsafe { bstack_raii::dealloc_range(&alloc, r) };
    }

    let ok_leak = leaked.is_empty();
    let ok_alias = !before.is_empty() && after == before;

    println!(
        "  {:<11} clone before drop {before:?} -> after {after:?}   original's children leaked: {leaked:?}   {}",
        name,
        if ok_leak && ok_alias { "ok" } else { "BAD" }
    );
    assert!(
        ok_leak,
        "{name}: dropping the collection left its own children allocated"
    );
    assert!(
        ok_alias,
        "{name}: the clone's values changed when the original was dropped (aliasing)"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn owning_collections_drop_and_clone_their_children() {
    println!();
    check::<BStackDeque<Leaf>, _, _>(
        "Deque",
        |a, offs| {
            let d = BStackDeque::<Leaf>::new(a).unwrap();
            for i in 0..3 {
                let l = Leaf::new(a, i).unwrap();
                offs.push(l.handle().range().start());
                d.handle().push_back(a, l).unwrap();
            }
            d
        },
        |a, h| {
            h.to_vec(a.stack())
                .unwrap()
                .iter()
                .map(|x| x.get_v(a.stack()).unwrap())
                .collect()
        },
    );

    check::<BStackLinkedList<Leaf>, _, _>(
        "LinkedList",
        |a, offs| {
            let l = BStackLinkedList::<Leaf>::new(a).unwrap();
            for i in 0..3 {
                let x = Leaf::new(a, i).unwrap();
                offs.push(x.handle().range().start());
                l.handle().push_back(a, x).unwrap();
            }
            l
        },
        |a, h| {
            h.to_vec(a.stack())
                .unwrap()
                .iter()
                .map(|x| x.get_v(a.stack()).unwrap())
                .collect()
        },
    );

    check::<BStackBTreeMap<u32, Leaf>, _, _>(
        "BTreeMap",
        |a, offs| {
            let t = BStackBTreeMap::<u32, Leaf>::new(a).unwrap();
            for i in 0..3u32 {
                let l = Leaf::new(a, i).unwrap();
                offs.push(l.handle().range().start());
                t.handle().insert(a, i, l).unwrap();
            }
            t
        },
        |a, h| {
            (0..3u32)
                .map(|i| {
                    h.get(a.stack(), &i)
                        .unwrap()
                        .unwrap()
                        .get_v(a.stack())
                        .unwrap()
                })
                .collect()
        },
    );

    check::<BStackHashMap<u32, Leaf>, _, _>(
        "HashMap",
        |a, offs| {
            let m = BStackHashMap::<u32, Leaf>::new(a).unwrap();
            for i in 0..3u32 {
                let l = Leaf::new(a, i).unwrap();
                offs.push(l.handle().range().start());
                m.handle().insert(a, i, l).unwrap();
            }
            m
        },
        |a, h| {
            (0..3u32)
                .map(|i| {
                    h.get(a.stack(), &i)
                        .unwrap()
                        .unwrap()
                        .get_v(a.stack())
                        .unwrap()
                })
                .collect()
        },
    );

    check::<BStackBinaryHeap<u32, Leaf>, _, _>(
        "BinaryHeap",
        |a, offs| {
            let p = BStackBinaryHeap::<u32, Leaf>::new(a).unwrap();
            for i in 0..3u32 {
                let l = Leaf::new(a, i).unwrap();
                offs.push(l.handle().range().start());
                p.handle().push(a, i, l).unwrap();
            }
            p
        },
        // `peek` is the only non-destructive read the heap offers.
        |a, h| {
            let (k, v) = h.peek(a.stack()).unwrap().expect("heap is non-empty");
            vec![
                k,
                v.get_v(a.stack()).unwrap(),
                h.len(a.stack()).unwrap() as u32,
            ]
        },
    );
}
}
