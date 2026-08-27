//! FUZZ.md **O5** — the RTTI differential, randomized.
//!
//! The oracle: build the *same* structure twice, in two fresh files with identical
//! allocation sequences, then tear one down through the generated `bstack_drop` and
//! the other through `RttiRegistry::teardown`. Both must reclaim exactly the same
//! set of blocks.
//!
//! Comparing *freed sets* rather than file length makes this insensitive to
//! allocator fragmentation (the reason O1 is not a valid leak oracle on `GhostTree`)
//! and to the persistent WAL block (both paths now allocate one — the RTTI teardown
//! is WAL-backed too): only ranges allocated during the
//! *build* phase are compared.
//!
//! This generalizes the two hand-built teardown/clone differential fixtures
//! (see `tests/regression/schema.rs`) into a shape-randomized oracle.
#![allow(dead_code)]
#[path = "o5/recorder.rs"]
mod recorder;

use bstack::BStackAllocator;
use bstack_raii::rtti::{self, RttiRegistry, Value};
use bstack_raii::{BStackBlock, BStackCast, BStackDrop, BStackOwned, TryCloneIn, bstack_class};
use proptest::prelude::*;
use recorder::Recorder;
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Fixtures — the shape space RTTI has to agree with the generated code on.
// ---------------------------------------------------------------------------

#[bstack_class]
struct Leaf {
    v: u32,
}

#[bstack_class(rc)]
struct RcLeaf {
    v: u32,
}

#[bstack_class(rc, weak)]
struct WkLeaf {
    v: u32,
}

#[bstack_class]
struct Node {
    id: u32,
    #[bstack_owned]
    child: Option<Leaf>,
    #[bstack_owned]
    kids: Vec<Leaf>,
    #[bstack_strong]
    shared: Option<RcLeaf>,
    #[bstack_weak]
    watch: Option<WkLeaf>,
    #[bstack_ref]
    alias: Option<Leaf>,
    bytes: Vec<u8>,
    #[embed]
    inline: Leaf,
    #[bstack_owned]
    nested: Option<Node>,
}

// ---------------------------------------------------------------------------
// The randomized instance shape.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Spec {
    child: bool,
    kids: usize,
    shared: bool,
    watch: bool,
    alias: bool,
    bytes: usize,
    nested: Option<Box<Spec>>,
}

fn spec_strategy(depth: u32) -> BoxedStrategy<Spec> {
    let leaf = (
        any::<bool>(),
        0usize..4,
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        0usize..6,
    )
        .prop_map(|(child, kids, shared, watch, alias, bytes)| Spec {
            child,
            kids,
            shared,
            watch,
            alias,
            bytes,
            nested: None,
        });
    if depth == 0 {
        return leaf.boxed();
    }
    (leaf, prop::option::of(spec_strategy(depth - 1)))
        .prop_map(|(mut s, n)| {
            s.nested = n.map(Box::new);
            s
        })
        .boxed()
}

/// Build one `Node` per `spec`, plus whatever side blocks it references. Returns
/// the root and any handles the caller must keep alive (the `ref` targets, which
/// the structure does not own).
fn build(a: &Recorder, spec: &Spec, keep: &mut Vec<BStackOwned<Leaf>>) -> BStackOwned<Node> {
    let child = spec.child.then(|| Leaf::new(a, 1).unwrap());
    let kids: Vec<BStackOwned<Leaf>> = (0..spec.kids)
        .map(|i| Leaf::new(a, i as u32).unwrap())
        .collect();
    let shared = spec.shared.then(|| RcLeaf::new(a, 2).unwrap());
    let alias = spec.alias.then(|| {
        let t = Leaf::new(a, 4).unwrap();
        let r = unsafe { bstack_raii::BStackRef::<Leaf>::from_range(t.handle().range()) };
        keep.push(t);
        r
    });
    let bytes: Vec<u8> = (0..spec.bytes).map(|i| i as u8).collect();
    let inline = Leaf::new(a, 5).unwrap();
    let nested = spec.nested.as_ref().map(|s| build(a, s, keep));
    let node = Node::new(a, 0, child, kids, shared, alias, &bytes, inline, nested).unwrap();
    if spec.watch {
        // A `#[bstack_weak]` field is not a constructor parameter — it is installed
        // afterwards through the unconditional weak setter. Dropping the strong
        // handle leaves only the control block, which the weak field then owns.
        let w = WkLeaf::new(a, 3).unwrap();
        let d = w.downgrade().unwrap();
        // Release the strong reference by DROPPING the guard. `w.bstack_drop(a)`
        // compiles but derefs to the bare handle and frees without decrementing
        // (a defect this harness found).
        drop(w);
        node.handle().set_watch(a, d).unwrap();
    }
    node
}

fn set(v: Vec<(u64, u64)>) -> BTreeSet<(u64, u64)> {
    v.into_iter().collect()
}

/// Backing file, honouring the same `BSTACK_RAII_FUZZ_DIR` RAM-disk knob the
/// hypercube harness uses (FUZZ.md's "throughput problem" — every committing op
/// pays a real fsync otherwise, and each O5 case builds two whole structures).
fn tmp(tag: &str) -> std::path::PathBuf {
    // A monotonic counter, not a timestamp: proptest runs the tests in this binary
    // in parallel and each case opens several files, so nanosecond stamps collide
    // and two threads end up sharing a backing file. (Observed as a flaky failure
    // that vanished when the binary was run alone.)
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut p = std::env::var_os("BSTACK_RAII_FUZZ_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    p.push(format!(
        "bstack_raii_o5_{tag}_{}_{n}.bstack",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&p);
    p
}

/// Force the persistent WAL block into existence and clear the log, so it is not
/// mistaken for structure.
///
/// The static clone/teardown paths are WAL-backed and lazily allocate this block on
/// first use; `RttiRegistry::clone_value` / `teardown` bypass the WAL entirely.
/// Without this the block — and the offset shift it causes for
/// everything after it — shows up as a spurious differential.
fn wal_block_offset(a: &Recorder) -> Option<u64> {
    let mut b = [0u8; 8];
    a.stack()
        .get_into(bstack_raii::STD_WAL_ANCHOR.as_u64(), &mut b)
        .ok()?;
    let off = u64::from_le_bytes(b);
    (off != 0).then_some(off)
}

fn warmup(a: &Recorder) {
    let l = Leaf::new(a, 0).unwrap();
    l.bstack_drop(a).unwrap(); // WAL-backed: creates the block
    a.reset_log();
}

/// A set of `(offset, len)` block extents.
type ExtentSet = BTreeSet<(u64, u64)>;

/// One differential run. Returns `(built, freed_static, freed_rtti)`.
fn differential(reg: &RttiRegistry, spec: &Spec) -> (ExtentSet, ExtentSet, ExtentSet) {
    let ord = reg.ordinal_of(Node::eightcc()).unwrap();

    // --- run A: the generated teardown -----------------------------------
    let pa = tmp("a");
    let a = Recorder::new(bstack::BStack::open(&pa).unwrap()).unwrap();
    warmup(&a);
    let mut keep_a = Vec::new();
    let root_a = build(&a, spec, &mut keep_a);
    let built_a = set(a.allocated());
    root_a.bstack_drop(&a).unwrap();
    let freed_a = set(a.freed());
    for k in keep_a {
        let _ = k.into_inner();
    }
    drop(a);
    std::fs::remove_file(&pa).ok();

    // --- run B: the interpreted teardown ---------------------------------
    let pb = tmp("b");
    let b = Recorder::new(bstack::BStack::open(&pb).unwrap()).unwrap();
    warmup(&b);
    let mut keep_b = Vec::new();
    let root_b = build(&b, spec, &mut keep_b);
    let built_b = set(b.allocated());
    let root_off = root_b.handle().range().start();
    let _ = root_b.into_inner();
    unsafe { reg.teardown(&b, ord, root_off) }.unwrap();
    let freed_b = set(b.freed());
    for k in keep_b {
        let _ = k.into_inner();
    }
    drop(b);
    std::fs::remove_file(&pb).ok();

    assert_eq!(
        built_a, built_b,
        "the two runs did not build the same structure"
    );
    // Restrict to build-phase blocks: the static path also allocates the persistent
    // WAL block during teardown, which the RTTI path never does.
    (
        built_a.clone(),
        freed_a.intersection(&built_a).copied().collect(),
        freed_b.intersection(&built_a).copied().collect(),
    )
}

#[test]
fn o5_smoke_fixed_shape() {
    let schema = tmp("schema");
    let reg = rtti::sync(&schema).unwrap();
    let spec = Spec {
        child: true,
        kids: 2,
        shared: true,
        watch: true,
        alias: true,
        bytes: 4,
        nested: Some(Box::new(Spec {
            child: true,
            kids: 1,
            shared: false,
            watch: false,
            alias: false,
            bytes: 0,
            nested: None,
        })),
    };
    let (built, fa, fb) = differential(&reg, &spec);
    println!("built {} blocks", built.len());
    println!("static freed {}   rtti freed {}", fa.len(), fb.len());
    let only_static: Vec<_> = fa.difference(&fb).collect();
    let only_rtti: Vec<_> = fb.difference(&fa).collect();
    println!("freed only by static: {only_static:?}");
    println!("freed only by rtti:   {only_rtti:?}");
    println!(
        "left by static: {:?}",
        built.difference(&fa).collect::<Vec<_>>()
    );
    println!(
        "left by rtti:   {:?}",
        built.difference(&fb).collect::<Vec<_>>()
    );
    std::fs::remove_file(&schema).ok();
}

/// Read the whole structure through the **typed** accessors, as a comparable tree.
/// Used to prove a clone survives the original's destruction intact.
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    id: u32,
    child: Option<u32>,
    kids: Vec<u32>,
    shared: Option<u32>,
    bytes: Vec<u8>,
    inline: u32,
    nested: Option<Box<Snapshot>>,
}

fn snapshot(a: &Recorder, h: &Node) -> Snapshot {
    let s = a.stack();
    Snapshot {
        id: h.get_id(s).unwrap(),
        child: h.get_child(s).unwrap().map(|c| c.get_v(s).unwrap()),
        kids: h
            .get_kids(a)
            .unwrap()
            .to_vec()
            .unwrap()
            .iter()
            .map(|k| k.get_v(s).unwrap())
            .collect(),
        shared: h.get_shared(s).unwrap().map(|r| r.get_v(s).unwrap()),
        bytes: h.get_bytes(a).unwrap().to_vec().unwrap(),
        inline: h.get_inline().unwrap().get_v(s).unwrap(),
        nested: h.get_nested(s).unwrap().map(|n| Box::new(snapshot(a, &n))),
    }
}

/// One clone-differential run: build, clone, drop the original, scribble over the
/// reclaimed space, then read the clone back.
///
/// Returns `(blocks the clone allocated, the clone's contents afterwards)`.
fn clone_run(reg: &RttiRegistry, spec: &Spec, via_rtti: bool) -> (BTreeSet<(u64, u64)>, Snapshot) {
    let ord = reg.ordinal_of(Node::eightcc()).unwrap();
    let path = tmp(if via_rtti { "cr" } else { "cs" });
    let a = Recorder::new(bstack::BStack::open(&path).unwrap()).unwrap();
    warmup(&a);
    let mut keep = Vec::new();
    let root = build(&a, spec, &mut keep);
    let built = set(a.allocated());
    let mark = a.mark();
    let fmark = a.free_mark();

    let clone_off = if via_rtti {
        unsafe { reg.clone_value(&a, ord, root.handle().range().start()) }.unwrap()
    } else {
        let c = root.try_clone_in(&a).unwrap();
        let off = c.handle().range().start();
        let _ = c.into_inner();
        off
    };
    // Everything allocated after the build phase belongs to the clone — except the
    // persistent WAL block, which `ClonePlan` may have *grown* mid-transaction (free
    // old, allocate larger). Identify it by the anchor slot rather than by size.
    // The clone's *net* new blocks: allocated in this window, still live at the end
    // of it, and not the persistent WAL block. `ClonePlan` may grow the WAL
    // mid-transaction (free the old block, allocate a larger one), so both halves of
    // that churn have to drop out — the freed old one via the `freed_since` subtraction,
    // the surviving new one via the anchor.
    let wal = wal_block_offset(&a);
    let freed_in_window: BTreeSet<(u64, u64)> = a.freed_since(fmark).into_iter().collect();
    let clone_alloc: BTreeSet<(u64, u64)> = a
        .allocated_since(mark)
        .into_iter()
        .filter(|r| !freed_in_window.contains(r) && Some(r.0) != wal)
        .collect();

    root.bstack_drop(&a).unwrap();
    // Scribble over the reclaimed space so an aliasing clone cannot read
    // stale-but-intact bytes and look healthy.
    let mut scribble = Vec::new();
    for (_, len) in built.iter() {
        if let Ok(mut sl) = a.alloc(*len) {
            let _ = sl.write(vec![0xFEu8; *len as usize]);
            scribble.push(sl.as_range());
        }
    }

    let handle = unsafe {
        <Node as BStackBlock>::from_range(bstack::BStackRange::new(
            clone_off,
            core::mem::size_of::<<Node as BStackBlock>::OnDisk>() as u64,
        ))
    };
    let snap = snapshot(&a, &handle);

    for r in scribble {
        let _ = unsafe { bstack_raii::dealloc_range(&a, r) };
    }
    for k in keep {
        let _ = k.into_inner();
    }
    drop(a);
    std::fs::remove_file(&path).ok();
    (clone_alloc, snap)
}

/// Rebuild a [`Snapshot`] from the **interpreter's** `Value` tree, so it can be
/// compared field-for-field against the typed reader above.
///
/// This is the read half of O5. Where the teardown/clone halves compare *which
/// blocks* the two implementations touch, this compares *what they think the data
/// is* — the oracle that catches a field recorded with the wrong shape, which reads
/// past its extent rather than freeing the wrong thing.
fn snapshot_from_value(v: &Value) -> Snapshot {
    let Value::Block { fields, .. } = v else {
        panic!("expected a Block, got {v:?}")
    };
    let f = |name: &str| -> &Value {
        &fields
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("missing field {name} in {fields:?}"))
            .1
    };
    let pod_u32 = |v: &Value| -> u32 {
        let Value::Pod(b) = v else {
            panic!("expected Pod, got {v:?}")
        };
        u32::from_le_bytes(b[..4].try_into().unwrap())
    };
    // A block whose only field is `v: u32` (Leaf / RcLeaf).
    let leaf_v = |v: &Value| -> u32 {
        let Value::Block { fields, .. } = v else {
            panic!("expected Block, got {v:?}")
        };
        pod_u32(&fields[0].1)
    };
    fn opt(v: &Value) -> Option<&Value> {
        match v {
            Value::Null => None,
            Value::Some(inner) => Some(inner),
            other => Some(other),
        }
    }

    Snapshot {
        id: pod_u32(f("id")),
        child: opt(f("child")).map(leaf_v),
        kids: match f("kids") {
            Value::Vec(items) => items.iter().map(leaf_v).collect(),
            Value::Null => Vec::new(),
            other => panic!("expected Vec for kids, got {other:?}"),
        },
        // `strong` is followed like an owned child, so it reads back as a Block.
        shared: opt(f("shared")).map(leaf_v),
        bytes: match f("bytes") {
            Value::Vec(items) => items
                .iter()
                .map(|e| {
                    let Value::Pod(b) = e else {
                        panic!("expected Pod byte, got {e:?}")
                    };
                    b[0]
                })
                .collect(),
            Value::Null => Vec::new(),
            other => panic!("expected Vec for bytes, got {other:?}"),
        },
        inline: leaf_v(f("inline")),
        nested: opt(f("nested")).map(|n| Box::new(snapshot_from_value(n))),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

    /// **O5.** For every generated shape, the generated `bstack_drop` and
    /// `RttiRegistry::teardown` must reclaim exactly the same set of blocks — and
    /// between them leave nothing from the build phase behind.
    #[test]
    fn o5_teardowns_agree(spec in spec_strategy(2)) {
        let schema = tmp("schema");
        let reg = rtti::sync(&schema).unwrap();
        let (built, fa, fb) = differential(&reg, &spec);
        std::fs::remove_file(&schema).ok();

        let only_static: Vec<_> = fa.difference(&fb).copied().collect();
        let only_rtti: Vec<_> = fb.difference(&fa).copied().collect();
        prop_assert!(
            only_static.is_empty() && only_rtti.is_empty(),
            "teardowns disagree for {spec:?}\n  freed only by static: {only_static:?}\n  freed only by rtti:   {only_rtti:?}"
        );

        // Both agreeing is necessary but not sufficient — they can agree on being
        // wrong. `alias` targets are deliberately not owned by the structure; every
        // other built block must be reclaimed.
        let expected_survivors = spec_alias_count(&spec);
        let left: Vec<_> = built.difference(&fa).copied().collect();
        prop_assert_eq!(
            left.len(), expected_survivors,
            "blocks left allocated by both teardowns for {:?}: {:?}", spec, left
        );
    }

    /// **O5, read half.** `RttiRegistry::read_value` and the generated typed
    /// accessors must agree on the contents of the same live structure.
    #[test]
    fn o5_reads_agree(spec in spec_strategy(2)) {
        let schema = tmp("schema");
        let reg = rtti::sync(&schema).unwrap();
        let ord = reg.ordinal_of(Node::eightcc()).unwrap();
        let path = tmp("rd");
        let a = Recorder::new(bstack::BStack::open(&path).unwrap()).unwrap();
        warmup(&a);
        let mut keep = Vec::new();
        let root = build(&a, &spec, &mut keep);

        let typed = snapshot(&a, root.handle());
        let value = reg.read_value(a.stack(), ord, root.handle().range().start()).unwrap();
        let interpreted = snapshot_from_value(&value);

        root.bstack_drop(&a).unwrap();
        for k in keep { let _ = k.into_inner(); }
        drop(a);
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&schema).ok();

        prop_assert_eq!(&typed, &interpreted, "typed and interpreted reads disagree for {:?}", spec);
    }

    /// **O5, clone half.** `TryCloneIn` and `RttiRegistry::clone_value` must
    /// allocate the same fresh blocks and produce the same contents — and the copy
    /// must survive the original's destruction, which it cannot if it aliased any of
    /// the original's owned blocks.
    #[test]
    fn o5_clones_agree(spec in spec_strategy(2)) {
        let schema = tmp("schema");
        let reg = rtti::sync(&schema).unwrap();
        let (alloc_static, snap_static) = clone_run(&reg, &spec, false);
        let (alloc_rtti, snap_rtti) = clone_run(&reg, &spec, true);
        std::fs::remove_file(&schema).ok();

        // Compare the *multiset of block sizes*, not offsets: the two
        // implementations legitimately allocate the copy's blocks in different
        // orders (the plan stages them, the interpreter clones bottom-up), so exact
        // offsets differ while the set of blocks does not. An aliasing clone still
        // trips this — it allocates one block fewer.
        let sizes = |m: &BTreeSet<(u64, u64)>| {
            let mut v: Vec<u64> = m.iter().map(|(_, l)| *l).collect();
            v.sort_unstable();
            v
        };
        prop_assert_eq!(
            sizes(&alloc_static), sizes(&alloc_rtti),
            "the two clone paths allocated different blocks for {:?}\n  static: {:?}\n  rtti:   {:?}",
            spec, &alloc_static, &alloc_rtti
        );
        prop_assert_eq!(
            &snap_static, &snap_rtti,
            "the two clones differ in contents for {:?}", spec
        );
    }
}

/// How many `#[bstack_ref]` targets the spec creates — blocks the structure points
/// at but does not own, so teardown must leave them alone.
fn spec_alias_count(s: &Spec) -> usize {
    (s.alias as usize) + s.nested.as_ref().map_or(0, |n| spec_alias_count(n))
}

/// Regression fixture. An un-annotated `Option<Vec<u8>>` field is recorded in the
/// schema as the container region matching its 16-byte on-disk `VecDesc`, so the
/// interpreter reads (and tears down) this field exactly as the typed accessor /
/// static path do — following the descriptor to the vec's data block, not reading a
/// mis-sized 24-byte inline POD blob that overruns into `marker`. The two tests below
/// pin that agreement.
#[bstack_class]
struct Bad {
    v: Option<Vec<u8>>,
    marker: u64,
}

#[test]
fn o5_read_agrees_on_optvec_field() {
    let schema = tmp("badschema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(Bad::eightcc()).unwrap();
    let path = tmp("bad");
    let a = Recorder::new(bstack::BStack::open(&path).unwrap()).unwrap();

    let h = Bad::new(&a, Some(b"hello".as_slice()), 0xAB).unwrap();
    let typed: Vec<u8> = h.handle().get_v(&a).unwrap().unwrap().to_vec().unwrap();
    let value = reg
        .read_value(a.stack(), ord, h.handle().range().start())
        .unwrap();
    let Value::Block { fields, .. } = &value else {
        panic!()
    };
    let interpreted = &fields.iter().find(|(n, _)| n == "v").unwrap().1;

    println!("typed accessor -> Vec {typed:?}");
    println!("interpreter    -> {interpreted:?}");

    // The interpreter records the container region, so it reads the field as the
    // SAME vector the typed accessor sees — a `Some(Vec)`, not a mis-sized opaque
    // POD blob that overruns into `marker`.
    assert!(
        !matches!(interpreted, Value::Pod(_)),
        "`Option<Vec>` read back as opaque POD {interpreted:?}"
    );
    let Value::Some(inner) = interpreted else {
        panic!("expected Some(Vec) for Some(b\"hello\"), got {interpreted:?}");
    };
    let Value::Vec(elems) = &**inner else {
        panic!("expected a Vec region, got {inner:?}");
    };
    let bytes: Vec<u8> = elems
        .iter()
        .map(|e| match e {
            Value::Pod(b) => b[0],
            other => panic!("vec element not a POD byte: {other:?}"),
        })
        .collect();
    assert_eq!(
        bytes, typed,
        "interpreter and typed accessor disagree on the vec contents"
    );

    h.bstack_drop(&a).unwrap();
    drop(a);
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&schema).ok();
}

/// Regression. Same `Bad` shape: the interpreter knows the field owns a vector data
/// block, so `RttiRegistry::teardown` frees exactly what the static walk does —
/// including the vec data block, not just 24 bytes of inline POD.
#[test]
fn o5_teardown_agrees_on_optvec_field() {
    let schema = tmp("bad2schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(Bad::eightcc()).unwrap();

    let mut freed = Vec::new();
    for via_rtti in [false, true] {
        let path = tmp(if via_rtti { "bt2r" } else { "bt2s" });
        let a = Recorder::new(bstack::BStack::open(&path).unwrap()).unwrap();
        warmup(&a);
        let h = Bad::new(&a, Some(b"hello".as_slice()), 0xAB).unwrap();
        let built = set(a.allocated());
        let off = h.handle().range().start();
        if via_rtti {
            let _ = h.into_inner();
            unsafe { reg.teardown(&a, ord, off) }.unwrap();
        } else {
            h.bstack_drop(&a).unwrap();
        }
        let f: BTreeSet<(u64, u64)> = set(a.freed()).intersection(&built).copied().collect();
        println!(
            "  {:<12} built {:?}  freed {:?}",
            if via_rtti { "clone_value" } else { "static" },
            built,
            f
        );
        freed.push((built, f));
        drop(a);
        std::fs::remove_file(&path).ok();
    }
    std::fs::remove_file(&schema).ok();

    let (_built, fs_static) = &freed[0];
    let (_, fs_rtti) = &freed[1];
    let missed: Vec<_> = fs_static.difference(fs_rtti).collect();
    let extra: Vec<_> = fs_rtti.difference(fs_static).collect();
    println!("blocks the static teardown freed and the interpreter did not: {missed:?}");
    // The two teardowns free exactly the same set — the interpreter frees the
    // `Option<Vec>` data block and nothing the static walk doesn't.
    assert!(
        missed.is_empty(),
        "interpreter missed the vec data block(s) {missed:?}"
    );
    assert!(
        extra.is_empty(),
        "interpreter freed block(s) the static walk did not: {extra:?}"
    );
}
