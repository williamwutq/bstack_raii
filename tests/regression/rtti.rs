//! RTTI registry findings (schema resync).
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

mod weakforge {
    //! Regression: `RttiRegistry::swap` must not accept a forged
    //! "control block" into a `#[bstack_weak]` slot. A `(rc, weak)` block's control
    //! block carries the type's **control** tag in its header; the schema now persists
    //! that tag (`RttiType.ctrl_tag`), so `swap` validates a weak target's control block
    //! *directly* by its header — not only indirectly via its forward data pointer.
    //!
    //! The exploit below needs no `unsafe` on the data side and no corrupt file: an
    //! ordinary `Vec<u8>` whose bytes are laid out so its data block *looks* like a
    //! control block (`weak == 1` at +24, a real target's data offset at +32). Before the
    //! fix a later teardown drove that "weak" count to zero and freed the vector's own
    //! live data block, leaving the `Vec` a use-after-free. The one `unsafe` is the
    //! fabricated `AnyRef`, the demonstration itself.
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti::{self, AnyRef, RttiBody};
    use bstack_raii::{BStackBlock, BStackCast, bstack_class};

    // Control-block field offsets (a control block is `{ header:16, strong@16, weak@24,
    // x@32 }`); `layout` is crate-private, so mirror the two the forgery targets.
    const CTRL_WEAK_OFFSET: usize = 24;
    const CTRL_DATA_OFFSET: usize = 32;

    #[bstack_class(rc, weak)]
    struct WCell {
        v: u32,
    }

    #[bstack_class]
    struct Bait {
        v: Vec<u8>,
    }

    #[bstack_class]
    struct WHolder {
        #[bstack_mut]
        #[bstack_weak]
        w: Option<WCell>,
    }

    #[test]
    fn forged_control_block_rejected_from_a_weak_slot() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_wf_s_{stamp}.bstack"));
        let data = std::env::temp_dir().join(format!("bstack_raii_wf_d_{stamp}.bstack"));
        let reg = rtti::sync(&schema).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        let stack = alloc.stack();

        // A real (rc, weak) target and a holder with a live weak reference to it.
        let cell = WCell::new(&alloc, 5).unwrap();
        let cell_off = cell.handle().range().start();
        let holder = WHolder::new(&alloc).unwrap();
        holder
            .handle()
            .set_w(&alloc, cell.downgrade().unwrap())
            .unwrap();
        let hord = reg.ordinal_of(WHolder::eightcc()).unwrap();
        let hoff = holder.handle().range().start();

        // Craft a byte vector whose *data block* mimics a control block: the vec data
        // block is `[len@0 | cap@8 | elements@16..]`, so element bytes 8..16 land at the
        // block's +24 (`CTRL_WEAK_OFFSET`) and 16..24 at +32 (`CTRL_DATA_OFFSET`).
        let _ = (CTRL_WEAK_OFFSET, CTRL_DATA_OFFSET); // documented above
        let mut content = [0u8; 32];
        content[8..16].copy_from_slice(&1u64.to_le_bytes()); // -> block+24 : weak == 1
        content[16..24].copy_from_slice(&cell_off.to_le_bytes()); // -> block+32 : data -> WCell
        let bait = Bait::new(&alloc, &content[..]).unwrap();

        // The vector's data block offset `D` — the forged "control block" — comes from the
        // inline `VecDesc` (its first `u64`) at Bait's `v` field.
        let bord = reg.ordinal_of(Bait::eightcc()).unwrap();
        let bty = reg.load_type(bord).unwrap();
        let RttiBody::Struct(fs) = &bty.body else {
            panic!("Bait is a struct")
        };
        let v_rel = fs.iter().find(|f| f.name == "v").unwrap().offset as u64;
        let bait_off = bait.handle().range().start();
        let mut d = [0u8; 8];
        stack.get_into(bait_off + v_rel, &mut d).unwrap();
        let forged_ctrl = u64::from_le_bytes(d); // == D, the vec data block

        // Installing the forgery into the weak slot must be rejected: `D`'s header tag
        // (really the vector's `cap` word) is not WCell's control tag.
        let r = reg.swap(stack, hord, hoff, &["w"], unsafe {
            AnyRef::new(WCell::eightcc(), forged_ctrl)
        });
        assert!(
            r.is_err(),
            "a forged control block (an ordinary Vec<u8> data block) was accepted into a weak slot"
        );

        // Sanity: the real control block round-trips (the check accepts a genuine one).
        // Null the weak field back out — a nullable weak field accepts the `0` niche —
        // then the returned real control `AnyRef` re-installs cleanly.
        let real_ctrl = reg
            .swap(stack, hord, hoff, &["w"], unsafe {
                AnyRef::new(WCell::eightcc(), 0)
            })
            .unwrap()
            .expect("weak field was set");
        reg.swap(stack, hord, hoff, &["w"], real_ctrl)
            .expect("a genuine control block must still be accepted");

        // The Vec's data block was never freed by the rejected swap: it still reads back.
        let got = bait.handle().get_v(&alloc).unwrap();
        assert_eq!(got.len().unwrap(), 32);

        let _ = holder.into_inner();
        let _ = bait.into_inner();
        drop(cell);
        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}

mod cycle {
    //! Probe: an unbounded-recursion cycle can be built through `Foreign` +
    //! `swap_foreign`. This asks whether the same cycle can be formed **in-file**, with
    //! no `Foreign`, no registry, and no cross-file hop.
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti::{self, AnyRef};
    use bstack_raii::{BStackBlock, BStackCast, BStackDrop, BStackOwned, bstack_class};

    #[bstack_class]
    struct Node {
        id: u32,
        #[bstack_mut]
        #[bstack_owned]
        next: Option<Node>,
    }

    fn setup() -> (
        rtti::RttiRegistry,
        FirstFitBStackAllocator,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_cyc_s_{stamp}.bstack"));
        let data = std::env::temp_dir().join(format!("bstack_raii_cyc_d_{stamp}.bstack"));
        let reg = rtti::sync(&schema).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        (reg, alloc, schema, data)
    }

    /// Can safe code form an in-file `#[bstack_owned]` cycle at all?
    #[test]
    fn in_file_owned_cycle_is_formable() {
        let (reg, alloc, schema, data) = setup();
        let ord = reg.ordinal_of(Node::eightcc()).unwrap();

        let a: BStackOwned<Node> = Node::new(&alloc, 1, None).unwrap();
        let b: BStackOwned<Node> = Node::new(&alloc, 2, None).unwrap();
        let a_off = a.handle().range().start();
        let b_off = b.handle().range().start();

        // Both swaps are safe, and both pass `verify_data_block`: a and b really are
        // live `Node` blocks. No ownership is transferred, so both handles stay ours.
        let old_a = reg
            .swap(alloc.stack(), ord, a_off, &["next"], unsafe {
                AnyRef::new(Node::eightcc(), b_off)
            })
            .unwrap();
        let old_b = reg
            .swap(alloc.stack(), ord, b_off, &["next"], unsafe {
                AnyRef::new(Node::eightcc(), a_off)
            })
            .unwrap();
        println!("a@{a_off}.next := b@{b_off}  (old {old_a:?})");
        println!("b@{b_off}.next := a@{a_off}  (old {old_b:?})");
        println!(
            "cycle: a.next = {:?}, b.next = {:?}",
            a.handle()
                .get_next(alloc.stack())
                .unwrap()
                .map(|h| h.range().start()),
            b.handle()
                .get_next(alloc.stack())
                .unwrap()
                .map(|h| h.range().start()),
        );

        // The interpreter bounds it.
        let r = unsafe { reg.teardown(&alloc, ord, a_off) };
        println!(
            "\nRTTI teardown  -> {:?}",
            r.as_ref().err().map(|e| e.to_string())
        );
        assert!(r.is_err(), "the interpreter must reject a cycle");

        let _ = a.into_inner();
        let _ = b.into_inner();
        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }

    /// The generated teardown bounds recursion on the same cycle: a depth guard makes it
    /// return a clean `Err` rather than overflowing the native stack.
    #[test]
    fn in_file_owned_cycle_errors_in_the_static_teardown() {
        let (reg, alloc, schema, data) = setup();
        let ord = reg.ordinal_of(Node::eightcc()).unwrap();
        let a: BStackOwned<Node> = Node::new(&alloc, 1, None).unwrap();
        let b: BStackOwned<Node> = Node::new(&alloc, 2, None).unwrap();
        let a_off = a.handle().range().start();
        let b_off = b.handle().range().start();
        reg.swap(alloc.stack(), ord, a_off, &["next"], unsafe {
            AnyRef::new(Node::eightcc(), b_off)
        })
        .unwrap();
        reg.swap(alloc.stack(), ord, b_off, &["next"], unsafe {
            AnyRef::new(Node::eightcc(), a_off)
        })
        .unwrap();
        let _ = b.into_inner();
        // Must return an error, not overflow the native stack.
        let r = a.bstack_drop(&alloc);
        assert!(r.is_err(), "the static teardown must bound an owned cycle");

        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}

mod allocblock {
    //! Regression: `bstack_raii::alloc_block` is crate-private and
    //! test-only. A public, safe `pub fn` stamping a caller-chosen tag over a
    //! caller-chosen size would be a factory for the exact credential every
    //! header-trusting identity gate (`bstack_cast!`, `AnyRef::from_block`,
    //! `verify_data_block`) validates; no code path outside generated constructors needs
    //! it, and `AnyRef::new` is an `unsafe fn`, so safe code cannot mint a mislabelled
    //! block or a fabricated runtime-typed reference.
    //!
    //! Residual runtime check kept here: even a deliberately (unsafely) fabricated
    //! `AnyRef` over an offset holding no block of the claimed type is rejected by the
    //! validated mutator (`[BSTACK0815]`), so the RTTI surface does not trust the pair.

    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti::{self, AnyRef};
    use bstack_raii::{BStackBlock, BStackCast, bstack_class};

    #[bstack_class]
    struct Big {
        a: u64,
        b: u64,
        c: u64,
        d: u64,
    }

    #[bstack_class]
    struct Holder {
        #[bstack_mut]
        #[bstack_owned]
        o: Option<Big>,
    }

    #[test]
    fn fabricated_anyref_is_rejected_by_swap() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_ab_s_{stamp}.bstack"));
        let data = std::env::temp_dir().join(format!("bstack_raii_ab_d_{stamp}.bstack"));
        let reg = rtti::sync(&schema).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        let stack = alloc.stack();

        let h = Holder::new(&alloc, None).unwrap();
        let hord = reg.ordinal_of(Holder::eightcc()).unwrap();
        let h_off = h.handle().range().start();

        // A raw allocation with no block header: safe code cannot stamp a `Big` tag
        // here (`alloc_block` is not exported), so the bytes carry whatever the
        // allocator left.
        let bare = alloc.alloc(24).unwrap().as_range();

        // Fabricating the credential now requires `unsafe` — and even then the
        // validated mutator re-reads the on-disk header and refuses.
        let forged = unsafe { AnyRef::new(Big::eightcc(), bare.start()) };
        let r = reg.swap(stack, hord, h_off, &["o"], forged);
        let msg = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("BSTACK0815"),
            "swap accepted a fabricated AnyRef over a non-block: {msg:?}"
        );

        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}

mod resync {
    //! Harness: the RTTI schema's persistence round-trip.
    //!
    //! `sync_compiled` compares the *decoded* on-disk descriptor against the *compiled*
    //! one and rejects a mismatch with [BSTACK0814]. So syncing a second time over the
    //! same file is an end-to-end encode -> decode -> compare identity test across the
    //! whole `Shape` grammar. Also checks that a mutated `#[bstack_mut] #[bstack_static]`
    //! class variable — whose persisted value legitimately diverges from the compiled
    //! initial value — survives a re-sync rather than reading as schema drift.
    #![allow(dead_code)]
    #[allow(unused_imports)]
    use bstack_raii::Foreign; // named in field types; the macro re-resolves it
    use bstack_raii::rtti::{self, RttiBody};
    use bstack_raii::{BStackCast, bstack_class};

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
    struct Every {
        p_scalar: u32,
        p_arr: [u64; 3],
        p_tuple: (u32, u16),
        #[bstack_owned]
        owned: Leaf,
        #[bstack_owned]
        owned_opt: Option<Leaf>,
        #[bstack_owned]
        owned_arr: [Leaf; 2],
        #[bstack_strong]
        strong: RcLeaf,
        #[bstack_weak]
        weak: WkLeaf,
        #[bstack_ref]
        reference: Leaf,
        #[bstack_owned]
        vec_owned: Vec<Leaf>,
        vec_pod: Vec<u8>,
        #[embed]
        embedded: Leaf,
        #[bstack_owned]
        foreign: Foreign<Leaf>,
        #[bstack_owned]
        foreign_vec: Vec<Foreign<Leaf>>,
        #[bstack_static(7u32)]
        const_var: u32,
        #[bstack_mut]
        #[bstack_static(0u64)]
        mut_var: u64,
    }

    #[bstack_class]
    enum EveryVariant {
        Unit,
        Pod(u32),
        Tup(u32, u16),
        Arr([u64; 2]),
        #[bstack_owned]
        Owns(Leaf),
        #[bstack_ref]
        Refs(Leaf),
        #[bstack_strong]
        Strong(RcLeaf),
        #[bstack_owned]
        Many(Vec<Leaf>),
        Bytes(Vec<u8>),
        Far = 33,
    }

    #[test]
    fn schema_survives_a_round_trip_and_a_class_var_write() {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_resync_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // 1. First sync writes every descriptor.
        let reg = rtti::sync(&path).unwrap();
        println!("first sync: {} types", reg.len());
        drop(reg);

        // 2. Second sync decodes them and compares against the compiled descriptors.
        //    A lossy encode/decode anywhere in the `Shape` grammar surfaces here.
        let reg = rtti::sync(&path).expect("re-sync must accept its own output");
        println!("re-sync ok");

        // Guard against a vacuous pass: every declared field/variant must actually be
        // in the decoded schema (unrecognised attributes are dropped silently).
        let ty = reg
            .load_type(reg.ordinal_of(Every::eightcc()).unwrap())
            .unwrap();
        let RttiBody::Struct(fs) = &ty.body else {
            panic!("struct body")
        };
        println!("Every: {} fields", fs.len());
        for f in fs {
            println!("    {:<12} @{:<3} {:?}", f.name, f.offset, f.shape);
        }
        assert_eq!(
            fs.len(),
            16,
            "a declared field went missing from the schema"
        );
        let ety = reg
            .load_type(reg.ordinal_of(EveryVariant::eightcc()).unwrap())
            .unwrap();
        let RttiBody::Enum(e) = &ety.body else {
            panic!("enum body")
        };
        println!("EveryVariant: {} variants", e.variants.len());
        assert_eq!(e.variants.len(), 10);

        // 3. Mutate a `#[bstack_mut] #[bstack_static]` class variable, then sync again.
        //    Its persisted value now differs from the compiled initial value; that must
        //    NOT read as schema drift.
        reg.set_class_value(Every::eightcc(), "mut_var", &99u64.to_le_bytes())
            .unwrap();
        // A const class variable must stay rejected.
        let const_err = reg.set_class_value(Every::eightcc(), "const_var", &1u32.to_le_bytes());
        println!(
            "set const_var -> {:?}",
            const_err.map(|_| ()).map_err(|e| e.to_string())
        );
        // `RttiType` documents mutable class-var values as "read live from the stack,
        // never cached here" — so `load_type` must now report 99, not the compiled 0.
        let ty2 = reg
            .load_type(reg.ordinal_of(Every::eightcc()).unwrap())
            .unwrap();
        let RttiBody::Struct(fs2) = &ty2.body else {
            panic!()
        };
        let mv = fs2.iter().find(|f| f.name == "mut_var").unwrap();
        println!("after write, mut_var shape = {:?}", mv.shape);
        drop(reg);

        let reg = rtti::sync(&path).expect("re-sync after a class-var write must succeed");
        println!("re-sync after class-var write ok; {} types", reg.len());
        drop(reg);

        std::fs::remove_file(&path).ok();
    }
}

mod rtti_wal {
    //! Regression: `RttiRegistry::teardown` routes its collected frees
    //! through the WAL on a non-bulk allocator, so a crash *between* the commit and the
    //! frees actually running is rolled forward by `finish` on the next open — rather
    //! than leaking permanently, as it did when the interpreter bypassed the WAL its
    //! static counterpart (`wal_teardown`) uses. Requires `--features fault-injection`.
    #![cfg(feature = "fault-injection")]
    use bstack::fault::FaultPolicy;
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti;
    use bstack_raii::{BStackBlock, BStackCast, BStackRaiiAllocator, bstack_class};
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

    /// Fail the first WAL-execute write *after* the transaction is committed — the
    /// `inplace_gen` that flips `txn_status` to `Complete`. The frees are then staged
    /// `Complete` but not yet applied: exactly a crash between commit and finish.
    struct FailAfterCommit {
        committed: AtomicBool,
        fired: AtomicBool,
    }
    impl FaultPolicy for FailAfterCommit {
        fn next_fault(&self, op: &'static str, _seq: u64) -> Option<io::Error> {
            if op == "inplace_gen" {
                self.committed.store(true, Ordering::SeqCst);
                return None;
            }
            if op == "set"
                && self.committed.load(Ordering::SeqCst)
                && !self.fired.swap(true, Ordering::SeqCst)
            {
                return Some(io::Error::other("injected post-commit crash"));
            }
            None
        }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("bstack_raii_rttiwal_{tag}_{stamp}.bstack"))
    }

    #[test]
    fn rtti_teardown_crash_is_recovered_by_finish() {
        let schema = tmp("schema");
        let data = tmp("data");
        let reg = rtti::sync(&schema).unwrap();
        let ord = reg.ordinal_of(Parent::eightcc()).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        // A non-bulk allocator, so the teardown takes the WAL path (not an atomic bulk free).
        assert!(!alloc.atomic_bulk());

        let child = Leaf::new(&alloc, 7).unwrap();
        let child_sz = std::mem::size_of::<<Leaf as BStackBlock>::OnDisk>() as u64;
        let child_off = child.handle().range().start();
        let parent = Parent::new(&alloc, child).unwrap();
        let parent_off = parent.handle().range().start();
        // The interpreter takes over ownership; the bare handle frees nothing on drop.
        let _ = parent.into_inner();

        // Crash the teardown right after it commits its free transaction.
        alloc
            .stack()
            .set_fault_policy(Some(Arc::new(FailAfterCommit {
                committed: AtomicBool::new(false),
                fired: AtomicBool::new(false),
            })));
        // SAFETY: `parent_off` is the live, detached root of this registered structure.
        let r = unsafe { reg.teardown(&alloc, ord, parent_off) };
        alloc.stack().set_fault_policy(None);
        assert!(
            r.is_err(),
            "the teardown never committed a WAL transaction (no post-commit fault \
         could land) — did it bypass the WAL?"
        );

        // Reopen and complete the crash-left transaction: the staged frees roll forward.
        drop(alloc);
        let alloc2 = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        let reclaimed = bstack_raii::finish(&alloc2).unwrap();
        assert!(
            reclaimed >= 2,
            "recovery reclaimed {reclaimed} slices — the interpreter's frees were not \
         WAL-backed, so the parent + child leaked permanently"
        );

        // The child block is genuinely free now: a fresh same-size allocation reuses it.
        let mut reused = false;
        for _ in 0..4 {
            let s = alloc2.alloc(child_sz).unwrap();
            if s.as_range().start() == child_off {
                reused = true;
            }
        }
        assert!(
            reused,
            "the recovered child block @{child_off} was not reclaimed"
        );

        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }

    /// Fail the first `copy` (a clone's block-body copy) and everything after it — a
    /// crash mid-clone that also prevents the in-process error path from cleaning up, so
    /// the intention-first `Pending` allocation is left for `finish` to reclaim on reopen.
    struct FailFromFirstCopy {
        armed: AtomicBool,
    }
    impl FaultPolicy for FailFromFirstCopy {
        fn next_fault(&self, op: &'static str, _seq: u64) -> Option<io::Error> {
            if op == "copy" {
                self.armed.store(true, Ordering::SeqCst);
            }
            if self.armed.load(Ordering::SeqCst) {
                return Some(io::Error::other("injected crash mid-clone"));
            }
            None
        }
    }

    #[test]
    fn rtti_clone_crash_is_recovered_by_finish() {
        // `clone_value` logs each freshly allocated block intention-first, so a crash
        // mid-clone leaves the orphaned partial clone reclaimable by `finish` on the next
        // open — rather than a permanent leak, as it was when the interpreter allocated
        // straight through the allocator with no WAL (its static `ClonePlan` counterpart
        // logs the same way).
        let schema = tmp("clschema");
        let data = tmp("cldata");
        let reg = rtti::sync(&schema).unwrap();
        let ord = reg.ordinal_of(Parent::eightcc()).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        assert!(
            alloc.wal_anchor().is_some(),
            "the allocator must be WAL-backed"
        );

        let child = Leaf::new(&alloc, 7).unwrap();
        let parent = Parent::new(&alloc, child).unwrap();
        let src_off = parent.handle().range().start();
        let _ = parent.into_inner();

        let base = alloc.len().unwrap();

        // Crash the clone as soon as it copies its first freshly-allocated block, and keep
        // failing so the in-process abandon (`finish_at_locked`) cannot reclaim either.
        alloc
            .stack()
            .set_fault_policy(Some(Arc::new(FailFromFirstCopy {
                armed: AtomicBool::new(false),
            })));
        // SAFETY: `src_off` is the live, detached root of this registered structure.
        let r = unsafe { reg.clone_value(&alloc, ord, src_off) };
        alloc.stack().set_fault_policy(None);
        assert!(r.is_err(), "the faulted clone should have failed");

        // The clone allocated at least one block before the crash; it grew the file and was
        // left orphaned (the in-process cleanup was also crashed).
        let grown = alloc.len().unwrap();
        assert!(grown > base, "the clone never allocated a block to orphan");

        // Reopen and complete the crash-left transaction: the orphaned clone block(s) are
        // reclaimed (without WAL backing this would be a permanent leak).
        drop(alloc);
        let alloc2 = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        let reclaimed = bstack_raii::finish(&alloc2).unwrap();
        assert!(
            reclaimed >= 1,
            "recovery reclaimed {reclaimed} slices — the interpreted clone's \
         allocations were not WAL-backed, so they leaked permanently"
        );

        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}
