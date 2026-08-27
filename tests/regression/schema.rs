//! On-disk schema / clone / enum-shape findings.
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

mod clonediff {
    //! Harness: aliasing oracle for the two deep-clone implementations.
    //!
    //! A deep clone must give the copy **its own** blocks for every owning shape, and
    //! must deliberately **share** the target of a non-owning or refcounted one. This
    //! builds one structure covering both classes, clones it via the generated
    //! `TryCloneIn` and via `RttiRegistry::clone_value`, and checks each field against
    //! its expected aliasing.
    #![allow(dead_code)]
    use bstack::{BStack, FirstFitBStackAllocator};
    use bstack_raii::rtti;
    use bstack_raii::{BStackBlock, BStackCast, BStackOwned, TryCloneIn, bstack_class};

    #[bstack_class]
    struct Leaf {
        v: u32,
    }

    #[bstack_class(rc)]
    struct RcLeaf {
        v: u32,
    }

    #[bstack_class]
    struct Wide {
        #[bstack_owned]
        a: Leaf,
        #[bstack_strong]
        c: RcLeaf,
        #[bstack_ref]
        d: Leaf,
        #[bstack_owned]
        many: Vec<Leaf>,
        bytes: Vec<u8>,
        optvec: Option<Vec<u8>>,
        optstr: Option<String>,
        arr: [Vec<u8>; 2],
    }

    /// (field, offset, must the clone share this block with the original?)
    fn probe<A: bstack_raii::BStackRaiiAllocator>(
        h: &Wide,
        alloc: &A,
    ) -> Vec<(&'static str, u64, bool)> {
        let stack = alloc.stack();
        let mut v = Vec::new();
        v.push(("a (owned)", h.get_a(stack).unwrap().range().start(), false));
        v.push(("c (strong)", h.get_c(stack).unwrap().range().start(), true));
        v.push(("d (ref)", h.get_d(stack).unwrap().range().start(), true));
        let many = h.get_many(alloc).unwrap();
        v.push((
            "many.data (owned vec)",
            many.descriptor().data_off.get(),
            false,
        ));
        v.push((
            "many[0] (owned elem)",
            many.get(0).unwrap().unwrap().range().start(),
            false,
        ));
        v.push((
            "bytes.data (pod vec)",
            h.get_bytes(alloc).unwrap().descriptor().data_off.get(),
            false,
        ));
        v.push((
            "optvec.data (Option<Vec>)",
            h.get_optvec(alloc)
                .unwrap()
                .unwrap()
                .descriptor()
                .data_off
                .get(),
            false,
        ));
        v.push((
            "optstr.data (Option<String>)",
            h.get_optstr(alloc)
                .unwrap()
                .unwrap()
                .descriptor()
                .data_off
                .get(),
            false,
        ));
        let arr = h.get_arr(alloc).unwrap();
        v.push((
            "arr[0].data ([Vec;2])",
            arr[0].descriptor().data_off.get(),
            false,
        ));
        v.push((
            "arr[1].data ([Vec;2])",
            arr[1].descriptor().data_off.get(),
            false,
        ));
        v
    }

    #[test]
    fn deep_clone_aliases_exactly_the_shared_shapes() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_cd_s_{stamp}.bstack"));
        let data = std::env::temp_dir().join(format!("bstack_raii_cd_d_{stamp}.bstack"));
        let reg = rtti::sync(&schema).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();

        let d_keep: BStackOwned<Leaf> = Leaf::new(&alloc, 4).unwrap();
        let root = Wide::new(
            &alloc,
            Leaf::new(&alloc, 1).unwrap(),
            RcLeaf::new(&alloc, 3).unwrap(),
            unsafe { bstack_raii::BStackRef::<Leaf>::from_range(d_keep.handle().range()) },
            vec![Leaf::new(&alloc, 5).unwrap()],
            b"hello",
            Some(b"world".as_slice()),
            Some("string!"),
            [b"aa".as_slice(), b"bb".as_slice()],
        )
        .unwrap();
        let orig = probe(&root.handle(), &alloc);

        let static_clone: Wide = {
            let c = root.try_clone_in(&alloc).unwrap();
            let r = c.handle().range();
            let _ = c.into_inner();
            unsafe { <Wide as BStackBlock>::from_range(r) }
        };
        let rtti_clone: Wide = {
            let ord = reg.ordinal_of(Wide::eightcc()).unwrap();
            let off =
                unsafe { reg.clone_value(&alloc, ord, root.handle().range().start()) }.unwrap();
            unsafe {
                <Wide as BStackBlock>::from_range(bstack::BStackRange::new(
                    off,
                    core::mem::size_of::<<Wide as BStackBlock>::OnDisk>() as u64,
                ))
            }
        };

        let mut all_bad: Vec<String> = Vec::new();
        for (label, clone_h) in [
            ("static TryCloneIn", static_clone),
            ("RttiRegistry::clone_value", rtti_clone),
        ] {
            println!("\n--- {label} ---");
            let cl = probe(&clone_h, &alloc);
            let mut bad: Vec<&str> = Vec::new();
            for ((name, o, want_shared), (_, c, _)) in orig.iter().zip(cl.iter()) {
                let shared = o == c;
                let verdict = if shared == *want_shared { "ok " } else { "BAD" };
                println!(
                    "  {verdict} {name:<26} orig @{o:<5} clone @{c:<5} shared={shared} (want {want_shared})"
                );
                if shared != *want_shared {
                    bad.push(name);
                }
            }
            println!("  mismatched: {bad:?}");
            all_bad.extend(bad.iter().map(|n| format!("{label}: {n}")));
        }

        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
        let _ = d_keep;

        // Both clone paths must agree with the aliasing contract. `RttiRegistry::clone_value`
        // records an `Option<Vec>` / `Option<String>` field as the container region rather
        // than opaque POD (its Rust handle width), so it allocates the copy its own data
        // block instead of byte-copying the `VecDesc` verbatim and aliasing the original's
        // block — which would be a double free on the second teardown.
        assert!(
            all_bad.is_empty(),
            "deep-clone aliasing diverged from contract: {all_bad:?}"
        );
    }
}

mod tddiff {
    //! Harness: differential between the two teardown implementations.
    //!
    //! Builds one structure covering the main owning shapes, records every block it
    //! allocates, runs `RttiRegistry::teardown`, then probes each recorded range: a
    //! range that still frees cleanly is one RTTI **missed**; one that reports a double
    //! free is one RTTI reclaimed. The static `bstack_drop` walk is the reference for
    //! what *should* have been freed.
    #![allow(dead_code)]
    use bstack::{BStack, FirstFitBStackAllocator};
    use bstack_raii::rtti;
    use bstack_raii::{BStackBlock, BStackCast, BStackOwned, bstack_class};

    #[bstack_class]
    struct Leaf {
        v: u32,
    }

    #[bstack_class(rc)]
    struct RcLeaf {
        v: u32,
    }

    #[bstack_class]
    struct Wide {
        #[bstack_owned]
        a: Leaf,
        #[bstack_owned]
        b: Option<Leaf>,
        #[bstack_strong]
        c: RcLeaf,
        #[bstack_ref]
        d: Leaf,
        #[bstack_owned]
        many: Vec<Leaf>,
        bytes: Vec<u8>,
        optvec: Option<Vec<u8>>,
    }

    #[test]
    fn rtti_teardown_frees_what_the_static_walk_frees() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_tdd_s_{stamp}.bstack"));
        let data = std::env::temp_dir().join(format!("bstack_raii_tdd_d_{stamp}.bstack"));
        let reg = rtti::sync(&schema).unwrap();
        // FirstFit *reports* a double free as `Err`; DebugCheckingAllocator panics, which
        // would stop the sweep at the first reclaimed block.
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();

        // --- build ------------------------------------------------------------
        let a = Leaf::new(&alloc, 1).unwrap();
        let b = Leaf::new(&alloc, 2).unwrap();
        let c = RcLeaf::new(&alloc, 3).unwrap();
        let d_keep: BStackOwned<Leaf> = Leaf::new(&alloc, 4).unwrap(); // `ref` target, ours
        let m0 = Leaf::new(&alloc, 5).unwrap();
        let m1 = Leaf::new(&alloc, 6).unwrap();

        let leaf_sz = core::mem::size_of::<<Leaf as BStackBlock>::OnDisk>() as u64;
        let mut rec: Vec<(&str, u64, u64)> = Vec::new(); // (name, off, len)
        rec.push(("a", a.handle().range().start(), leaf_sz));
        rec.push(("b", b.handle().range().start(), leaf_sz));
        rec.push(("many[0]", m0.handle().range().start(), leaf_sz));
        rec.push(("many[1]", m1.handle().range().start(), leaf_sz));
        let c_data = c.handle().range().start();
        rec.push((
            "c.data",
            c_data,
            core::mem::size_of::<<RcLeaf as BStackBlock>::OnDisk>() as u64,
        ));

        let root = Wide::new(
            &alloc,
            a,
            Some(b),
            c,
            unsafe { bstack_raii::BStackRef::<Leaf>::from_range(d_keep.handle().range()) },
            vec![m0, m1],
            b"hello",
            Some(b"world".as_slice()),
        )
        .unwrap();
        let root_off = root.handle().range().start();
        rec.push((
            "root",
            root_off,
            core::mem::size_of::<<Wide as BStackBlock>::OnDisk>() as u64,
        ));
        // Vec data blocks are allocated by the constructor; read them back.
        let many_d = root.handle().get_many(&alloc).unwrap().descriptor();
        let bytes_d = root.handle().get_bytes(&alloc).unwrap().descriptor();
        let optvec_d = root
            .handle()
            .get_optvec(&alloc)
            .unwrap()
            .unwrap()
            .descriptor();
        rec.push(("many.data", many_d.data_off.get(), many_d.data_size));
        rec.push(("bytes.data", bytes_d.data_off.get(), bytes_d.data_size));
        rec.push(("optvec.data", optvec_d.data_off.get(), optvec_d.data_size));
        // Control: a `#[bstack_ref]` target owns nothing, so teardown must NOT free it.
        // It should come back as "still allocated".
        rec.push((
            "d (ref, must survive)",
            d_keep.handle().range().start(),
            leaf_sz,
        ));

        // --- RTTI teardown ----------------------------------------------------
        let ord = reg.ordinal_of(Wide::eightcc()).unwrap();
        let _ = root.into_inner(); // RTTI owns the shell now
        unsafe { reg.teardown(&alloc, ord, root_off) }.unwrap();

        // --- probe ------------------------------------------------------------
        println!("after RttiRegistry::teardown:");
        let mut missed = Vec::new();
        for (name, off, len) in &rec {
            let r =
                unsafe { bstack_raii::dealloc_range(&alloc, bstack::BStackRange::new(*off, *len)) };
            match r {
                Ok(()) => {
                    println!("  MISSED  {name:<12} @{off} ({len} B) — still allocated");
                    missed.push(*name);
                }
                Err(e) => println!("  freed   {name:<12} @{off} ({len} B)  [{e}]"),
            }
        }
        println!("\nRTTI missed: {missed:?}");
        assert!(
            missed.contains(&"d (ref, must survive)"),
            "a `ref` target must not be freed by teardown"
        );
        let _ = d_keep.into_inner(); // the probe above already freed it

        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}

mod enumschema {
    //! Harness: differential check of the `#[bstack_class] enum` RTTI schema
    //! against the real compiled on-disk layout — the enum counterpart of the struct
    //! field-width sweep. Verifies disc offset/width, payload offset, and that every
    //! variant's fields are in declaration order, inside the payload, non-overlapping,
    //! and no wider than the space they claim.
    #![allow(dead_code)]
    use bstack_raii::rtti::{self, RttiBody, RttiRegistry, Shape};
    use bstack_raii::{BStackBlock, BStackCast, bstack_class};

    #[bstack_class]
    struct Leaf {
        v: u32,
    }

    #[bstack_class(rc)]
    struct RcLeaf {
        v: u32,
    }

    #[bstack_class]
    enum Wide {
        Unit,
        Pod(u32),
        PodPair(u32, u16),
        Arr([u64; 3]),
        #[bstack_owned]
        Owns(Leaf),
        #[bstack_ref]
        Refs(Leaf),
        #[bstack_strong]
        Strong(RcLeaf),
        #[bstack_owned]
        Many(Vec<Leaf>),
        Bytes(Vec<u8>),
        // The three shapes — `Option<Vec<u8>>`, `Option<String>`, `[Vec<u8>; 2]` —
        // do NOT compile here: the variant payload must be `Pod` / `PodInOption`.
        // They compile as struct fields, where they were once mis-recorded as
        // `Pod { 24 }`; that struct-path bug is now fixed.
        Big = 40,
    }

    fn claimed(sh: &Shape, reg: &RttiRegistry) -> Option<u64> {
        Some(match sh {
            Shape::Pod { width } => *width as u64,
            Shape::Owned(_) | Shape::Strong(_) | Shape::Weak(_) | Shape::Ref(_) => 8,
            Shape::Vec(_) | Shape::Foreign { .. } => 16,
            Shape::Option(i) => claimed(i, reg)?,
            Shape::Array { n, inner } => *n as u64 * claimed(inner, reg)?,
            Shape::Tuple(items) => items.iter().map(|i| claimed(i, reg)).sum::<Option<u64>>()?,
            Shape::Embed(tag) => reg.load_type(reg.ordinal_of(*tag)?).ok()?.ondisk_size,
            Shape::Class { .. } => return None,
        })
    }

    #[test]
    fn enum_schema_matches_the_compiled_layout() {
        let schema = std::env::temp_dir().join(format!(
            "bstack_raii_enumschema_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let reg = rtti::sync(&schema).unwrap();
        let ord = reg.ordinal_of(Wide::eightcc()).unwrap();
        let ty = reg.load_type(ord).unwrap();
        let od_size = core::mem::size_of::<<Wide as BStackBlock>::OnDisk>() as u64;
        assert_eq!(ty.ondisk_size, od_size, "ondisk_size");

        let RttiBody::Enum(e) = &ty.body else {
            panic!("expected an enum body")
        };
        println!(
            "OnDisk = {od_size} B; disc_off = {}, disc_width = {}, payload_off = {}",
            e.disc_off, e.disc_width, e.payload_off
        );
        let payload_space = od_size - e.payload_off as u64;
        println!("payload area = {payload_space} B\n");

        let mut bad: Vec<String> = Vec::new();

        // The discriminant must sit inside the block and not overlap the payload.
        if (e.disc_off as u64) + (e.disc_width as u64) > e.payload_off as u64 {
            bad.push(format!(
                "disc [{}..{}) overlaps payload_off {}",
                e.disc_off,
                e.disc_off as u64 + e.disc_width as u64,
                e.payload_off
            ));
        }
        if e.payload_off as u64 > od_size {
            bad.push("payload_off past the block".into());
        }

        for v in &e.variants {
            let mut cursor = 0u64;
            let mut desc = Vec::new();
            for f in &v.fields {
                let w = claimed(&f.shape, &reg);
                desc.push(format!(
                    "{}@{} claims {:?} as {:?}",
                    f.name, f.offset, w, f.shape
                ));
                let off = f.offset as u64;
                if off < cursor {
                    bad.push(format!(
                        "{}::{} @{off} overlaps previous end {cursor}",
                        v.name, f.name
                    ));
                }
                match w {
                    Some(w) => {
                        if off + w > payload_space {
                            bad.push(format!(
                                "{}::{} @{off}+{w} runs past the {payload_space}-byte payload",
                                v.name, f.name
                            ));
                        }
                        cursor = off + w;
                    }
                    None => bad.push(format!("{}::{} has an unmeasurable shape", v.name, f.name)),
                }
            }
            println!(
                "  {:<8} disc {:>3}  {}",
                v.name,
                v.disc_value,
                if desc.is_empty() {
                    "-".into()
                } else {
                    desc.join(", ")
                }
            );
        }

        println!("\nproblems: {bad:?}");
        assert!(bad.is_empty(), "{bad:#?}");
        std::fs::remove_file(&schema).ok();
    }
}

mod enumdrop {
    //! Harness: the drop/clone contract per **enum variant kind**. `enum_.rs`
    //! emits its own drop_arms / clone_arms, which the struct differentials never touch.
    #![allow(dead_code)]
    use bstack::{BStack, BStackRange, FirstFitBStackAllocator};
    use bstack_raii::{
        BStackBlock, BStackDrop, BStackOwned, TryCloneIn, bstack_block, bstack_enum,
    };

    #[bstack_block]
    struct Leaf {
        v: u32,
    }

    #[bstack_enum]
    enum E {
        Unit,
        Pod(u32),
        #[bstack_owned]
        Owns(Leaf),
        #[bstack_ref]
        Refs(Leaf),
        #[bstack_owned]
        Many(Vec<Leaf>),
    }

    type A = FirstFitBStackAllocator;

    fn lsz() -> u64 {
        core::mem::size_of::<<Leaf as BStackBlock>::OnDisk>() as u64
    }

    fn still_allocated(a: &A, off: u64) -> bool {
        unsafe { bstack_raii::dealloc_range(a, BStackRange::new(off, lsz())) }.is_ok()
    }

    fn run(name: &str, build: impl Fn(&A, &mut Vec<u64>) -> BStackOwned<E>, expect_owned: bool) {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_ed_{name}_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

        let mut offs = Vec::new();
        let orig = build(&alloc, &mut offs);
        let _clone = orig.try_clone_in(&alloc).unwrap();

        orig.bstack_drop(&alloc).unwrap();
        let leaked: Vec<u64> = offs
            .iter()
            .copied()
            .filter(|&o| still_allocated(&alloc, o))
            .collect();

        let verdict = if expect_owned {
            if leaked.is_empty() { "ok " } else { "BAD" }
        } else if leaked.is_empty() {
            "BAD"
        } else {
            "ok "
        };
        println!(
            "  {verdict} {name:<10} children {offs:?}  still allocated after drop: {leaked:?}"
        );
        if expect_owned {
            assert!(leaked.is_empty(), "{name}: owning variant leaked its child");
        } else {
            assert!(
                !leaked.is_empty(),
                "{name}: a `ref` variant must NOT free its target"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn enum_variants_drop_and_clone() {
        println!();
        run("Unit", |a, _| E::new(a, EData::Unit).unwrap(), true);
        run("Pod", |a, _| E::new(a, EData::Pod(5)).unwrap(), true);
        run(
            "Owns",
            |a, offs| {
                let l = Leaf::new(a, 1).unwrap();
                offs.push(l.handle().range().start());
                E::new(a, EData::Owns(l)).unwrap()
            },
            true,
        );
        run(
            "Many",
            |a, offs| {
                let v: Vec<BStackOwned<Leaf>> = (0..3)
                    .map(|i| {
                        let l = Leaf::new(a, i).unwrap();
                        offs.push(l.handle().range().start());
                        l
                    })
                    .collect();
                let bv = bstack_raii::BStackBlockVec::from_handles(a, v).unwrap();
                E::new(a, EData::Many(bv)).unwrap()
            },
            true,
        );
    }

    /// A `#[bstack_ref]` variant owns nothing: the target must survive the enum's drop.
    #[test]
    fn ref_variant_does_not_free_its_target() {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_ed_ref_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let alloc = A::new(BStack::open(&path).unwrap()).unwrap();
        let target = Leaf::new(&alloc, 42).unwrap();
        let t_off = target.handle().range().start();
        // SAFETY (harness only): `target` is a live `Leaf` we own; this just names it.
        let r = unsafe { bstack_raii::BStackRef::<Leaf>::from_range(target.handle().range()) };
        let e = E::new(&alloc, EData::Refs(r)).unwrap();
        let clone = e.try_clone_in(&alloc).unwrap();
        e.bstack_drop(&alloc).unwrap();
        println!(
            "  ref target @{t_off} still allocated after the enum's drop: {}",
            still_allocated(&alloc, t_off)
        );
        let _ = clone.into_inner();
        let _ = target.into_inner();
        std::fs::remove_file(&path).ok();
    }
}

mod embedswap {
    //! Probe (partially open): an `#[embed]`ed child's bytes are
    //! byte-identical to a standalone block, header and all, so `verify_data_block`
    //! cannot tell an embed slot from a real block and `swap` accepts it into an owning
    //! field. The *complete* fix is the affine-handle redesign (Phase 1: a non-owning
    //! embed accessor); until then the residual is bounded by `bstack` refusing the
    //! interior free — teardown fails and leaks rather than corrupting the free list.
    //! This test pins that bounded outcome: the parent's interior is never handed back.
    //!
    //! The `unsafe` here fabricates the exact credential the interpreter still trusts;
    //! it is the demonstration, not incidental.
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti::{self, AnyRef, RttiBody};
    use bstack_raii::{BStackBlock, BStackCast, bstack_class};

    #[bstack_class]
    struct Child {
        v: u32,
    }

    #[bstack_class]
    struct Parent {
        #[embed]
        e: Child,
        #[bstack_mut]
        #[bstack_owned]
        o: Option<Child>,
    }

    #[test]
    fn embedded_region_as_an_owned_target() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_es_s_{stamp}.bstack"));
        let data = std::env::temp_dir().join(format!("bstack_raii_es_d_{stamp}.bstack"));
        let reg = rtti::sync(&schema).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        let stack = alloc.stack();

        let p = Parent::new(&alloc, Child::new(&alloc, 7).unwrap(), None).unwrap();
        let p_off = p.handle().range().start();
        let p_len = core::mem::size_of::<<Parent as BStackBlock>::OnDisk>() as u64;
        let pord = reg.ordinal_of(Parent::eightcc()).unwrap();

        // The embed slot's offset comes straight out of the schema — a safe read.
        let ty = reg.load_type(pord).unwrap();
        let RttiBody::Struct(fs) = &ty.body else {
            panic!()
        };
        let e_rel = fs.iter().find(|f| f.name == "e").unwrap().offset as u64;
        let embed_off = p_off + e_rel;
        println!(
            "Parent @{p_off}..{}  embed slot @{embed_off}",
            p_off + p_len
        );

        // Does the embedded region carry the child's own header?
        match AnyRef::from_block(stack, embed_off) {
            Ok(a) => println!(
                "AnyRef::from_block(embed_off) -> tag {:?}; is Child: {}",
                a.tag().0,
                a.tag() == Child::eightcc()
            ),
            Err(e) => println!("AnyRef::from_block(embed_off) -> Err({e})"),
        }

        // Point the parent's OWNING field at its own interior (still accepted — the
        // complete fix is Phase 1's non-owning embed accessor).
        let r = reg.swap(stack, pord, p_off, &["o"], unsafe {
            AnyRef::new(Child::eightcc(), embed_off)
        });
        println!(
            "swap(o := embed_off) -> {:?}",
            r.as_ref().err().map(|e| e.to_string())
        );

        // Teardown of the parent whose (forged) owning field names its own interior.
        // The interpreter frees children before parents, so this
        // completes without corrupting the free list: the interior slice double-frees
        // *within* the parent block being torn down (the allocator merges it) instead
        // of writing a bogus free-list node inside an already-freed region — the actual
        // corruption a parent-first free produced. The forged structure is only
        // reachable via `unsafe` (the safe embed accessor is non-owning).
        let td = unsafe { reg.teardown(&alloc, pord, p_off) };

        // The stable safety property: the allocator's free list is intact and usable —
        // a fresh allocation succeeds (rather than panicking on a corrupt node) and
        // hands back a well-formed, independently-freeable block. The parent itself was
        // torn down, so its storage being reusable is correct, not corruption.
        let probe = alloc
            .alloc(16)
            .unwrap_or_else(|e| {
                panic!(
                    "free list corrupted by the forged teardown: {e} (td={:?})",
                    td.err().map(|e| e.to_string())
                )
            })
            .as_range();
        assert_eq!(probe.len(), 16, "allocation returned a malformed range");
        // Round-trips cleanly — no corruption.
        unsafe { bstack_raii::dealloc_range(&alloc, probe) }.unwrap();

        let _ = p.into_inner();
        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}

mod null {
    //! Regression: `rtti::swap` preserves a field's nullability and
    //! rejects a null `AnyRef` into a **non-nullable** owned field (`[BSTACK0815]`), and
    //! the generated teardown guards `!= 0` unconditionally — so a null is never freed as
    //! the reserved region at offset 0.
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti::{self, AnyRef};
    use bstack_raii::{BStackBlock, BStackCast, BStackDrop, BStackOwned, bstack_class};

    #[bstack_class]
    struct Child {
        v: u32,
    }

    #[bstack_class]
    struct Parent {
        #[bstack_owned]
        c: Child, // NOT Option<Child>
    }

    #[test]
    fn null_into_non_nullable_owned_field() {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_null_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let schema = path.with_extension("schema");
        let reg = rtti::sync(&schema).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
        let pord = reg.ordinal_of(Parent::eightcc()).unwrap();

        let child: BStackOwned<Child> = Child::new(&alloc, 42).unwrap();
        let parent: BStackOwned<Parent> = Parent::new(&alloc, child).unwrap();
        let poff = parent.handle().range().start();

        // Installing a NULL reference into a field the schema says is non-nullable must
        // now be rejected.
        let r = reg.swap(alloc.stack(), pord, poff, &["c"], unsafe {
            AnyRef::new(Child::eightcc(), 0)
        });
        let msg = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("BSTACK0815"),
            "swap accepted a null into a non-nullable field: {msg:?}"
        );

        // The field is untouched: it still owns the original child, and a normal
        // teardown reclaims everything cleanly.
        parent.bstack_drop(&alloc).unwrap();

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&schema).ok();
    }
}

mod generic {
    //! Harness: the drop/clone contract for **generic** blocks. Generic
    //! instantiation goes through its own codegen path (parameter usage analysis,
    //! per-parameter tag mixing, const-generic array sizing) that the concrete-type
    //! differentials never exercise.
    #![allow(dead_code)]
    use bstack::{BStack, BStackAllocator, BStackRange, FirstFitBStackAllocator};
    use bstack_raii::{BStackBlock, BStackDrop, BStackOwned, TryCloneIn, bstack_block};

    #[bstack_block]
    struct Leaf {
        v: u32,
    }

    #[bstack_block]
    struct GenHolder<T> {
        tag: u32,
        #[bstack_owned]
        child: T,
        #[bstack_owned]
        many: Vec<T>,
    }

    #[bstack_block]
    struct GenArr<T, const N: usize> {
        #[bstack_owned]
        arr: [T; N],
    }

    type A = FirstFitBStackAllocator;

    fn sz<T: BStackBlock>() -> u64 {
        core::mem::size_of::<T::OnDisk>() as u64
    }

    fn still_allocated(a: &A, off: u64, len: u64) -> bool {
        unsafe { bstack_raii::dealloc_range(a, BStackRange::new(off, len)) }.is_ok()
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "bstack_raii_gen_{tag}_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn generic_holder_drop_and_clone() {
        let path = tmp("holder");
        let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

        let mut offs = Vec::new();
        let child = Leaf::new(&alloc, 9).unwrap();
        offs.push(child.handle().range().start());
        let many: Vec<BStackOwned<Leaf>> = (0..3)
            .map(|i| {
                let l = Leaf::new(&alloc, i).unwrap();
                offs.push(l.handle().range().start());
                l
            })
            .collect();

        let orig: BStackOwned<GenHolder<Leaf>> =
            GenHolder::<Leaf>::new(&alloc, 7, child, many).unwrap();
        let clone = orig.try_clone_in(&alloc).unwrap();

        let read = |h: &GenHolder<Leaf>| -> (u32, u32, Vec<u32>) {
            let s = alloc.stack();
            (
                h.get_tag(s).unwrap(),
                h.get_child(s).unwrap().get_v(s).unwrap(),
                h.get_many(&alloc)
                    .unwrap()
                    .to_vec()
                    .unwrap()
                    .iter()
                    .map(|x| x.get_v(s).unwrap())
                    .collect(),
            )
        };
        let before = read(clone.handle());
        orig.bstack_drop(&alloc).unwrap();
        let leaked: Vec<u64> = offs
            .iter()
            .copied()
            .filter(|&o| still_allocated(&alloc, o, sz::<Leaf>()))
            .collect();
        let mut scribble = Vec::new();
        for _ in 0..8 {
            let mut sl = alloc.alloc(sz::<Leaf>()).unwrap();
            sl.write(vec![0xFFu8; sz::<Leaf>() as usize]).unwrap();
            scribble.push(sl.as_range());
        }
        let after = read(clone.handle());
        for r in scribble {
            let _ = unsafe { bstack_raii::dealloc_range(&alloc, r) };
        }

        println!(
            "GenHolder<Leaf>  clone {before:?} -> {after:?}   original's children leaked: {leaked:?}"
        );
        assert!(leaked.is_empty(), "generic holder leaked its children");
        assert_eq!(before, after, "generic clone aliased the original");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn const_generic_array_drop_and_clone() {
        let path = tmp("arr");
        let alloc = A::new(BStack::open(&path).unwrap()).unwrap();

        let mut offs = Vec::new();
        let elems: [BStackOwned<Leaf>; 3] = core::array::from_fn(|i| {
            let l = Leaf::new(&alloc, i as u32).unwrap();
            offs.push(l.handle().range().start());
            l
        });
        let orig: BStackOwned<GenArr<Leaf, 3>> = GenArr::<Leaf, 3>::new(&alloc, elems).unwrap();
        let clone = orig.try_clone_in(&alloc).unwrap();

        let read = |h: &GenArr<Leaf, 3>| -> Vec<u32> {
            let s = alloc.stack();
            h.get_arr(s)
                .unwrap()
                .iter()
                .map(|x| x.get_v(s).unwrap())
                .collect()
        };
        let before = read(clone.handle());
        orig.bstack_drop(&alloc).unwrap();
        let leaked: Vec<u64> = offs
            .iter()
            .copied()
            .filter(|&o| still_allocated(&alloc, o, sz::<Leaf>()))
            .collect();
        let mut scribble = Vec::new();
        for _ in 0..8 {
            let mut sl = alloc.alloc(sz::<Leaf>()).unwrap();
            sl.write(vec![0xFFu8; sz::<Leaf>() as usize]).unwrap();
            scribble.push(sl.as_range());
        }
        let after = read(clone.handle());
        for r in scribble {
            let _ = unsafe { bstack_raii::dealloc_range(&alloc, r) };
        }

        println!(
            "GenArr<Leaf,3>   clone {before:?} -> {after:?}   original's children leaked: {leaked:?}"
        );
        assert!(leaked.is_empty(), "const-generic array leaked its children");
        assert_eq!(before, after, "const-generic clone aliased the original");
        std::fs::remove_file(&path).ok();
    }
}
