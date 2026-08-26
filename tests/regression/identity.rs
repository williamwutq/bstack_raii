//! Type-identity / tag findings.
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

mod modpath {
    //! Regression: the type tag folds the type's `module_path!()`,
    //! so two `#[bstack_class]` types with the **same identifier in different modules**
    //! of one crate hash to distinct tags. `v1::Node` and `v2::Node` each keep their own
    //! tag, so `sync` registers both and every RTTI operation resolves a tag to that
    //! type's own layout — never to a same-named type in another module.
    //!
    //! The two fixtures below are byte-for-byte identical in name and layout; the only
    //! thing that distinguishes them is the module they live in. So a distinct tag here
    //! is `module_path!()` folding doing its job, nothing else.
    #![allow(dead_code)]

    use bstack::{BStack, FirstFitBStackAllocator};
    use bstack_raii::BStackCast;
    use bstack_raii::rtti;

    mod v1 {
        #[bstack_raii::bstack_class]
        pub struct Node {
            pub a: u64,
        }
    }

    mod v2 {
        #[bstack_raii::bstack_class]
        pub struct Node {
            pub a: u64,
        }
    }

    #[test]
    fn same_name_different_module_get_distinct_tags() {
        let t1 = <v1::Node as BStackCast>::eightcc();
        let t2 = <v2::Node as BStackCast>::eightcc();
        // The readable prefix is derived from the (identical) name, so it matches...
        assert_eq!(
            t1.0[0..2],
            t2.0[0..2],
            "prefixes should share the name's initials"
        );
        // ...but the module-path-folded hash tail must differ.
        assert_ne!(
            t1, t2,
            "same-named types in different modules must not share a tag"
        );
        assert_ne!(
            t1.0[2..],
            t2.0[2..],
            "the hash tail must carry the module path"
        );
    }

    #[test]
    fn both_modules_register_without_collapse() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_modpath_{stamp}.bstack"));

        let reg = rtti::sync(&schema).unwrap();

        // Both distinct types are registered (the pre-fix bug silently dropped one,
        // leaving a single ordinal that resolved the shared tag to the wrong layout).
        let o1 = reg
            .ordinal_of(<v1::Node as BStackCast>::eightcc())
            .expect("v1::Node registered");
        let o2 = reg
            .ordinal_of(<v2::Node as BStackCast>::eightcc())
            .expect("v2::Node registered");
        assert_ne!(o1, o2, "the two Nodes must occupy distinct ordinals");

        // Sanity: a fresh allocator over the same schema round-trips (no corruption).
        let data = std::env::temp_dir().join(format!("bstack_raii_modpath_d_{stamp}.bstack"));
        let _alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();

        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}

mod tag8 {
    //! Regression: an explicit 8-byte `tag = "…"` left no hash
    //! bytes, so `mix` — which only perturbs high-bit-set bytes — became the identity and
    //! every instantiation of a generic shared one tag; a safe `bstack_cast!` to the
    //! wrong instantiation then wrote past the block. Two fixes verified here:
    //!
    //! 1. An 8-byte tag on a generic item is now rejected at expansion time
    //!    (`[BSTACK0005]`, asserted by a `compile_fail` doctest in `src/lib.rs`).
    //! 2. The cast gate carries a second check: the allocator-attested slice length
    //!    must equal the target's on-disk size, so even a colliding tag cannot produce
    //!    a wrong-size handle.
    #![allow(dead_code)]
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::{BStackCast, bstack_block, bstack_cast};

    // A 5-byte tag (the auto-prefix ceiling) keeps 3 hash bytes for `mix`.
    #[bstack_block(tag = "PINTG")]
    struct Pinned<T: bstack_raii::Pod> {
        marker: u32,
        #[bstack_mut]
        payload: T,
    }

    #[test]
    fn instantiations_are_distinguished_and_size_gated() {
        let small = <Pinned<u32> as BStackCast>::eightcc();
        let big = <Pinned<[u64; 8]> as BStackCast>::eightcc();
        // With hash bytes present, `mix` distinguishes the instantiations.
        assert_ne!(small, big, "instantiation tags must differ");

        let path = std::env::temp_dir().join(format!(
            "bstack_raii_tag8_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();

        let real = Pinned::<u32>::new(&alloc, 0xAAAA, 7).unwrap();
        let stack = alloc.stack();
        let sl = real.handle().as_slice(stack);

        // The wrong-instantiation cast is refused — by the tag now, and (were the tags
        // ever to collide) independently by the slice-length gate.
        assert!(
            bstack_cast!(sl as Pinned<[u64; 8]>).unwrap().is_none(),
            "cast to a different-size instantiation must be rejected"
        );
        // The right instantiation still casts.
        assert!(bstack_cast!(sl as Pinned<u32>).unwrap().is_some());

        let _ = real.into_inner();
        std::fs::remove_file(&path).ok();
    }
}

mod ctrltag {
    //! Regression: for an `(rc, weak)` block the control tag
    //! defaulted to the data prefix **lowercased**, and `to_ascii_lowercase` is a no-op
    //! on caseless bytes — so a prefix with no cased letter (all-digits, all-symbols)
    //! yielded `ctrl_tag == data_tag`, collapsing the distinction `verify_data_block`
    //! relies on to keep a control block out of a data slot.
    //!
    //! The colliding declarations (`tag = "1234"`, `tag = "**"`) are now rejected at
    //! expansion time (`[BSTACK0006]`, asserted by a `compile_fail` doctest in
    //! `src/lib.rs`). Here we confirm that an ordinary cased-prefix type keeps the two
    //! tags distinct.
    #![allow(dead_code)]
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti::AnyRef;
    use bstack_raii::{BStackBlock, BStackCast, bstack_block};

    #[bstack_block(rc, weak)]
    struct Normal {
        v: u32,
    }

    #[test]
    fn control_tag_differs_from_data_tag() {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_ctag_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
        let stack = alloc.stack();

        let n = Normal::new(&alloc, 1).unwrap();
        let data_off = n.handle().range().start();
        // The data block's control back-pointer sits right after the 16-byte header.
        let mut b = [0u8; 8];
        stack.get_into(data_off + 16, &mut b).unwrap();
        let ctrl_off = u64::from_le_bytes(b);

        let data_tag = <Normal as BStackCast>::eightcc();
        let ctrl_tag = AnyRef::from_block(stack, ctrl_off).unwrap().tag();
        assert_ne!(data_tag, ctrl_tag, "control tag must differ from data tag");

        std::fs::remove_file(&path).ok();
    }
}

mod ctrlswap {
    //! Regression: a control block must not be
    //! installable into an `#[bstack_owned]` slot. With a normally-tagged `(rc, weak)`
    //! type the control tag differs from the data tag, so `verify_data_block` rejects
    //! the move; the colliding all-caseless-tag types that would defeat this
    //! (`tag = "1234"` etc.) do not compile (`[BSTACK0006]`).
    use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
    use bstack_raii::rtti::{self, AnyRef};
    use bstack_raii::{BStackBlock, BStackCast, bstack_class};

    #[bstack_class(rc, weak)]
    struct Normal {
        v: u32,
    }

    #[bstack_class]
    struct HolderN {
        #[bstack_mut]
        #[bstack_weak]
        w: Option<Normal>,
        #[bstack_owned]
        o: Option<Normal>,
    }

    #[test]
    fn control_block_rejected_as_an_owned_target() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let schema = std::env::temp_dir().join(format!("bstack_raii_cs_s_{stamp}.bstack"));
        let data = std::env::temp_dir().join(format!("bstack_raii_cs_d_{stamp}.bstack"));
        let reg = rtti::sync(&schema).unwrap();
        let alloc = FirstFitBStackAllocator::new(BStack::open(&data).unwrap()).unwrap();
        let stack = alloc.stack();

        let n = Normal::new(&alloc, 1).unwrap();
        let hn = HolderN::new(&alloc, None).unwrap();
        hn.handle().set_w(&alloc, n.downgrade().unwrap()).unwrap();
        let hn_ord = reg.ordinal_of(HolderN::eightcc()).unwrap();
        let hn_off = hn.handle().range().start();

        // Take the weak field's control `AnyRef` out (safe; `swap` hands it back).
        // Nulling a *nullable* weak field is allowed, so this null swap is fine.
        let ctrl_n = reg
            .swap(stack, hn_ord, hn_off, &["w"], unsafe {
                AnyRef::new(Normal::eightcc(), 0)
            })
            .unwrap()
            .expect("weak field was set");

        // Installing that control block into the OWNED field must be rejected: the
        // control block's header carries the (distinct) control tag.
        let a = reg.swap(stack, hn_ord, hn_off, &["o"], ctrl_n);
        assert!(
            a.is_err(),
            "a control block was accepted as an #[bstack_owned] target"
        );

        let _ = hn.into_inner();
        drop(n); // release the strong ref via Drop
        std::fs::remove_file(&schema).ok();
        std::fs::remove_file(&data).ok();
    }
}
