//! Cross-file `Foreign` findings.
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

mod brand {
//! Regression: the `'a` brand on `Foreign<'a, T>` does not confine a `SELF`
//! pointer to its own file. Safe code moves one from F1 into a block in F2, and
//! F2's teardown then frees an unrelated, live F2 block.
use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::registry::FileId;
use bstack_raii::{BStackBlock, BStackDrop, BStackOwned, Foreign, ForeignOwned, bstack_block};

#[bstack_block]
struct Leaf {
    val: u32,
}

#[bstack_block]
struct Holder {
    #[bstack_mut]
    #[bstack_owned]
    link: Foreign<Leaf>,
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "bstack_raii_brand_{tag}_{}.bstack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn self_leaf<A: bstack_raii::BStackRaiiAllocator>(a: &A, v: u32) -> u64 {
    let l = Leaf::new(a, v).unwrap();
    let off = l.handle().range().start();
    let _ = l.into_inner(); // relinquish: the Foreign field owns it now
    off
}

#[test]
fn self_foreign_crosses_files_and_frees_a_live_stranger() {
    let p1 = temp_path("f1");
    let p2 = temp_path("f2");
    let f1 = FirstFitBStackAllocator::new(BStack::open(&p1).unwrap()).unwrap();
    let f2 = FirstFitBStackAllocator::new(BStack::open(&p2).unwrap()).unwrap();

    // --- F1: leaf A, owned by h1's SELF `Foreign` field. -------------------
    let a_off = self_leaf(&f1, 111);
    let spare = self_leaf(&f1, 999);
    let h1 = Holder::new(&f1, unsafe { Foreign::<Leaf>::new(FileId::SELF, a_off) }).unwrap();

    // --- F2: `victim` is a live, ordinary block nobody shares. --------------
    // Same allocation sequence, so it lands at the same offset A did in F1.
    let victim: BStackOwned<Leaf> = Leaf::new(&f2, 777).unwrap();
    let victim_off = victim.handle().range().start();
    let b_off = self_leaf(&f2, 333);
    let h2 = Holder::new(&f2, unsafe { Foreign::<Leaf>::new(FileId::SELF, b_off) }).unwrap();
    assert_eq!(
        a_off, victim_off,
        "fixture: F1's A and F2's victim must collide for the demo"
    );

    // --- Move F1's SELF-owned target out (safe). ----------------------------
    let replacement =
        unsafe { ForeignOwned::from_foreign(Foreign::<Leaf>::new(FileId::SELF, spare)) };
    let escapee: ForeignOwned<'_, Leaf> = h1.handle().replace_link(&f1, replacement).unwrap();
    assert!(escapee.is_self());
    assert_eq!(escapee.as_foreign().offset(), a_off);

    // --- Store it into a block in the OTHER file (safe; no `unsafe` here). --
    // The brand is supposed to make this impossible. It compiles.
    let displaced = h2.handle().replace_link(&f2, escapee).unwrap();
    assert_eq!(h2.handle().get_link(f2.stack()).unwrap().offset(), a_off);

    // --- Consequences, all from safe code. ----------------------------------
    // 1. F2's real target B is stranded: `displaced` is the only handle left.
    displaced.bstack_drop(&f2).unwrap();
    // 2. h2's teardown resolves SELF against F2 and frees `victim`'s block.
    h2.bstack_drop(&f2).unwrap();
    // 3. `victim` is still live as far as its owner is concerned...
    println!(
        "victim reads back as {:?} after being freed out from under it",
        victim.handle().get_val(f2.stack()).unwrap()
    );
    // ...and dropping it is now a double free.
    let err = victim.bstack_drop(&f2).unwrap_err();
    println!("victim.bstack_drop -> {err}");

    // 4. F1's leaf A is leaked: nothing in F1 owns it any more.
    h1.bstack_drop(&f1).unwrap();

    std::fs::remove_file(&p1).ok();
    std::fs::remove_file(&p2).ok();
}
}

mod intolocal {
//! Regression: `ForeignOwned::into_local()` takes the target
//! allocator (like its strong / weak siblings) and rejects an explicit-`FileId`
//! pointer whose home file is not that allocator's — so a wrong-file free is caught,
//! not performed.
use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::registry;
use bstack_raii::{BStackBlock, BStackDrop, BStackOwned, Foreign, ForeignOwned, bstack_block};

#[bstack_block]
struct Leaf {
    v: u32,
}

#[bstack_block]
struct Holder {
    #[bstack_mut]
    #[bstack_owned]
    link: Foreign<Leaf>,
}

fn tmp(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "bstack_raii_itl_{tag}_{}.bstack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn foreign_owned_into_local_frees_in_the_wrong_file() {
    let pr = tmp("reg");
    let pa = tmp("home");
    let pb = tmp("far");
    registry::init(&pr).unwrap();

    // --- File B (the foreign file): a leaf that the Holder in A will own. --------
    let leaf_off;
    let fid;
    {
        let b = FirstFitBStackAllocator::new(BStack::open(&pb).unwrap()).unwrap();
        let l = Leaf::new(&b, 555).unwrap();
        leaf_off = l.handle().range().start();
        let _ = l.into_inner(); // the foreign field owns it now
        fid = registry::attach(&pb, b).unwrap();
    }
    println!(
        "file B attached as {:?}; its leaf is at {leaf_off}",
        fid.as_u64()
    );

    // --- File A (home): a live victim block, then the Holder. --------------------
    let home = FirstFitBStackAllocator::new(BStack::open(&pa).unwrap()).unwrap();
    let victim: BStackOwned<Leaf> = Leaf::new(&home, 777).unwrap();
    assert_eq!(
        victim.handle().range().start(),
        leaf_off,
        "fixture: same alloc sequence, so A's victim collides with B's leaf offset"
    );
    // The `unsafe` here is honest: `leaf_off` really does name a live `Leaf` in `fid`.
    let holder = Holder::new(&home, unsafe { Foreign::<Leaf>::new(fid, leaf_off) }).unwrap();

    // --- Everything from here is safe. ------------------------------------------
    let spare = Leaf::new(&home, 1).unwrap();
    let spare_off = spare.handle().range().start();
    let _ = spare.into_inner();
    let replacement = unsafe {
        ForeignOwned::from_foreign(Foreign::<Leaf>::new(registry::FileId::SELF, spare_off))
    };
    let fo: ForeignOwned<'_, Leaf> = holder.handle().replace_link(&home, replacement).unwrap();
    println!(
        "ForeignOwned names file {:?} offset {}",
        fo.as_foreign().file_id().as_u64(),
        fo.as_foreign().offset()
    );

    // The pointer names file B; resolving it against the HOME allocator must be
    // rejected rather than handing back a handle that would free A's block.
    match fo.into_local(&home) {
        Ok(_) => panic!("into_local accepted a mismatched target file"),
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput),
    }

    // File A's victim is untouched and still readable / freeable.
    assert_eq!(victim.handle().get_v(home.stack()).unwrap(), 777);
    victim.bstack_drop(&home).unwrap();

    holder.bstack_drop(&home).unwrap();
    registry::detach(fid);
    for p in [&pr, &pa, &pb] {
        std::fs::remove_file(p).ok();
    }
}
}
