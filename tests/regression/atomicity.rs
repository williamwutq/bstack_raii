//! Read-modify-write atomicity.
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

mod atomicity {
    #![allow(clippy::module_inception)]
    //! Regression tests for read-modify-write atomicity of the pointer/vector slots.
    //! Each mutator below fuses the read of the displaced pointer and the write of the
    //! new one into one atomic `BStack::swap` (pointer slots) or a CAS-guarded grow
    //! (the vector), so only the caller that displaces a block reclaims it and two
    //! concurrent callers never both reclaim the same displaced block — a double free.
    //!
    //! The oracle is `DebugCheckingAllocator`, which panics in-line on a double free;
    //! if a test completes, no double free occurred. These are contention tests, so
    //! they loop enough to expose any non-atomic slot under load (seconds) — a pass is
    //! the absence of that panic.
    #![forbid(unsafe_code)]
    use std::sync::Arc;
    use std::thread;

    use bstack::{BStack, BStackAllocator, DebugCheckingAllocator, FirstFitBStackAllocator};
    use bstack_raii::{BStackDrop, BStackString, bstack_block};

    type Alloc = DebugCheckingAllocator<FirstFitBStackAllocator>;

    fn alloc(tag: &str) -> (Arc<Alloc>, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_atom_{tag}_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let inner = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
        (Arc::new(DebugCheckingAllocator::new(inner)), path)
    }

    #[bstack_block]
    struct Child {
        v: u32,
    }

    #[bstack_block]
    struct Holder {
        #[bstack_mut]
        #[bstack_owned]
        c: Child,
        #[bstack_mut]
        v: Vec<u64>,
    }

    /// Two threads calling `BStackString::set` in a loop: the atomic 16-byte swap makes
    /// each caller free the distinct bytes block it displaced, so the displaced block is
    /// never double-freed.
    #[test]
    fn concurrent_string_set_no_double_free() {
        let (a, path) = alloc("str");
        let s = BStackString::new(&*a, "initial").unwrap();

        thread::scope(|sc| {
            for t in 0..2 {
                let a = &a;
                let s = &s;
                sc.spawn(move || {
                    for i in 0..600u32 {
                        s.handle().set(&**a, &format!("t{t}-value-{i}")).unwrap();
                    }
                });
            }
        });

        // The winner's content is one of the threads' last writes; teardown is clean.
        s.bstack_drop(&*a).unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// Two threads calling the generated `replace_c` mutator: each installs a fresh
    /// child and frees the displaced one. The old read-then-write let both threads
    /// take the same displaced child.
    #[test]
    fn concurrent_replace_field_no_double_free() {
        let (a, path) = alloc("repl");
        let holder = Holder::new(&*a, Child::new(&*a, 0).unwrap(), &[]).unwrap();

        thread::scope(|sc| {
            for t in 0..2 {
                let a = &a;
                let holder = &holder;
                sc.spawn(move || {
                    for i in 0..350u32 {
                        let fresh = Child::new(&**a, t * 100_000 + i).unwrap();
                        let old = holder.handle().replace_c(a.stack(), fresh).unwrap();
                        old.bstack_drop(&**a).unwrap();
                    }
                });
            }
        });

        holder.bstack_drop(&*a).unwrap();
        std::fs::remove_file(&path).ok();
    }

    /// Two threads pushing onto the same field-resident vector, forcing repeated
    /// growth. The old grow path had both threads free the same displaced ring;
    /// the CAS-guarded grow makes the loser free its own new block and retry.
    #[test]
    fn concurrent_vec_push_no_double_free() {
        let (a, path) = alloc("vec");
        let holder = Holder::new(&*a, Child::new(&*a, 0).unwrap(), &[]).unwrap();

        thread::scope(|sc| {
            for t in 0..2 {
                let a = &a;
                let holder = &holder;
                sc.spawn(move || {
                    for i in 0..200u64 {
                        let mut v = holder.handle().get_v(&**a).unwrap();
                        v.push(t as u64 * 1_000_000 + i).unwrap();
                    }
                });
            }
        });

        holder.bstack_drop(&*a).unwrap();
        std::fs::remove_file(&path).ok();
    }
}
