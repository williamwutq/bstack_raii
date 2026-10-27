//! Read-modify-write atomicity.
//!
//! Consolidated from the former per-finding test binaries;
//! each finding is an isolated `mod` (fault-injection ones self-gate via their
//! own `#![cfg(feature = "fault-injection")]`).

/// `DebugCheckingAllocator` documents that concurrent use is unsupported without
/// external synchronization. This wrapper is that external synchronization. One
/// mutex is held across each allocator call that eliminates false overlap panics.
mod serialized {
    use std::io;
    use std::sync::Mutex;

    use bstack::{
        BStack, BStackAllocError, BStackAllocator, BStackOwnedSlice, DebugCheckingAllocator,
        FirstFitBStackAllocator,
    };
    use bstack_raii::{BStackRaiiAllocator, NonNullOffset, STD_WAL_ANCHOR};

    pub struct Serialized {
        inner: DebugCheckingAllocator<FirstFitBStackAllocator>,
        lock: Mutex<()>,
    }

    impl Serialized {
        pub fn new(inner: DebugCheckingAllocator<FirstFitBStackAllocator>) -> Self {
            Serialized {
                inner,
                lock: Mutex::new(()),
            }
        }
    }

    impl BStackAllocator for Serialized {
        type Error = io::Error;
        type Allocated<'a> = BStackOwnedSlice<'a, Self>;

        fn stack(&self) -> &BStack {
            self.inner.stack()
        }

        fn into_stack(self) -> BStack {
            self.inner.into_stack()
        }

        fn alloc(&self, len: u64) -> io::Result<BStackOwnedSlice<'_, Self>> {
            let _g = self.lock.lock().unwrap();
            let r = self.inner.alloc(len)?.as_range();
            // SAFETY: `r` is a fresh live allocation from `inner`
            Ok(unsafe { BStackOwnedSlice::from_raw_range(self, r) })
        }

        fn realloc<'a>(
            &'a self,
            handle: BStackOwnedSlice<'a, Self>,
            new_len: u64,
        ) -> Result<BStackOwnedSlice<'a, Self>, BStackAllocError<'a, Self>> {
            let _g = self.lock.lock().unwrap();
            // SAFETY: the live allocation `handle` named owned by `inner`
            let inner_h =
                unsafe { BStackOwnedSlice::from_raw_range(&self.inner, handle.as_range()) };
            match self.inner.realloc(inner_h, new_len) {
                Ok(s) => Ok(unsafe { BStackOwnedSlice::from_raw_range(self, s.as_range()) }),
                Err(e) => Err(match e.handle {
                    Some(h) => BStackAllocError::with_handle(e.source, unsafe {
                        BStackOwnedSlice::from_raw_range(self, h.as_range())
                    }),
                    None => BStackAllocError::lost(e.source),
                }),
            }
        }

        fn dealloc<'a>(
            &'a self,
            handle: BStackOwnedSlice<'a, Self>,
        ) -> Result<(), BStackAllocError<'a, Self>> {
            let _g = self.lock.lock().unwrap();
            // SAFETY: the live allocation `handle` named owned by `inner`
            let inner_h =
                unsafe { BStackOwnedSlice::from_raw_range(&self.inner, handle.as_range()) };
            self.inner.dealloc(inner_h).map_err(|e| match e.handle {
                Some(h) => BStackAllocError::with_handle(e.source, unsafe {
                    BStackOwnedSlice::from_raw_range(self, h.as_range())
                }),
                None => BStackAllocError::lost(e.source),
            })
        }
    }

    // SAFETY: forwards every allocation to `FirstFitBStackAllocator` through the
    // debug checking allocator, which upholds the null niche at payload offset 0 and
    // reserves the `[8, 16)` WAL anchor slot; this wrapper only serializes the calls.
    unsafe impl BStackRaiiAllocator for Serialized {
        fn wal_anchor(&self) -> Option<NonNullOffset> {
            Some(STD_WAL_ANCHOR)
        }
    }
}

mod atomicity {
    #![allow(clippy::module_inception)]
    //! Regression tests for read-modify-write atomicity of the pointer/vector slots.
    //! Each mutator below fuses the read of the displaced pointer and the write of the
    //! new one into one atomic `BStack::swap` (pointer slots) or a CAS-guarded grow
    //! (the vector), so only the caller that displaces a block reclaims it and two
    //! concurrent callers never both reclaim the same displaced block — a double free.
    //!
    //! The oracle is `DebugCheckingAllocator`, which panics in-line on a double free;
    //! if a test completes, no double free occurred. It is wrapped in
    //! `super::serialized::Serialized`, the external synchronization its docs require
    //! for concurrent use. These are contention tests, so
    //! they loop enough to expose any non-atomic slot under load (seconds) — a pass is
    //! the absence of that panic.
    #![forbid(unsafe_code)]
    use std::sync::Arc;
    use std::thread;

    use bstack::{BStack, BStackAllocator, DebugCheckingAllocator, FirstFitBStackAllocator};
    use bstack_raii::{BStackDrop, BStackString, bstack_block};

    use super::serialized::Serialized;

    type Alloc = Serialized;

    fn alloc(tag: &str) -> (Arc<Alloc>, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "bstack_raii_atom_{tag}_{}.bstack",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let inner = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
        (Arc::new(Serialized::new(DebugCheckingAllocator::new(inner))), path)
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
