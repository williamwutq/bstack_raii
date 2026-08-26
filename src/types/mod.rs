//! The crate's **semantic types** — the vocabulary the layer is written against,
//! the way `bstack` centralises its `BStackRange` / `BStackSlice` / `BStackOwnedSlice`.
//!
//! * [`alloc`] — the allocator capability [`BStackRaiiAllocator`](alloc::BStackRaiiAllocator)
//!   and its object-safe cross-file projection ([`alloc::host`]).
//! * [`drop`] — the RAII drop protocol ([`BStackDrop`](drop::BStackDrop),
//!   [`AutoDrop`](drop::AutoDrop), and the `BlockShell` it builds on).
//!
//! Distinct from [`crate::io_core`], which holds the *mechanism* (WAL, refcounts,
//! teardown, the file registry) these types drive.

pub mod alloc;
pub mod drop;
