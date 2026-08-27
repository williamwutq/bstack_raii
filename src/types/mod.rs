//! The crate's **semantic types** — the vocabulary the layer is written against,
//! the way `bstack` centralises its `BStackRange` / `BStackSlice` / `BStackOwnedSlice`.
//!
//! Split by *what kind* of thing each is:
//!
//! * [`alloc`] — the allocator capability [`BStackRaiiAllocator`](alloc::BStackRaiiAllocator)
//!   and its object-safe cross-file projection ([`alloc::host`]) — the front-door
//!   bound the whole layer is generic over, kept at the top of `types` rather than
//!   among the block-level contracts.
//! * [`traits`] — the interfaces and contracts the block model is written against
//!   (the block / refcount / move / embed / cast / drop protocols, and the typed
//!   [`BStackRef`](traits::BStackRef)).
//! * [`compiled`] — the concrete, hand-written handle and on-disk types built *on*
//!   those contracts ([`BStackOwned`](compiled::owned::BStackOwned), the shared
//!   [`BStackRc`](compiled::rc::BStackRc) / [`BStackWeak`](compiled::rc::BStackWeak),
//!   the growable [`BStackVec`](compiled::vec::BStackVec), and the on-disk
//!   [`BlockHeader`](compiled::block::BlockHeader)).
//!
//! Distinct from [`crate::io_core`], which holds the *mechanism* (WAL, refcounts,
//! teardown, the file registry) these types drive.

pub mod alloc;
pub mod compiled;
pub mod traits;
