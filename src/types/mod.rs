//! The crate's **semantic types** — the vocabulary the layer is written against,
//! the way `bstack` centralises its `BStackRange` / `BStackSlice` / `BStackOwnedSlice`.
//!
//! * [`alloc`] — the allocator capability [`BStackRaiiAllocator`](alloc::BStackRaiiAllocator)
//!   and its object-safe cross-file projection ([`alloc::host`]).
//! * [`block`] — the core block contracts [`BStackCast`](block::BStackCast) /
//!   [`BStackBlock`](block::BStackBlock).
//! * [`cast`] — typed ↔ untyped handle conversion behind `bstack_cast!`
//!   ([`BStackCastInto`](cast::BStackCastInto) / [`BStackCastAs`](cast::BStackCastAs),
//!   [`CastError`](cast::CastError)).
//! * [`r#move`] — the `bstack_move!` contracts ([`BStackMove`](r#move::BStackMove),
//!   [`BStackMoveExpr`](r#move::BStackMoveExpr)).
//! * [`rc`] — the refcount capabilities ([`BStackShared`](rc::BStackShared),
//!   [`BStackWeakable`](rc::BStackWeakable)).
//! * [`embed`] — the `#[embed]` marker [`BStackEmbeddable`](embed::BStackEmbeddable).
//! * [`drop`] — the RAII drop protocol ([`BStackDrop`](drop::BStackDrop),
//!   [`AutoDrop`](drop::AutoDrop), and the `BlockShell` it builds on).
//!
//! Distinct from [`crate::io_core`], which holds the *mechanism* (WAL, refcounts,
//! teardown, the file registry) these types drive.

pub mod alloc;
pub mod block;
pub mod cast;
pub mod drop;
pub mod embed;
pub mod r#move;
pub mod rc;
