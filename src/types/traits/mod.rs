//! The **interfaces and contracts** the block model is generic over — traits, and
//! the lightweight typed pointer they all speak in terms of. (The allocator
//! capability itself lives one level up, in [`crate::types::alloc`].)
//!
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
//! * [`reference`] — the typed, non-owning [`BStackRef`](reference::BStackRef) over a
//!   [`bstack::BStackRange`].

pub mod block;
pub mod cast;
pub mod drop;
pub mod embed;
pub mod r#move;
pub mod rc;
pub mod reference;
