//! The `bstack_move!` contracts: [`BStackMove`] (the per-block field destructure)
//! and [`BStackMoveExpr`] (the wrapper-dispatched entry point).

use std::io;

use super::super::compiled::BStackOwned;
use super::BStackBlock;
use crate::BStackRaiiAllocator;

/// The per-block field destructure behind `bstack_move!`: read every field, then
/// free only the parent *shell* (the children stay live on disk). Ownership is
/// transferred out — owned children as `BStackOwned`, strong as `BStackRc`, weak
/// as `Option<BStackWeak>`, refs as `BStackRef`, POD by value.
///
/// Generated on the block type `X` (a local type downstream, so it satisfies the
/// orphan rule) for **all** modes. It is the shared core used by both the owned
/// and the `Rc` `bstack_move!` paths in [`BStackMoveExpr`]; it does not touch
/// refcounts or the control block, so the caller must have already established
/// that the shell may be freed.
///
/// Takes the (bare, allocator-less) [`BStackOwned`] handle plus an explicit
/// allocator, since neither the owned handle nor the block type carries one.
pub trait BStackMove: BStackBlock {
    /// The tuple of field handles produced, in field-declaration order.
    type Fields<'a, A: BStackRaiiAllocator>;
    fn bstack_move<'a, A: BStackRaiiAllocator>(
        owned: BStackOwned<Self>,
        allocator: &'a A,
    ) -> io::Result<Self::Fields<'a, A>>;
}

/// The `bstack_move!` entry point, dispatched on the wrapper handle type.
///
/// * `BStackOwned<X>` — infallible; `Output = io::Result<X::Fields>`.
/// * `BStackRc<X>` — a `try_unwrap`: `Output = io::Result<Result<X::Fields, Self>>`,
///   succeeding only when this handle is the **sole strong owner** (else the
///   handle is returned in `Err`).
///
/// The macro emits `BStackMoveExpr::bstack_move(expr)` and lets inference select
/// the impl from the argument's type.
pub trait BStackMoveExpr {
    type Output;
    fn bstack_move(self) -> Self::Output;
}
