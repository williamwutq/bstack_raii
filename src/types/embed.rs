//! The `#[embed]` capability marker, [`BStackEmbeddable`].

use crate::types::block::BStackBlock;

/// Marker for a block that may be [`#[embed]`](macro@crate::bstack_block)ded: a
/// **self-contained** block whose entire state lives in its own `OnDisk` payload, with
/// no separate control block. Plain `#[bstack_block]` / `#[bstack_enum]` types and the
/// stdlib collections implement it; `#[bstack_block(rc)]` / `(rc, weak)` blocks do
/// **not**.
///
/// `#[embed]` folds the child's data inline and frees its shell. For a reference-counted
/// child that is corrupting: an `(rc, weak)` child's *separate* control block would be
/// left pointing at the now-freed data offset, and an `(rc)` child's shared-ownership
/// refcount is meaningless once the block is uniquely embedded. The derive requires an
/// `#[embed]` target to be `BStackEmbeddable`, turning that hazard into a compile error
/// instead of the incidental `BStackOwned` vs `BStackRc` type mismatch it used to be.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be `#[embed]`ed (not a plain, self-contained block)",
    label = "not embeddable",
    note = "`#[embed]` requires a plain `#[bstack_block]` / `#[bstack_enum]` (or a stdlib \
            collection). A reference-counted `#[bstack_block(rc)]` / `(rc, weak)` block carries a \
            refcount / separate control block that embedding would strand — hold it out-of-line \
            with `#[bstack_owned]` (or `#[bstack_strong]`) instead."
)]
pub trait BStackEmbeddable: BStackBlock {}
