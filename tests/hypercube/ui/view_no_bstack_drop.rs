// A bare block handle is a non-owning *view* (`Copy`, no `BStackDrop`), so it cannot
// be freed — freeing requires an affine `BStackOwned` / `*Ref` token.
use bstack_raii::{BStackDrop, BStackOwnedSliceAllocator, bstack_block};
#[bstack_block]
struct Cell {
    v: u32,
}
fn f<A: BStackOwnedSliceAllocator>(owned: bstack_raii::BStackOwned<Cell>, a: &A) {
    let view: Cell = owned.into_inner(); // a Copy view
    let _ = view.bstack_drop(a); // no `bstack_drop` on a bare handle
}
fn main() {}
