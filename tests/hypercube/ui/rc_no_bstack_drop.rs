// `BStackRc` release is `drop(rc)`, not `bstack_drop` (which no longer resolves
// through `Deref` to a raw handle free).
use bstack_raii::{BStackDrop, BStackOwnedSliceAllocator, bstack_block};
#[bstack_block(rc)]
struct S {
    v: u32,
}
fn f<A: BStackOwnedSliceAllocator>(rc: bstack_raii::BStackRc<'_, S, A>, a: &A) {
    let _ = rc.bstack_drop(a); // gone: use `drop(rc)` for a refcount release
}
fn main() {}
