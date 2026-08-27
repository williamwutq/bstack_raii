// A shared (`rc`) block has no `try_clone_in` — a shared handle is duplicated with
// `BStackRc::try_clone`, not deep-cloned.
use bstack_raii::{BStackOwnedSliceAllocator, TryCloneIn, bstack_block};
#[bstack_block(rc)]
struct S {
    v: u32,
}
fn f<A: BStackOwnedSliceAllocator>(s: &S, a: &A) {
    let _ = s.try_clone_in(a); // no such method on a shared block
}
fn main() {}
