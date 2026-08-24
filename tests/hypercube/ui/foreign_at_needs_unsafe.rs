// `Foreign::at(&handle)` builds a `SELF` pointer from a bare offset that carries no file
// identity. A `SELF` pointer is sound only in its own file, so storing this into a block
// in a *different* file drives a wrong-file free/clone from otherwise-safe code
// (NEW-20260818-F1). `at` is therefore `unsafe`; safe code must not call it — the safe
// way to name a local block is `bstack_cast!(slice as Foreign<T>)` / `from_local`.
use bstack_raii::{Foreign, bstack_block};
#[bstack_block]
struct Leaf {
    v: u32,
}
fn make(leaf: &Leaf) -> Foreign<'static, Leaf> {
    Foreign::at(leaf)
}
fn main() {}
