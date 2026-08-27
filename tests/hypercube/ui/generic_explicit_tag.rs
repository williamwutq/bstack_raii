// [BSTACK0005] an explicit 8-byte tag on a generic block collapses every
// instantiation to one tag (`mix` has no hash bytes left to perturb).
use bstack_raii::bstack_block;
#[bstack_block(tag = "EIGHTTAG")]
struct Pinned<T: bstack_raii::Pod> {
    #[bstack_mut]
    v: T,
}
fn main() {}
