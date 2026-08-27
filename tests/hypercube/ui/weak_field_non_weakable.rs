// A `#[bstack_weak]` field whose target isn't weak-observable (`#[bstack_block(rc, weak)]`).
use bstack_raii::bstack_block;
#[bstack_block]
struct X {
    #[bstack_weak]
    f: u32,
}
fn main() {}
