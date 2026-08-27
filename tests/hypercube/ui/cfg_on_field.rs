// [BSTACK0004] `#[cfg]` on a block field is rejected (the generated layout,
// constructor, and accessors cannot be made conditional).
use bstack_raii::bstack_block;
#[bstack_block]
struct X {
    x: u64,
    #[cfg(feature = "nope")]
    gated: u64,
}
fn main() {}
