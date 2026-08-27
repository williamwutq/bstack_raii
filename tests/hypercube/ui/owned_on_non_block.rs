// An ownership annotation (`#[bstack_owned]`) targeting a non-block type — even one
// that would be fine as inline POD without the annotation.
use bstack_raii::bstack_block;
#[bstack_block]
struct X {
    #[bstack_owned]
    f: core::num::Wrapping<u32>,
}
fn main() {}
