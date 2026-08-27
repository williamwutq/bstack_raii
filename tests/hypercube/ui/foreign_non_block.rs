// A `Foreign<T>` whose target `T` is not a bstack block (`Foreign<u32>`).
use bstack_raii::bstack_block;
#[bstack_block]
struct Holder {
    #[bstack_owned]
    link: bstack_raii::Foreign<u32>,
}
fn main() {}
