// #[default] on a non-unit variant is rejected: the generated `Default for <Enum>Data`
// takes no allocator, so it cannot build a variant carrying a payload.
use bstack_raii::bstack_enum;
#[bstack_enum]
enum E {
    #[default]
    Num(u32),
    Nothing,
}
fn main() {}
