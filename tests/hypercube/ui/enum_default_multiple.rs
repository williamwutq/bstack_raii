// More than one #[default] variant is rejected: exactly one may be marked.
use bstack_raii::bstack_enum;
#[bstack_enum]
enum E {
    #[default]
    A,
    #[default]
    B,
}
fn main() {}
