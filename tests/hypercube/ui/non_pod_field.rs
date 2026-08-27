// An un-annotated field of a non-`Pod` type.
use bstack_raii::bstack_block;
struct NotPod(String);
#[bstack_block]
struct X {
    f: NotPod,
}
fn main() {}
