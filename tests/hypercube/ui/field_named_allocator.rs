// A field named `allocator` collides with the generated constructor's fixed `allocator`
// parameter (`S::new(allocator, ..fields..)`). The macro rejects it up front with a clear
// [BSTACK0310] rather than leaving a confusing `E0415`/`E0599` to the compiler.
use bstack_raii::bstack_block;
#[bstack_block]
struct S {
    allocator: u64,
}
fn main() {}
