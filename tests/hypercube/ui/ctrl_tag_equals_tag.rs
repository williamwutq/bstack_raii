// [BSTACK0006] an (rc, weak) block whose explicit `ctrl_tag` equals its data `tag`
// collapses the data/control distinction.
use bstack_raii::bstack_block;
#[bstack_block(tag = "DUP", ctrl_tag = "DUP", rc, weak)]
struct Dup {
    v: u32,
}
fn main() {}
