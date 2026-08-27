// [BSTACK0003] an unrecognised `bstack_*` field attribute (here a typo) is a hard
// error, not a silently-dropped annotation.
use bstack_raii::bstack_block;
#[bstack_block]
struct X {
    #[bstack_mutt]
    y: u64,
}
fn main() {}
