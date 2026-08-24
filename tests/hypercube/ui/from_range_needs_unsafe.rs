// `BStackBlock::from_range` is the raw handle constructor (bstack_raii's analogue of
// `BStackSlice::from_raw_parts`): it wraps an arbitrary range as an owning handle with
// no validation. From *safe* code that would let `from_range(bogus).bstack_drop(..)`
// free an unbacked or live-and-duplicated range — a use-after-free / double
// -free with no `unsafe` in sight. It must therefore be `unsafe fn`, so this fails.
use bstack_raii::{bstack_block, BStackBlock, BStackRange};
#[bstack_block]
struct Leaf {
    v: u32,
}
fn main() {
    let _ = <Leaf as BStackBlock>::from_range(BStackRange::new(0, 8));
}
