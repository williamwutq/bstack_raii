//! The concrete, hand-written types built **on** the [`traits`](super::traits)
//! contracts — the compiled handle and on-disk shapes, as opposed to the
//! interfaces they implement.
//!
//! * [`block`] — the on-disk [`BlockHeader`](block::BlockHeader) prefixing every
//!   block, and the injected refcount / control field offsets.
//! * [`owned`] — [`BStackOwned`](owned::BStackOwned), the without-allocator,
//!   uniquely-owned block handle.
//! * [`rc`] — the with-allocator shared handles [`BStackRc`](rc::BStackRc) /
//!   [`BStackWeak`](rc::BStackWeak).
//! * [`vec`] — the persistent growable [`BStackVec`](vec::BStackVec) family.

pub mod block;
pub mod owned;
pub mod rc;
pub mod vec;
