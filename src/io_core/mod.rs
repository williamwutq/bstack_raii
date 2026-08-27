//! The crate's low-level **on-disk I/O core**: atomic primitives and stateful
//! mechanism over `bstack` files that sit *below* the object model — as opposed to
//! [`crate::types`], which holds the semantic vocabulary these drive.
//!
//! * [`bulk`] — sequential / atomic-bulk fallbacks behind the allocator's
//!   `alloc_many` / `free_many`, and the [`FreeManyError`](bulk::FreeManyError)
//!   they hand back.
//! * [`refcount`] — atomic on-disk `u64` counter ops.
//! * [`clone`] — the two-phase [`TryCloneIn`](clone::TryCloneIn) deep-clone engine
//!   ([`ClonePlan`](clone::ClonePlan)), WAL-integrated for crash-safe reclamation.
//! * [`wal`] — the write-ahead log for atomic multi-slice transactions.
//! * [`teardown`] — the recursive block-teardown mechanism (`dealloc_range`, the
//!   WAL-integrated `wal_teardown`).
//! * [`registry`] — the process-wide path↔[`FileId`](registry::FileId) map
//!   resolving cross-file (`Foreign<T>`) pointers. Re-exported publicly as
//!   `crate::registry`.

pub(crate) mod bulk;
pub(crate) mod clone;
pub(crate) mod refcount;
pub mod registry;
pub(crate) mod teardown;
pub(crate) mod wal;

// Facade: this subsystem's surface, re-exported once at this module's root — the
// public items again at the crate root, the crate-internal mechanism for the rest
// of the crate. Callers import `crate::io_core::X`, not the submodule path.
pub use bulk::FreeManyError;
pub use clone::{ClonePlan, TryCloneIn};
pub use registry::ForeignHostAllocator;
pub use teardown::{dealloc_range, wal_teardown};
pub use wal::{STD_WAL_ANCHOR, finish};

// Crate-internal mechanism (not part of the public API).
pub(crate) use teardown::TeardownDepthGuard;
pub(crate) use wal::{WalStatus, WalTxn, commit_frees, commit_home_frees, wal_lock_for};
