//! The **deep-clone interpreter** — the non-recursive clone walk: `owned` sub-structure
//! deep-copied, shared (`strong` / `weak`) targets refcount-bumped, WAL-integrated for
//! crash-safe reclamation. The RTTI analog of `io_core::clone`.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * `CloneOp` (work-stack step) and `CloneState` (the accumulating clone state).
//! * `RttiRegistry::clone_value` / `clone_build`, `alloc_copy`, `wal_log_alloc`,
//!   `strong_bump_off`, `ensure_type`, and the cross-file
//!   `clone_foreign` / `clone_foreign_in`.
