//! The **free interpreter** — the non-recursive teardown walk that reclaims a
//! structure's `owned` / `embed` / `strong` / `weak` / `ref` / `vec` / array / tuple /
//! option storage, refcount decrements and all. The RTTI analog of `io_core::teardown`.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * `TdOp`, the teardown work-stack step.
//! * `RttiRegistry::teardown`, `commit_strong_release`,
//!   `teardown_foreign` / `teardown_foreign_in` (the cross-file case).
