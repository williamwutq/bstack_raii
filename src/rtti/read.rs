//! The **read interpreter** — schema over a live data file → a [`Value`](super::Value)
//! tree, with no compiled-in types. The non-recursive counterpart of a typed block read.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * `Op`, the read machine's work-stack step.
//! * `RttiRegistry::read_value` / `run_read` (the machine), `read_ptr`,
//!   `any_ref`, `read_any`.
