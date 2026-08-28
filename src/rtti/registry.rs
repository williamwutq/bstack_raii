//! The in-memory **registry** and link-time registration — the RTTI analog of
//! `io_core::registry`: the scanned schema stack every lookup resolves against.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * [`RttiRegistration`](super::RttiRegistration) + the `linkme`
//!   [`RTTI_TYPES`](super::RTTI_TYPES) distributed slice, and [`sync`](super::sync).
//! * `RecordRef` and [`RttiRegistry`](super::RttiRegistry) with its core methods:
//!   `open` / `scan` / `index` / `append` / `sync_compiled`, `ordinal_of` / `tag_of` /
//!   `resolve_ptr` / `load_type`, and the class-variable accessors
//!   (`class_value` / `set_class_value` / `locate_class_value`).
