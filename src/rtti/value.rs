//! The interpreters' **output vocabulary** — the structured values read out of, or
//! moved out of, a data file with no compiled-in Rust type.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * [`AnyRef`](super::AnyRef), the RTTI `&dyn Any` bridge back to compiled-in types.
//! * [`Value`](super::Value), the read interpreter's tree output.
//! * [`Moved`](super::Moved) / [`VecRef`](super::VecRef) /
//!   [`ForeignPtr`](super::ForeignPtr), the `move_out` transfer vocabulary.
//! * `Resolved`, what a field path resolves to (instance slot vs class variable).
