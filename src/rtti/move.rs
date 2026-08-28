//! The **move interpreter** — the RTTI `bstack_move!`: disassemble a block into its
//! owned parts (a [`SmallStringMap`](crate::util::SmallStringMap)`<`[`Moved`](super::Moved)`>`),
//! freeing only the shell and handing each owned reference back to the caller.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * `RttiRegistry::move_out`, `move_fields`, `move_field`.
//!
//! (Declared as `mod r#move;` — `move` is a keyword — with this file named `move.rs`.)
