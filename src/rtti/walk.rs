//! Shared **walk primitives** the interpreters all build on — the non-recursive
//! traversal helpers and constants used by `read` / `teardown` / `clone` / `move`
//! alike (which is why they live in their own module rather than any one interpreter).
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * `RttiRegistry::shape_stride` — the element/field stride every walk chains offsets by.
//! * Shape classification: `foreign_leaf`, `element_ref_tag`, `weak_element_tag`,
//!   `shape_has_reference`, `option_present`.
//! * Enum discriminant reads: `read_disc`, `disc_mask`.
//! * Recursion/allocation guards: `DepthGuard` (+ `MAX_RTTI_DEPTH`), `budget_exceeded`,
//!   `checked_vec_len`.
//! * Block/target checks: `verify_data_block`, `commit_weak_release`.
//! * Result-stack helpers `pop_n` / `pop_named`, the interp constants
//!   (`VECDESC_LEN`, `BYTEVEC_HEADER`, `FOREIGN_REPR_LEN`, `HEADER_TAG_OFFSET`,
//!   `CONTROL_SIZE`), and the shared error constructors.
