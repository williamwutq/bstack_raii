//! The parsed, on-disk **schema vocabulary** and its little-endian wire codec — the
//! RTTI analog of the crate's `types` layer plus its serialization.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * [`Shape`](super::Shape) and its `encode`/`decode` (+ `MAX_SHAPE_DEPTH`, `shape_tag`).
//! * [`RttiField`](super::RttiField) / [`RttiVariant`](super::RttiVariant) /
//!   [`RttiBody`](super::RttiBody) / [`RttiEnum`](super::RttiEnum) /
//!   [`RttiType`](super::RttiType) and their wire encode/decode.
//! * The record framing (`encode_type` / `decode_type` / `frame_record`, `FLAG_*`).
//! * The typed reads over the generic [`Reader`](crate::util::Reader) cursor
//!   (`need` / `u8`..`u64` / `i64` / `eightcc` / `string` / `align`).
//! * `layouts_match`, `class_value_slot` / `class_value_within_shape`.
