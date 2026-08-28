//! Path-addressed **field access** — reach one field by a `["outer", "inner", …]`
//! path and read, overwrite, or exchange it in place.
//!
//! Planned contents (to be moved here from [`super`], not yet split out):
//! * `RttiRegistry::resolve_field` (path → instance slot or class variable).
//! * `get` (read a field), `set` (overwrite a POD / `ref` leaf),
//!   `swap` / `swap_foreign` (exchange an owning reference, eightcc-checked).
