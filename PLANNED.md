# Planned Features

This document outlines upcoming features planned for the `bstack_raii` crate. These enhancements aim to extend the ownership model, the `#[bstack_block]` / `#[bstack_enum]` / `#[bstack_class]` macros, and the RTTI interpreter while preserving the core guarantees the layer rests on: typed RAII ownership, per-method atomicity, crash-safety, and the null-niche allocator contract. Changes aim to be backward-compatible with the on-disk form and the public API. New capabilities are preferred as additive traits, attributes, or feature-gated modules rather than modifications to existing ones, to avoid breaking changes. All features aim to follow [Rust's API design guidelines](https://rust-lang.github.io/api-guidelines/), the crate's [design guidelines](GUIDELINES.md), and the underlying [`bstack`](https://github.com/williamwutq/bstack) design principles.

Two conventions carried from the format used here:

- **`NOT PLANNED`** records design directions that were considered and *rejected*, each with the reasoning — so a rejected idea is not re-proposed without new information.
- Each **planned** entry states its **feature gate** and whether it is a **breaking change** up front, then a **Motivation** / **Design** / **Open questions** body. "Design" is a sketch to be driven further, not a frozen contract.

---

