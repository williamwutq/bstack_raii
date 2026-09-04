//! Static analysis of the [`Shape`] type — the on-disk **layout** and
//! **classification** helpers the interpreters share, with no traversal of their own.
//!
//! Where [`walk`](super::walk) holds the *mechanics* of a walk (guards, the result
//! stack, block/offset checks), this module answers questions *about a `Shape`*: most
//! are inherent [`Shape`] methods ([`has_reference`](Shape::has_reference),
//! [`foreign_leaf`](Shape::foreign_leaf), [`option_present`](Shape::option_present), …);
//! [`shape_stride`](RttiRegistry::shape_stride) is on [`RttiRegistry`] because resolving
//! an `Embed`'s width needs the registry. Every interpreter
//! (`read` / `teardown` / `clone` / `field` / `move`) builds on these.

use std::collections::HashMap;

use bstack::BStack;

use crate::primitives::{EightCC, OwnershipKind, WidePtr};
use crate::util::read_u64;

use super::RttiResult;
use super::{
    FOREIGN_REPR_LEN, RttiOrdinal, RttiRegistry, RttiType, Shape, VECDESC_LEN, add_off, mul_off,
    unknown_tag,
};

impl Shape {
    /// Whether this shape contains any block reference anywhere (so it is not pure POD).
    pub(in crate::rtti) fn has_reference(&self) -> bool {
        match self {
            Shape::Pod { .. } | Shape::Class { .. } => false,
            Shape::Owned(_)
            | Shape::Strong(_)
            | Shape::Weak(_)
            | Shape::Ref(_)
            | Shape::Embed(_)
            | Shape::Foreign { .. } => true,
            Shape::Option(inner) | Shape::Vec(inner) | Shape::Array { inner, .. } => {
                inner.has_reference()
            }
            Shape::Tuple(items) => items.iter().any(Shape::has_reference),
        }
    }

    /// The element tag of a reference-array element (`owned` / `strong` / `weak` / `ref`,
    /// optionally `Option`-wrapped) — its slot is a single `u64` offset. `None` for an
    /// element the move interpreter can't hand out one-per-`u64` (embed / foreign /
    /// nested).
    pub(in crate::rtti) fn element_ref_tag(&self) -> Option<EightCC> {
        match self {
            Shape::Owned(t) | Shape::Strong(t) | Shape::Weak(t) | Shape::Ref(t) => Some(*t),
            Shape::Option(inner) => inner.element_ref_tag(),
            _ => None,
        }
    }

    /// The tag of a **weak** reference leaf (optionally `Option`-wrapped) — its slot holds
    /// a `u64` *control-block* offset, not a data offset. `None` for any non-weak shape.
    /// Lets `move_out` hand a weak array back as a [`Moved::WeakList`] distinct from a
    /// data-ref [`Moved::List`].
    pub(in crate::rtti) fn weak_element_tag(&self) -> Option<EightCC> {
        match self {
            Shape::Weak(t) => Some(*t),
            Shape::Option(inner) => inner.weak_element_tag(),
            _ => None,
        }
    }

    /// The `(tag, kind)` of a cross-file `Foreign` leaf (optionally `Option`-wrapped) —
    /// its slot is a 16-byte [`WidePtr`]. `None` for any non-foreign shape. Used to
    /// drive the per-element foreign path in a `Vec` / array / tuple.
    pub(in crate::rtti) fn foreign_leaf(&self) -> Option<(EightCC, OwnershipKind)> {
        match self {
            Shape::Foreign { tag, kind } => Some((*tag, *kind)),
            Shape::Option(inner) => inner.foreign_leaf(),
            _ => None,
        }
    }

    /// Whether an `Option<self>` slot at `base` is `Some`. The null niche's **location
    /// depends on the inner shape**: a `Foreign` slot is a 16-byte `WidePtr`
    /// `{ file_id:u32 @0, type_index:u32 @4, offset:u64 @8 }` whose niche is the target
    /// `offset` word at byte 8 — *not* the leading `file_id|type_index` word (which is
    /// `0` for a present untyped SELF-file pointer, so testing it would misread a live
    /// pointer as `None`). Every other offset-bearing inner — a block reference (`owned` /
    /// `strong` / `weak` / `ref`) or a `Vec` descriptor (`data_off`) — uses the leading
    /// `u64`.
    pub(in crate::rtti) fn option_present(&self, data: &BStack, base: u64) -> RttiResult<bool> {
        Ok(match self {
            Shape::Foreign { .. } => !WidePtr::read_from_stack(data, base)?.is_null(),
            _ => read_u64(data, base)? != 0,
        })
    }

    /// Strip any leading `Option` wrapper(s), returning the inner leaf shape. An
    /// `Option<owned/strong/weak/foreign>` element occupies the **same** on-disk slot as
    /// the bare leaf (a nullable offset / `WidePtr`, `0`/null = `None`), so the
    /// destructive `Vec` walks in `teardown` / `clone` dispatch on the peeled leaf — an
    /// `Option`-wrapped owning element must be freed / deep-copied exactly as the bare
    /// one is, never fall through as inert POD (which would leak, or alias the source's
    /// children into the clone → double-free). Mirrors [`foreign_leaf`](Self::foreign_leaf)
    /// / [`element_ref_tag`](Self::element_ref_tag), which already peel `Option`.
    pub(in crate::rtti) fn peel_option(&self) -> &Shape {
        match self {
            Shape::Option(inner) => inner.peel_option(),
            other => other,
        }
    }
}

impl RttiRegistry {
    /// The on-disk byte width of one element of `shape` — the stride for array / vec /
    /// tuple element addressing. References are a `u64` offset; a foreign is a
    /// `WidePtr`; an embedded child is its whole block; a vector is its inline
    /// `VecDesc`.
    pub(in crate::rtti) fn shape_stride(
        &self,
        shape: &Shape,
        cache: &mut HashMap<RttiOrdinal, RttiType>,
    ) -> RttiResult<u64> {
        Ok(match shape {
            Shape::Pod { width } => *width as u64,
            Shape::Owned(_) | Shape::Strong(_) | Shape::Weak(_) | Shape::Ref(_) => 8,
            Shape::Foreign { .. } => FOREIGN_REPR_LEN,
            Shape::Vec(_) => VECDESC_LEN,
            Shape::Option(inner) => self.shape_stride(inner, cache)?,
            Shape::Array { n, inner } => mul_off(*n as u64, self.shape_stride(inner, cache)?)?,
            Shape::Tuple(items) => {
                let mut sum = 0u64;
                for it in items {
                    sum = add_off(sum, self.shape_stride(it, cache)?)?;
                }
                sum
            }
            Shape::Embed(tag) => {
                let ord = self.ordinal_of(*tag).ok_or_else(unknown_tag)?;
                if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(ord) {
                    e.insert(self.load_type(ord)?);
                }
                cache[&ord].ondisk_size
            }
            // A class variable is not part of the instance layout.
            Shape::Class { .. } => 0,
        })
    }
}
