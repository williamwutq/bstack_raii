//! Shared **walk primitives** the interpreters all build on — the non-recursive
//! traversal helpers, error constructors, recursion/allocation guards, and constants
//! used by `read` / `teardown` / `clone` / `field` / `move` alike (which is why they
//! live here rather than in any one interpreter).

use std::collections::HashMap;
use std::io;

use bstack::{BStack, BStackRange};

use crate::BStackRaiiAllocator;
use crate::io_core::refcount;
use crate::primitives::{EightCC, NonNullOffset, OwnershipKind, WidePtr};
use crate::types::compiled::rc::CTRL_WEAK_OFFSET;
use crate::util::{io_error, io_errorfn, read_u64};

use super::{
    AnyRef, BYTEVEC_HEADER, CONTROL_SIZE, FOREIGN_REPR_LEN, RttiOrdinal, RttiRegistry, RttiType,
    Shape, VECDESC_LEN, Value, add_off, mul_off, unknown_tag,
};

impl RttiRegistry {
    /// The on-disk byte width of one element of `shape` — the stride for array / vec /
    /// tuple element addressing. References are a `u64` offset; a foreign is a
    /// `WidePtr`; an embedded child is its whole block; a vector is its inline
    /// `VecDesc`.
    pub(in crate::rtti) fn shape_stride(
        &self,
        shape: &Shape,
        cache: &mut HashMap<RttiOrdinal, RttiType>,
    ) -> io::Result<u64> {
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

io_errorfn!(
    pub(in crate::rtti) set_error(msg: impl std::fmt::Display),
    InvalidInput,
    "[BSTACK080D] RTTI set: {}",
    msg
);

io_errorfn!(
    pub(in crate::rtti) swap_error(msg: impl std::fmt::Display),
    InvalidInput,
    "[BSTACK0810] RTTI swap: {}",
    msg
);

/// Verify a **live block of type `tag`** sits at `off` in `data` (`off == 0` is the null
/// sentinel, allowed). The safe RTTI mutators install caller-supplied offsets into
/// owning slots; without this check a fabricated [`AnyRef`] / [`ForeignPtr`] could point
/// a slot at an arbitrary location that a later teardown would free (recursively, for
/// `owned`) or a later path would descend into — the same hazard `Foreign::new` /
/// `raw_<field>_slice` are `unsafe` for, but here checkable against the on-disk header.
pub(in crate::rtti) fn verify_data_block(data: &BStack, off: u64, tag: EightCC) -> io::Result<()> {
    if off == 0 {
        return Ok(());
    }
    // Error for a mutator (`set` / `swap` / `swap_foreign`) whose caller-supplied
    // target offset does not name a live block of the field's declared type.
    let bad_target = |found: Option<EightCC>| {
        let found = match found {
            Some(t) => format!("found {t:?}"),
            None => "out of bounds or unreadable".to_string(),
        };
        io_error!(
            InvalidInput,
            "[BSTACK0815] RTTI mutator: offset {} does not hold a live {:?} block ({})",
            off,
            tag,
            found
        )
    };
    match AnyRef::from_block(data, off) {
        Ok(a) if a.tag() == tag => Ok(()),
        Ok(a) => Err(bad_target(Some(a.tag()))),
        Err(_) => Err(bad_target(None)),
    }
}

/// Whether `shape` contains any block reference anywhere (so it is not pure POD).
pub(in crate::rtti) fn shape_has_reference(shape: &Shape) -> bool {
    match shape {
        Shape::Pod { .. } | Shape::Class { .. } => false,
        Shape::Owned(_)
        | Shape::Strong(_)
        | Shape::Weak(_)
        | Shape::Ref(_)
        | Shape::Embed(_)
        | Shape::Foreign { .. } => true,
        Shape::Option(inner) | Shape::Vec(inner) | Shape::Array { inner, .. } => {
            shape_has_reference(inner)
        }
        Shape::Tuple(items) => items.iter().any(shape_has_reference),
    }
}

/// The element tag of a reference-array element (`owned` / `strong` / `weak` / `ref`,
/// optionally `Option`-wrapped) — its slot is a single `u64` offset. `None` for an
/// element the move interpreter can't hand out one-per-`u64` (embed / foreign / nested).
pub(in crate::rtti) fn element_ref_tag(shape: &Shape) -> Option<EightCC> {
    match shape {
        Shape::Owned(t) | Shape::Strong(t) | Shape::Weak(t) | Shape::Ref(t) => Some(*t),
        Shape::Option(inner) => element_ref_tag(inner),
        _ => None,
    }
}

/// The tag of a **weak** reference leaf (optionally `Option`-wrapped) — its slot holds a
/// `u64` *control-block* offset, not a data offset. `None` for any non-weak shape. Lets
/// `move_out` hand a weak array back as a [`Moved::WeakList`] distinct from a data-ref
/// [`Moved::List`].
pub(in crate::rtti) fn weak_element_tag(shape: &Shape) -> Option<EightCC> {
    match shape {
        Shape::Weak(t) => Some(*t),
        Shape::Option(inner) => weak_element_tag(inner),
        _ => None,
    }
}

/// The `(tag, kind)` of a cross-file `Foreign` leaf (optionally `Option`-wrapped) —
/// its slot is a 16-byte [`WidePtr`]. `None` for any non-foreign shape. Used to
/// drive the per-element foreign path in a `Vec` / array / tuple.
pub(in crate::rtti) fn foreign_leaf(shape: &Shape) -> Option<(EightCC, OwnershipKind)> {
    match shape {
        Shape::Foreign { tag, kind } => Some((*tag, *kind)),
        Shape::Option(inner) => foreign_leaf(inner),
        _ => None,
    }
}

/// Whether an `Option<inner>` slot at `base` is `Some`. The null niche's **location
/// depends on the inner shape**: a `Foreign` slot is a 16-byte `WidePtr`
/// `{ file_id:u32 @0, type_index:u32 @4, offset:u64 @8 }` whose niche is the target
/// `offset` word at byte 8 — *not* the leading `file_id|type_index` word (which is
/// `0` for a present untyped SELF-file pointer, so testing it would misread a live
/// pointer as `None`). Every other offset-bearing inner — a block reference (`owned` /
/// `strong` / `weak` / `ref`) or a `Vec` descriptor (`data_off`) — uses the leading
/// `u64`.
pub(in crate::rtti) fn option_present(data: &BStack, inner: &Shape, base: u64) -> io::Result<bool> {
    Ok(match inner {
        Shape::Foreign { .. } => !WidePtr::read_from_stack(data, base)?.is_null(),
        _ => read_u64(data, base)? != 0,
    })
}

thread_local! {
    /// The current cross-file RTTI recursion depth. The interpreter is non-recursive
    /// *within* a file (a work-list), but `teardown` / `clone_value` recurse **natively**
    /// at each `Foreign` hop (through `teardown_foreign_in` / `clone_foreign_in`), each
    /// starting a fresh per-node budget — so the in-file cycle guard can't see across
    /// files. A foreign cycle (`A --owns--> B --owns--> A`, or a SELF back-edge) would
    /// drive unbounded native recursion → stack-overflow abort. [`DepthGuard`] bounds it.
    static RTTI_DEPTH: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// Native cross-file recursion cap. Each hop costs a few KB of native stack (a per-file
/// teardown/clone frame plus the two foreign helpers), so this stays well under the
/// smallest default thread stack (≈2 MiB) with wide margin — yet is far deeper than any
/// sane cross-file `Foreign` chain (the *in-file* walk is non-recursive and unbounded).
const MAX_RTTI_DEPTH: u32 = 100;

/// A scope guard bounding cross-file RTTI recursion (see [`RTTI_DEPTH`]). Created at the
/// top of `teardown` / `clone_value`; increments the depth, decrements on drop (so an
/// error/panic unwinds it cleanly), and refuses to enter past [`MAX_RTTI_DEPTH`].
pub(in crate::rtti) struct DepthGuard;

impl DepthGuard {
    pub(in crate::rtti) fn enter() -> io::Result<Self> {
        let depth = RTTI_DEPTH.with(|c| {
            let n = c.get() + 1;
            c.set(n);
            n
        });
        if depth > MAX_RTTI_DEPTH {
            RTTI_DEPTH.with(|c| c.set(c.get() - 1)); // undo: no guard is returned
            return Err(io_error!(
                "[BSTACK0807] RTTI cross-file recursion too deep (a foreign cycle?)",
            ));
        }
        Ok(DepthGuard)
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        RTTI_DEPTH.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// The element count of a vec data block, **validated against the block's own size**.
///
/// The `@0` word is the byte length; a corrupt/forged value must never drive a read or
/// free past the block. `byte_len` must fit the block's element region
/// (`data_size - header`) — otherwise the walk would materialize a petabyte-sized
/// allocation (abort) or, in teardown, read `u64`s from neighboring **live** blocks and
/// free ranges over them. The bound also keeps `base + i*stride` from wrapping. Returns
/// `byte_len / stride`.
pub(in crate::rtti) fn checked_vec_len(
    byte_len: u64,
    data_size: u64,
    stride: u64,
) -> io::Result<u64> {
    let usable = data_size.saturating_sub(BYTEVEC_HEADER);
    if byte_len > usable {
        return Err(io_error!(
            "[BSTACK0813] RTTI vector length ({} bytes) exceeds its data block \
             ({} usable bytes) — corrupt length word",
            byte_len,
            usable
        ));
    }
    Ok(byte_len.checked_div(stride).unwrap_or(0))
}

/// Release one `weak` reference whose control block is at `ctrl_off`: decrement
/// `ctrl.weak`; the last weak handle (or phantom) frees the control block. The data
/// block is never touched by a weak drop.
/// Release one deferred `weak` reference (commit phase of teardown): decrement the
/// control block's weak count and free the control block if this was the last handle.
pub(in crate::rtti) fn commit_weak_release<A: BStackRaiiAllocator>(
    alloc: &A,
    ctrl_off: u64,
) -> io::Result<()> {
    let data = alloc.stack();
    if refcount::fetch_sub(
        data,
        NonNullOffset::from_field(add_off(ctrl_off, CTRL_WEAK_OFFSET)?)?,
        1,
    )? == 1
    {
        // SAFETY: last weak released — the control block is unreferenced.
        unsafe { alloc.free_many([BStackRange::new(ctrl_off, CONTROL_SIZE)])? };
    }
    Ok(())
}

/// The low-`width`-byte mask for comparing a stored discriminant against a variant's
/// (sign-extended) value.
pub(in crate::rtti) fn disc_mask(width: u8) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Read a `width`-byte discriminant at `off`, zero-extended to `u64`.
pub(in crate::rtti) fn read_disc(data: &BStack, off: u64, width: u8) -> io::Result<u64> {
    let w = width as usize;
    if w > 8 {
        // A discriminant fits in a `u64`; a wider width is a corrupt schema. Return
        // `Err` rather than index a `[u8; 8]` out of bounds (`disc_mask` already
        // tolerates `>= 8`). `decode_type` rejects such records on load, so this is
        // a defensive backstop.
        return Err(io_error!(
            "[BSTACK0816] RTTI enum discriminant width exceeds 8 bytes",
        ));
    }
    let mut b = [0u8; 8];
    data.get_into(off, &mut b[..w])?;
    Ok(u64::from_le_bytes(b))
}

io_errorfn!(
    pub(in crate::rtti) class_error(msg: impl std::fmt::Display),
    InvalidInput,
    "[BSTACK0812] RTTI class variable: {}",
    msg
);

/// Pop the `n` values a container's children pushed, restoring declaration order.
/// Children are pushed onto `work` in forward order, so they execute (and land on
/// `results`) in reverse — this hands back `[c0, c1, …]`.
pub(in crate::rtti) fn pop_n(results: &mut Vec<Value>, n: usize) -> io::Result<Vec<Value>> {
    let start = results
        .len()
        .checked_sub(n)
        .ok_or_else(|| io_error!("[BSTACK0809] RTTI interpret stack underflow"))?;
    let mut v = results.split_off(start);
    v.reverse();
    Ok(v)
}

/// Pop `names.len()` values and pair them with the field names, in order.
pub(in crate::rtti) fn pop_named(
    results: &mut Vec<Value>,
    names: &[String],
) -> io::Result<Vec<(String, Value)>> {
    let vals = pop_n(results, names.len())?;
    Ok(names.iter().cloned().zip(vals).collect())
}
