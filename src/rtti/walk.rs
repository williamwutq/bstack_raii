//! Shared **walk primitives** the interpreters all build on — the non-recursive
//! traversal helpers, error constructors, recursion/allocation guards, and constants
//! used by `read` / `teardown` / `clone` / `field` / `move` alike (which is why they
//! live here rather than in any one interpreter).
//!
//! Static analysis *of a [`Shape`](super::Shape)* — its on-disk layout and
//! classification (`shape_stride`, `has_reference`, `foreign_leaf`, …) — lives in
//! the sibling [`shape`](super::shape) module; this one holds the walk *mechanics*.

use bstack::{BStack, BStackRange};

use crate::BStackRaiiAllocator;
use crate::io_core::refcount;
use crate::primitives::{EightCC, NonNullOffset, Offset};
use crate::types::compiled::rc::CTRL_WEAK_OFFSET;

use super::{AnyRef, BYTEVEC_HEADER, CONTROL_SIZE, RttiEnum, RttiVariant, Value, add_off};
use super::{RttiResult, rtti_err};

/// Verify a **live block of type `tag`** sits at `off` in `data` (a null `off` is the
/// allowed sentinel — a null reference). The safe RTTI mutators install caller-supplied offsets into
/// owning slots; without this check a fabricated [`AnyRef`] / [`ForeignPtr`] could point
/// a slot at an arbitrary location that a later teardown would free (recursively, for
/// `owned`) or a later path would descend into — the same hazard `Foreign::new` /
/// `raw_<field>_slice` are `unsafe` for, but here checkable against the on-disk header.
pub(in crate::rtti) fn verify_data_block(
    data: &BStack,
    off: Offset,
    tag: EightCC,
) -> RttiResult<()> {
    // A null offset is the allowed sentinel (a null reference points nowhere).
    let Some(off) = off.to_non_null() else {
        return Ok(());
    };
    // Error for a mutator (`set` / `swap` / `swap_foreign`) whose caller-supplied
    // target offset does not name a live block of the field's declared type.
    let bad_target = |found: Option<EightCC>| {
        let found = match found {
            Some(t) => format!("found {t:?}"),
            None => "out of bounds or unreadable".to_string(),
        };
        rtti_err!(
            Mutator,
            "RTTI mutator: offset {} does not hold a live {:?} block ({})",
            off.as_u64(),
            tag,
            found
        )
    };
    match AnyRef::from_block(data, off.as_u64()) {
        Ok(a) if a.tag() == tag => Ok(()),
        Ok(a) => Err(bad_target(Some(a.tag()))),
        Err(_) => Err(bad_target(None)),
    }
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
    pub(in crate::rtti) fn enter() -> RttiResult<Self> {
        let depth = RTTI_DEPTH.with(|c| {
            let n = c.get() + 1;
            c.set(n);
            n
        });
        if depth > MAX_RTTI_DEPTH {
            RTTI_DEPTH.with(|c| c.set(c.get() - 1)); // undo: no guard is returned
            return Err(rtti_err!(
                Budget,
                "RTTI cross-file recursion too deep (a foreign cycle?)",
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
) -> RttiResult<u64> {
    let usable = data_size.saturating_sub(BYTEVEC_HEADER);
    if byte_len > usable {
        return Err(rtti_err!(
            VecLen,
            "RTTI vector length ({} bytes) exceeds its data block \
             ({} usable bytes) — corrupt length word",
            byte_len,
            usable
        ));
    }
    Ok(byte_len.checked_div(stride).unwrap_or(0))
}

/// Release one deferred `weak` reference (commit phase of teardown) whose control
/// block is at `ctrl_off`: decrement `ctrl.weak`; the last weak handle (or phantom)
/// frees the control block. The data block is never touched by a weak drop.
pub(in crate::rtti) fn commit_weak_release<A: BStackRaiiAllocator>(
    alloc: &A,
    ctrl_off: u64,
) -> RttiResult<()> {
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
pub(in crate::rtti) fn read_disc(data: &BStack, off: u64, width: u8) -> RttiResult<u64> {
    let w = width as usize;
    if w > 8 {
        // A discriminant fits in a `u64`; a wider width is a corrupt schema. Return
        // `Err` rather than index a `[u8; 8]` out of bounds (`disc_mask` already
        // tolerates `>= 8`). `decode_type` rejects such records on load, so this is
        // a defensive backstop.
        return Err(rtti_err!(
            DiscWidth,
            "RTTI enum discriminant width exceeds 8 bytes",
        ));
    }
    let mut b = [0u8; 8];
    data.get_into(off, &mut b[..w])?;
    Ok(u64::from_le_bytes(b))
}

impl RttiEnum {
    /// Resolve the **active variant** of an enum block at `block_off`: read the stored
    /// `disc_width`-byte discriminant (masked to that width), match it against the
    /// variants' `disc_value`s, and return the matched variant together with the base
    /// offset of its payload region (`block_off + payload_off`). `NoVariant` when no
    /// variant matches — a corrupt or truncated discriminant. Shared by every
    /// interpreter (`read` / `teardown` / `move` / `clone`) that walks into an enum.
    pub(in crate::rtti) fn resolve_variant<'e>(
        &'e self,
        data: &BStack,
        block_off: u64,
    ) -> RttiResult<(&'e RttiVariant, u64)> {
        let raw = read_disc(
            data,
            add_off(block_off, self.disc_off as u64)?,
            self.disc_width,
        )?;
        let mask = disc_mask(self.disc_width);
        let variant = self
            .variants
            .iter()
            .find(|v| (v.disc_value as u64) & mask == raw)
            .ok_or_else(|| rtti_err!(NoVariant, "no RTTI variant for discriminant {}", raw))?;
        let payload_base = add_off(block_off, self.payload_off as u64)?;
        Ok((variant, payload_base))
    }
}

/// Pop the `n` values a container's children pushed, restoring declaration order.
/// Children are pushed onto `work` in forward order, so they execute (and land on
/// `results`) in reverse — this hands back `[c0, c1, …]`.
pub(in crate::rtti) fn pop_n(results: &mut Vec<Value>, n: usize) -> RttiResult<Vec<Value>> {
    let start = results
        .len()
        .checked_sub(n)
        .ok_or_else(|| rtti_err!(Interpret, "RTTI interpret stack underflow"))?;
    let mut v = results.split_off(start);
    v.reverse();
    Ok(v)
}

/// Pop `names.len()` values and pair them with the field names, in order.
pub(in crate::rtti) fn pop_named(
    results: &mut Vec<Value>,
    names: &[String],
) -> RttiResult<Vec<(String, Value)>> {
    let vals = pop_n(results, names.len())?;
    Ok(names.iter().cloned().zip(vals).collect())
}
