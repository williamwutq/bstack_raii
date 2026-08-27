//! [`ReplaceError<V>`]: the error a generated `replace_<field>` mutator returns.

use std::fmt;
use std::io;

use bstack::BStackRange;

use crate::BStackRaiiAllocator;
use crate::types::traits::drop::BStackDrop;
use crate::util::handback::impl_source_error;

/// The error a [`#[bstack_mut]`](crate::bstack_block) `replace_<field>` mutator
/// returns when the swap fails partway through.
///
/// `replace_<field>` **consumes** the value you hand it. A bare `io::Result`
/// would then *lose* that value on an I/O failure — its on-disk block would be
/// neither linked into the field nor returned, an unreachable orphan. So a failed
/// `replace_` returns this instead, handing the still-valid value back in
/// [`value`](Self::value) — the same region-hand-back contract as bstack's
/// `BStackAllocError` and [`BStackRaiiAllocError`](crate::registry::BStackRaiiAllocError).
///
/// When the **commit** fails the old value is never at risk: the swap is a single
/// crash-atomic exchange, so the field still holds it (it is simply not moved
/// out) and [`value`](Self::value) hands the *new* one back. The one exception is
/// a `#[bstack_strong]` field whose commit *succeeds* but whose post-commit
/// reconstruction of the old value then fails — there [`value`](Self::value) is
/// `None`, but the old block's raw offset is handed back in
/// [`raw_old`](Self::raw_old) so it is **recoverable**, not lost (see its docs).
///
/// Implements [`std::error::Error`] (delegating [`Display`](fmt::Display) to
/// [`source`](Self::source)).
pub struct ReplaceError<V> {
    /// The underlying I/O error that caused the swap to fail.
    pub source: io::Error,
    /// The value that was to be installed, handed back if it survived.
    ///
    /// * `Some` — recovered: the value is intact and yours again. Re-attach it
    ///   (retry `replace_`) or free it — dropping it as-is may leak, since a bare
    ///   handle is unrooted (see the crate's *moved-out-is-unrooted* rule).
    /// * `None` — the value handle could not be reconstructed here (see
    ///   [`raw_old`](Self::raw_old) for what to do). This arises only when a
    ///   `#[bstack_strong]` `replace_`'s commit *succeeded* but the post-commit
    ///   reconstruction of the *old* value then failed: the new value is safely
    ///   installed, and it is the **old** block that could not be handed back as a
    ///   typed handle.
    pub value: Option<V>,
    /// Raw on-disk offsets of blocks that survived the operation but could not be
    /// reconstructed into a typed handle here — **still allocated and
    /// recoverable**, not leaked.
    ///
    /// Populated only on the `value == None` path above: a `#[bstack_strong]`
    /// `replace_` whose old-value reconstruction (a re-read of the target's
    /// control-block pointer) failed. The old block is untouched at these
    /// offsets; the caller can retry the reconstruction once I/O recovers
    /// (`BStackShared::strong_parts` + `BStackRc::from_raw`), re-attach it, or free
    /// it. Empty on every ordinary (`value`-carrying) error.
    pub raw_old: Vec<BStackRange>,
}

impl<V> ReplaceError<V> {
    /// An error that hands the still-valid `value` back to the caller.
    #[inline]
    pub fn recovered(source: io::Error, value: V) -> Self {
        Self {
            source,
            value: Some(value),
            raw_old: Vec::new(),
        }
    }

    /// An error whose value could not be recovered here (see [`value`](Self::value)).
    #[inline]
    pub fn lost(source: io::Error) -> Self {
        Self {
            source,
            value: None,
            raw_old: Vec::new(),
        }
    }

    /// Like [`lost`](Self::lost), but hands back the raw offsets of the surviving
    /// old block(s) in [`raw_old`](Self::raw_old) so they are recoverable rather
    /// than leaked.
    #[inline]
    pub fn lost_raw(source: io::Error, raw_old: Vec<BStackRange>) -> Self {
        Self {
            source,
            value: None,
            raw_old,
        }
    }

    /// Discard the recovered value (if any) and take just the underlying
    /// `io::Error`. Explicit, because dropping a recovered value may leak.
    #[inline]
    pub fn into_source(self) -> io::Error {
        self.source
    }
}

impl<V: BStackDrop> ReplaceError<V> {
    /// Free the recovered resource (if any) and return the underlying error.
    ///
    /// For an internal call site that consumed a resource *it* produced (not a
    /// resource its own caller handed in) and so has nothing to hand the value
    /// onward to — e.g. a fused entry helper whose inner `insert` failed. This
    /// keeps the failure a reclaimable free, not an orphan; it is *not* the
    /// right choice at a public boundary, where the value should be returned.
    #[inline]
    pub fn discard_freeing<A: BStackRaiiAllocator>(self, allocator: &A) -> io::Error {
        if let Some(value) = self.value {
            let _ = value.bstack_drop(allocator);
        }
        self.source
    }
}

// Manual, so `V` need not be `Debug` (the handed-back handles generally aren't).
impl<V> fmt::Debug for ReplaceError<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplaceError")
            .field("source", &self.source)
            .field("value", &self.value.as_ref().map(|_| "..."))
            .field("raw_old", &self.raw_old)
            .finish()
    }
}

impl_source_error!(ReplaceError<V>);
