//! [`ConstructError<F>`]: the error a generated `new` constructor returns when it
//! fails after consuming the caller's children.

use std::fmt;
use std::io;

use super::impl_source_error;

/// The error a generated `new` constructor returns when a fallible construction
/// step fails after it has already consumed the caller's owned/strong/embedded
/// children.
///
/// A generated `new` **consumes** the child handles you hand it (`#[bstack_owned]`,
/// `#[bstack_strong]`, `#[embed]`). A bare `io::Result` would then *lose* those
/// children on an allocation or write failure: their on-disk blocks would be
/// neither linked into the new block, freed, nor returned — an unreachable orphan
/// whose contents (an arbitrarily large subtree the caller just built) are
/// **unrecoverable**. So a failed `new` returns this instead,
/// handing the still-valid children back in [`fields`](Self::fields) — the same
/// region-hand-back contract as bstack's `BStackAllocError`,
/// [`BStackRaiiAllocError`](crate::registry::BStackRaiiAllocError), and
/// [`ReplaceError`](crate::ReplaceError).
///
/// `F` is the block's [`Fields`](crate::BStackMove::Fields) tuple — exactly
/// what `bstack_move!` hands back — so a recovered construction returns the
/// children in the same shape a later move would.
///
/// Only constructors that actually consume an owning child return this; a
/// constructor whose fields are all POD / `#[bstack_ref]` (nothing to orphan)
/// keeps a plain `io::Result`. There is deliberately **no**
/// `From<ConstructError> for io::Error`: propagating one with `?` would drop the
/// recovered children and silently re-orphan them, reintroducing the very defect
/// this type exists to prevent. Handle it explicitly, or `.unwrap()` it.
///
/// Implements [`std::error::Error`] (delegating [`Display`](fmt::Display) to
/// [`source`](Self::source)).
pub struct ConstructError<F> {
    /// The underlying I/O error that caused the construction to fail.
    pub source: io::Error,
    /// The children the constructor consumed, handed back if they survived.
    ///
    /// * `Some` — recovered: the children are intact and yours again, in the
    ///   block's `bstack_move!` [`Fields`](crate::BStackMove::Fields)
    ///   shape. Retry `new`, re-home them, or free each — dropping them as-is may
    ///   leak, since a bare handle is unrooted (the crate's
    ///   *moved-out-is-unrooted* rule). Every allocation/write failure path takes
    ///   this branch: the children were never touched, only their offsets read.
    /// * `None` — the children could not be handed back here. Generated
    ///   constructors never produce this today (they keep the original handles, so
    ///   recovery is infallible); it exists for parity with
    ///   [`ReplaceError`](crate::ReplaceError) and future fallible-recovery paths.
    pub fields: Option<F>,
}

impl<F> ConstructError<F> {
    /// An error that hands the still-valid children back to the caller.
    #[inline]
    pub fn recovered(source: io::Error, fields: F) -> Self {
        Self {
            source,
            fields: Some(fields),
        }
    }

    /// An error whose children could not be recovered here (see
    /// [`fields`](Self::fields)).
    #[inline]
    pub fn lost(source: io::Error) -> Self {
        Self {
            source,
            fields: None,
        }
    }

    /// Discard the recovered children (if any) and take just the underlying
    /// `io::Error`. Explicit, because dropping recovered children may leak — call
    /// this only when you have decided not to reclaim them.
    #[inline]
    pub fn into_source(self) -> io::Error {
        self.source
    }
}

/// Propagate an inner `io::Error` as a `lost` construction (`fields: None`).
///
/// This is for the generated constructor's own `?` on the fallible *preparation*
/// steps — allocating a child vector's data block, re-encoding a `Foreign`, an
/// `#[embed]` copy — that run before (or between) the block's own allocation and
/// write. Those faults are rare mid-construction I/O errors; degrading them to
/// `lost` matches the pre-existing behaviour (the children a partial prep already
/// consumed were orphaned there too) while the *primary* allocation / write /
/// commit failures still hand the children back through
/// [`recovered`](ConstructError::recovered).
///
/// This affects only `?` **inside** a constructor (an `io::Error` becoming a
/// `ConstructError`); it deliberately does **not** provide the reverse
/// (`From<ConstructError> for io::Error`), so a caller cannot silently `?` a
/// failed `new` and re-orphan the returned children.
impl<F> From<io::Error> for ConstructError<F> {
    #[inline]
    fn from(source: io::Error) -> Self {
        Self::lost(source)
    }
}

// Manual, so `F` need not be `Debug` (the handed-back handles generally aren't).
impl<F> fmt::Debug for ConstructError<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstructError")
            .field("source", &self.source)
            .field("fields", &self.fields.as_ref().map(|_| "..."))
            .finish()
    }
}

impl_source_error!(ConstructError<F>);
