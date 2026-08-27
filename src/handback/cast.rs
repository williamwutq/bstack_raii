//! [`CastError<S>`]: the error a fallible `bstack_cast!` downcast returns.

use std::error::Error;
use std::fmt;
use std::io;

/// The error a fallible downcast ([`BStackCastInto::cast_into`](crate::BStackCastInto::cast_into))
/// returns. It always hands the input **slice back** so an ownership-carrying
/// [`BStackOwnedSlice`](bstack::BStackOwnedSlice) is never dropped (and thus
/// leaked) on a failed cast — the same hand-back contract as the crate's other
/// consuming operations ([`ReplaceError`](crate::ReplaceError) /
/// [`ConstructError`](crate::ConstructError)).
///
/// It carries no `From<CastError> for io::Error` on purpose: a caller cannot `?` a
/// failed cast and silently drop the slice it was handed back — it must recover the
/// slice (try another type, or free it) via [`into_slice`](Self::into_slice).
///
/// Unlike the `source`-field hand-back errors, this is a two-variant enum — a clean
/// tag/size [`Mismatch`](Self::Mismatch) is not an I/O failure — so it hand-writes
/// its impls and does not implement [`HandBack`](crate::HandBack).
pub enum CastError<S> {
    /// The block's tag or on-disk size is not `T`'s — not an I/O failure, the block
    /// simply is not a `T`. The slice is handed back unchanged.
    Mismatch(S),
    /// Reading the block header faulted. The slice is intact and handed back with
    /// the underlying error.
    Io(io::Error, S),
}

impl<S> CastError<S> {
    /// Recover the handed-back slice, discarding *why* the cast failed. The slice
    /// still owns its block — try another type, free it, or re-wrap it.
    #[inline]
    pub fn into_slice(self) -> S {
        match self {
            CastError::Mismatch(s) | CastError::Io(_, s) => s,
        }
    }

    /// The underlying I/O error, or `None` for a clean tag/size mismatch (which is
    /// not an error condition).
    #[inline]
    pub fn io(&self) -> Option<&io::Error> {
        match self {
            CastError::Io(e, _) => Some(e),
            CastError::Mismatch(_) => None,
        }
    }
}

// Manual, so `S` (a slice handle) need not be `Debug`.
impl<S> fmt::Debug for CastError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CastError::Mismatch(_) => f.write_str("CastError::Mismatch(..)"),
            CastError::Io(e, _) => f.debug_tuple("CastError::Io").field(e).finish(),
        }
    }
}

impl<S> fmt::Display for CastError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CastError::Mismatch(_) => {
                f.write_str("bstack_cast!: block tag/size is not the target type")
            }
            CastError::Io(e, _) => fmt::Display::fmt(e, f),
        }
    }
}

impl<S> Error for CastError<S> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CastError::Io(e, _) => Some(e),
            CastError::Mismatch(_) => None,
        }
    }
}
