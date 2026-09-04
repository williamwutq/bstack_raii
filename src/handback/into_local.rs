//! [`IntoLocalError<H>`]: the error a fallible `Foreign*::into_local` returns.

use std::{fmt, io};

use super::impl_source_error;

/// The error [`ForeignOwned::into_local`](crate::ForeignOwned::into_local) /
/// [`ForeignRc::into_local`](crate::ForeignRc::into_local) /
/// [`ForeignWeak::into_local`](crate::ForeignWeak::into_local) returns.
///
/// `into_local` **consumes** the foreign handle to resolve it to an in-file handle. A
/// bare `io::Result` would drop that handle on failure — and because a `Foreign*` handle
/// has no `Drop` (release is explicit, via `bstack_drop` / `into_foreign`), the
/// transferred reference (a strong/weak count, or a whole owned block) would become an
/// unreachable orphan. So a failed `into_local` hands the original handle back in
/// [`handle`](Self::handle), intact — its reference was never released — the same
/// hand-back contract as [`ReplaceError`](crate::ReplaceError) /
/// [`ConstructError`](crate::ConstructError). Recover it with
/// [`into_handle`](Self::into_handle): re-store it, resolve it against the correct file,
/// or `bstack_drop` it. Like the other hand-back errors it carries no
/// `From<IntoLocalError> for io::Error`, so a caller cannot `?` it and silently drop the
/// handle.
///
/// Implements [`std::error::Error`] (delegating [`Display`](fmt::Display) to
/// [`source`](Self::source)).
pub struct IntoLocalError<H> {
    /// The underlying failure — a wrong-file rejection (the target allocator does not
    /// address the pointer's home file), or an I/O fault reading the control block.
    pub source: io::Error,
    /// The foreign handle, handed back intact so its reference is recoverable, not
    /// leaked. Re-store it, retry against the right file, or `bstack_drop` it — dropping
    /// it as-is leaks (a `Foreign*` handle is unrooted).
    pub handle: H,
}

impl<H> IntoLocalError<H> {
    /// An error handing the still-valid foreign `handle` back to the caller.
    #[inline]
    pub fn recovered(source: io::Error, handle: H) -> Self {
        Self { source, handle }
    }

    /// Recover the handed-back handle, discarding *why* the conversion failed. Its
    /// reference is intact — re-store it, resolve it, or free it.
    #[inline]
    pub fn into_handle(self) -> H {
        self.handle
    }

    /// Discard the recovered handle and take just the underlying `io::Error`. Explicit,
    /// because dropping the handle leaks its reference.
    #[inline]
    pub fn into_source(self) -> io::Error {
        self.source
    }
}

// Manual, so `H` (a foreign handle) need not be `Debug`.
impl<H> fmt::Debug for IntoLocalError<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoLocalError")
            .field("source", &self.source)
            .field("handle", &"...")
            .finish()
    }
}

impl_source_error!(IntoLocalError<H>);
