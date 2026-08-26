//! Shared machinery for the crate's **hand-back errors**.
//!
//! Several operations *consume* a resource (a moved-in child, a value to install,
//! a freshly resized region). When such an operation fails partway, a bare
//! `io::Error` would silently drop that resource — its on-disk block neither
//! linked nor returned, an unreachable orphan. So each such operation returns a
//! dedicated error that wraps the underlying [`io::Error`] **and hands the
//! resource back**: [`ConstructError`](crate::ConstructError),
//! [`ReplaceError`](crate::ReplaceError),
//! [`BStackRaiiAllocError`](crate::registry::BStackRaiiAllocError),
//! [`FreeManyError`](crate::FreeManyError), and the cast
//! [`CastError`](crate::CastError).
//!
//! Every one wraps an `io::Error` in a `source` field and delegates
//! [`Display`](std::fmt::Display) / [`Error`](std::error::Error) to it identically.
//! [`impl_source_error!`] emits that shared boilerplate (and the [`HandBack`] impl)
//! so each type only hand-writes what actually differs: its constructors, its
//! recovered-resource accessors, and its `Debug` (which masks the handed-back
//! resource, since the resource generally is not `Debug`).

use std::io;

/// The common surface of a hand-back error: it wraps an underlying [`io::Error`].
///
/// The handed-back *resource* differs per type (a value, a child tuple, a range,
/// a slice), so it stays on the concrete type's own accessors; this trait is just
/// the shared "what went wrong" half.
pub trait HandBack {
    /// Borrow the underlying I/O error that caused the operation to fail.
    fn io(&self) -> &io::Error;
}

/// Emit the `Display` / `Error` / [`HandBack`] impls shared by every hand-back
/// error whose underlying `io::Error` lives in a `source` field. `Debug` is left
/// to the type (it masks the non-`Debug` handed-back resource).
///
/// Usage: `impl_source_error!(BStackRaiiAllocError);` or, for a generic type,
/// `impl_source_error!(ReplaceError<V>);`.
macro_rules! impl_source_error {
    ($ty:ident $(< $($g:ident),+ >)?) => {
        impl $(< $($g),+ >)? ::core::fmt::Display for $ty $(< $($g),+ >)? {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(&self.source, f)
            }
        }
        impl $(< $($g),+ >)? ::std::error::Error for $ty $(< $($g),+ >)? {
            fn source(&self) -> ::core::option::Option<&(dyn ::std::error::Error + 'static)> {
                ::core::option::Option::Some(&self.source)
            }
        }
        impl $(< $($g),+ >)? $crate::util::handback::HandBack for $ty $(< $($g),+ >)? {
            fn io(&self) -> &::std::io::Error {
                &self.source
            }
        }
    };
}

pub(crate) use impl_source_error;
