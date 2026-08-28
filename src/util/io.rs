//! [`io_error!`] — terse construction of an [`io::Error`](std::io::Error).

/// Build an [`io::Error`](std::io::Error) from an [`ErrorKind`](std::io::ErrorKind)
/// variant and a message, without repeating `io::Error::new(io::ErrorKind::…, …)`.
///
/// * `$kind` — bare `ErrorKind` variant name (`InvalidData`, `NotFound`, …); expands
///   to `std::io::ErrorKind::$kind`. Omit it to default to `InvalidData`, the
///   overwhelmingly common kind at call sites in this crate (corrupt/overflowing
///   on-disk data).
/// * message — either a single expr (any `Into<Box<dyn Error + Send + Sync>>`: a
///   `&str`, `String`, or error value), or a format literal plus args (via
///   [`format!`]).
macro_rules! io_error {
    ($kind:ident, $fmt:literal, $($arg:tt)+) => {
        ::std::io::Error::new(
            ::std::io::ErrorKind::$kind,
            ::std::format!($fmt, $($arg)+),
        )
    };
    ($kind:ident, $msg:expr $(,)?) => {
        ::std::io::Error::new(::std::io::ErrorKind::$kind, $msg)
    };
    ($fmt:literal, $($arg:tt)+) => {
        ::std::io::Error::new(
            ::std::io::ErrorKind::InvalidData,
            ::std::format!($fmt, $($arg)+),
        )
    };
    ($msg:expr $(,)?) => {
        ::std::io::Error::new(::std::io::ErrorKind::InvalidData, $msg)
    };
}

/// Define a **named, reusable** error constructor — a `fn` returning an
/// [`io::Error`](std::io::Error) via [`io_error!`]. For an error value used at more
/// than one call site (so the message lives in one place), e.g. a shared overflow /
/// corruption sentinel, or a mutator's error shape shared across its call sites.
///
/// * Generated fn is `#[cold]`, `#[must_use]`, `#[inline]`; an optional visibility
///   precedes the name, e.g. `io_errorfn!(pub(crate) bad_tag, InvalidData, "…")`.
/// * No params: `io_errorfn!(offset_overflow, InvalidData, "…overflow")`, called as
///   `something.ok_or_else(offset_overflow)?`. Format args must be in scope where
///   the fn is *defined* (module consts), not supplied by the caller.
/// * Params: `io_errorfn!(pub(crate) bad_len(len: usize), InvalidData, "bad length
///   {len}")`, called as `Err(bad_len(len))` — args thread into the format string
///   like any [`io_error!`] call.
/// * Doc comments/attributes may precede the name; they attach to the generated fn
///   as on an ordinary `fn` item.
macro_rules! io_errorfn {
    ($(#[$attr:meta])* $vis:vis $name:ident, $kind:ident, $fmt:literal, $($arg:tt)+) => {
        $(#[$attr])*
        #[cold]
        #[must_use]
        #[inline]
        $vis fn $name() -> ::std::io::Error {
            $crate::util::io::io_error!($kind, $fmt, $($arg)+)
        }
    };
    ($(#[$attr:meta])* $vis:vis $name:ident, $kind:ident, $msg:expr $(,)?) => {
        $(#[$attr])*
        #[cold]
        #[must_use]
        #[inline]
        $vis fn $name() -> ::std::io::Error {
            $crate::util::io::io_error!($kind, $msg)
        }
    };
    ($(#[$attr:meta])* $vis:vis $name:ident ( $($pname:ident : $pty:ty),+ $(,)? ), $kind:ident, $fmt:literal, $($arg:tt)+) => {
        $(#[$attr])*
        #[cold]
        #[must_use]
        #[inline]
        $vis fn $name($($pname: $pty),+) -> ::std::io::Error {
            $crate::util::io::io_error!($kind, $fmt, $($arg)+)
        }
    };
    ($(#[$attr:meta])* $vis:vis $name:ident ( $($pname:ident : $pty:ty),+ $(,)? ), $kind:ident, $msg:expr $(,)?) => {
        $(#[$attr])*
        #[cold]
        #[must_use]
        #[inline]
        $vis fn $name($($pname: $pty),+) -> ::std::io::Error {
            $crate::util::io::io_error!($kind, $msg)
        }
    };
}

pub(crate) use io_error;
pub(crate) use io_errorfn;
