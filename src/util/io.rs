//! [`io_error!`] — terse construction of an [`io::Error`](std::io::Error).

/// Build an [`io::Error`](std::io::Error) from an [`ErrorKind`](std::io::ErrorKind)
/// variant and a message, without repeating `io::Error::new(io::ErrorKind::…, …)`.
///
/// * `io_error!(InvalidData, "block offset overflow")` — a static message (any
///   `Into<Box<dyn Error + Send + Sync>>`: a `&str`, a `String`, an error value).
/// * `io_error!(InvalidData, "bad tag {tag:#x} at {off}")` — a format string plus
///   args, formatted with [`format!`].
///
/// The first token is the bare `ErrorKind` variant name (`InvalidData`, `NotFound`,
/// `Other`, …); it expands to `std::io::ErrorKind::$variant`.
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
}

/// Define a **named, reusable** error constructor — a zero-argument `fn` returning a
/// fixed [`io::Error`](std::io::Error) via [`io_error!`]. For an error value used at
/// more than one call site (so the message lives in one place), e.g. a shared
/// overflow / corruption sentinel.
///
/// The generated fn is `#[cold]` (it is only called on the failure path),
/// `#[must_use]` (the returned error must be propagated, not dropped), and
/// `#[inline]`. An optional visibility precedes the name.
///
/// ```ignore
/// io_errorfn!(offset_overflow, InvalidData, "on-disk offset arithmetic overflow");
/// io_errorfn!(pub(crate) bad_tag, InvalidData, "unknown tag");
/// // then: `something.ok_or_else(offset_overflow)?`
/// ```
///
/// The message may also be a format string plus args, but — since the fn takes no
/// parameters — those args must be in scope where the fn is *defined* (module consts).
/// For a message built from runtime values, call [`io_error!`] inline instead.
macro_rules! io_errorfn {
    ($vis:vis $name:ident, $kind:ident, $fmt:literal, $($arg:tt)+) => {
        #[cold]
        #[must_use]
        #[inline]
        $vis fn $name() -> ::std::io::Error {
            $crate::util::io::io_error!($kind, $fmt, $($arg)+)
        }
    };
    ($vis:vis $name:ident, $kind:ident, $msg:expr $(,)?) => {
        #[cold]
        #[must_use]
        #[inline]
        $vis fn $name() -> ::std::io::Error {
            $crate::util::io::io_error!($kind, $msg)
        }
    };
}

pub(crate) use io_error;
pub(crate) use io_errorfn;
