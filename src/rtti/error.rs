//! [`RttiError`] — the error type of the RTTI interpreter and schema codec.
//!
//! RTTI failures are **schema / data-integrity** errors (a truncated record, an unknown
//! tag, a corrupt discriminant, a blown recursion budget), not the syscall failures that
//! make [`std::io::Error`] the right type for `bstack` proper. So the interpreter/codec
//! layer speaks [`RttiResult`], not `io::Result`, and this type carries the pieces that
//! actually matter: a [`kind`](RttiErrorKind) (the stable `[BSTACK08xx]` code plus its
//! `io::ErrorKind` mapping) and a human [`message`](RttiError::message).
//!
//! ## Direction of conversion
//!
//! RTTI is a **leaf**: nothing in `io_core` / `types` calls it, while it calls plenty of
//! `io::Result` primitives (stack reads, the allocator, the WAL, refcounts). So the
//! dominant flow is **`io::Error` → `RttiError`** ([`From<io::Error>`](RttiError), used
//! pervasively via `?` inside the interpreters). The reverse — **`RttiError` → `io::Error`**
//! ([`From<RttiError>`](io::Error)) — is needed only where an RTTI error legitimately
//! surfaces under `io::Result`: the registry **persistence** methods (`open` / `sync` /
//! `append`), which do real filesystem I/O and so keep returning `io::Result`.

use std::borrow::Cow;
use std::fmt;
use std::io;

/// The result type of the RTTI interpreter / schema codec.
pub type RttiResult<T> = Result<T, RttiError>;

/// The category of an [`RttiError`]: a stable `[BSTACK08xx]` code, plus the
/// [`io::ErrorKind`] it maps to should it cross into the persistence layer's
/// `io::Result`. `Copy` — the cheap half of the error that gets matched on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RttiErrorKind {
    /// `0800` — a malformed or conflicting record at load / registration.
    Malformed,
    /// `0801` — an RTTI ordinal is out of range.
    OrdinalRange,
    /// `0802` — an RTTI name is not valid UTF-8.
    Utf8,
    /// `0803` — an unknown RTTI shape tag or foreign kind.
    UnknownTag,
    /// `0804` — a truncated RTTI record.
    Truncated,
    /// `0805` — an RTTI field's shape length disagrees with its encoding.
    ShapeLenMismatch,
    /// `0806` — an RTTI eightcc collision.
    Collision,
    /// `0807` — an interpret / clone / teardown budget or cross-file recursion cap
    /// was exceeded (corrupt data or a cycle).
    Budget,
    /// `0808` — no RTTI variant matches the on-disk discriminant.
    NoVariant,
    /// `0809` — an RTTI interpret stack underflow or wrong final value count.
    Interpret,
    /// `080A` — an untyped / out-of-range RTTI pointer cannot be read.
    UntypedPointer,
    /// `080B` — a pointer or field references an unregistered type tag.
    UnregisteredTag,
    /// `080D` — an RTTI `set` mutator rejected its argument.
    Set,
    /// `080E` — a deep clone did not reproduce a required block.
    Clone,
    /// `080F` — a foreign target's file id is invalid or its file is detached.
    ForeignFile,
    /// `0810` — an RTTI `swap` mutator rejected its argument.
    Swap,
    /// `0811` — an RTTI operation is not (yet) supported.
    Unsupported,
    /// `0812` — an RTTI class-variable operation failed.
    Class,
    /// `0813` — a vector length word exceeds its data block.
    VecLen,
    /// `0814` — a persisted schema disagrees with the compiled type.
    SchemaMismatch,
    /// `0815` — an RTTI mutator's target offset is not a live block of the field type.
    Mutator,
    /// `0816` — an RTTI enum discriminant width is invalid.
    DiscWidth,
    /// `0817` — a value exceeds the maximum encodable RTTI size.
    TooLarge,
    /// `0818` — RTTI shape nesting exceeds the maximum depth.
    Depth,
    /// `0819` — `move_out` of a shared reference-counted block.
    SharedMove,
    /// `081A` — RTTI offset arithmetic overflow.
    OffsetOverflow,
    /// An absorbed [`std::io::Error`] from an underlying read / allocator / WAL call —
    /// the one non-RTTI category, carrying the original [`io::ErrorKind`] so a
    /// round-trip back to `io::Error` is lossless.
    Io(io::ErrorKind),
}

impl RttiErrorKind {
    /// The stable `BSTACK08xx` code for this category — every kind has one, including
    /// an absorbed [`Io`](Self::Io) error (`081B`).
    pub fn code(self) -> u16 {
        match self {
            Self::Malformed => 0x0800,
            Self::OrdinalRange => 0x0801,
            Self::Utf8 => 0x0802,
            Self::UnknownTag => 0x0803,
            Self::Truncated => 0x0804,
            Self::ShapeLenMismatch => 0x0805,
            Self::Collision => 0x0806,
            Self::Budget => 0x0807,
            Self::NoVariant => 0x0808,
            Self::Interpret => 0x0809,
            Self::UntypedPointer => 0x080A,
            Self::UnregisteredTag => 0x080B,
            Self::Set => 0x080D,
            Self::Clone => 0x080E,
            Self::ForeignFile => 0x080F,
            Self::Swap => 0x0810,
            Self::Unsupported => 0x0811,
            Self::Class => 0x0812,
            Self::VecLen => 0x0813,
            Self::SchemaMismatch => 0x0814,
            Self::Mutator => 0x0815,
            Self::DiscWidth => 0x0816,
            Self::TooLarge => 0x0817,
            Self::Depth => 0x0818,
            Self::SharedMove => 0x0819,
            Self::OffsetOverflow => 0x081A,
            Self::Io(_) => 0x081B,
        }
    }

    /// The [`io::ErrorKind`] this maps to when the error crosses into `io::Result` — used
    /// only at the RTTI persistence boundary (see the module docs). Almost everything is
    /// [`InvalidData`](io::ErrorKind::InvalidData); the mutators are
    /// [`InvalidInput`](io::ErrorKind::InvalidInput), the two lookups
    /// [`NotFound`](io::ErrorKind::NotFound), and an absorbed `Io` round-trips its own.
    pub fn io_kind(self) -> io::ErrorKind {
        match self {
            Self::OrdinalRange | Self::ForeignFile => io::ErrorKind::NotFound,
            Self::Set | Self::Swap | Self::Class | Self::Mutator => io::ErrorKind::InvalidInput,
            Self::Unsupported => io::ErrorKind::Unsupported,
            Self::Io(k) => k,
            _ => io::ErrorKind::InvalidData,
        }
    }
}

/// An RTTI interpreter / codec error: a [`kind`](RttiErrorKind) (the stable code) and a
/// human message. The message is a [`Cow`] — a fixed string is `Borrowed` (no
/// allocation), a message built from runtime values is `Owned`; construct with the
/// `rtti_err!` macro rather than by hand.
#[derive(Clone, Debug)]
pub struct RttiError {
    kind: RttiErrorKind,
    msg: Cow<'static, str>,
}

impl RttiError {
    /// Construct from a category and message. Prefer the `rtti_err!` macro, which
    /// picks `Cow::Borrowed` / `Cow::Owned` for you.
    pub fn new(kind: RttiErrorKind, msg: Cow<'static, str>) -> Self {
        Self { kind, msg }
    }

    /// This error's category.
    pub fn kind(&self) -> RttiErrorKind {
        self.kind
    }

    /// The stable `BSTACK08xx` code for this error (an absorbed [`io::Error`](RttiErrorKind::Io)
    /// is `081B`).
    pub fn code(&self) -> u16 {
        self.kind.code()
    }

    /// The human message, without the `[BSTACK08xx]` prefix (which [`Display`](fmt::Display)
    /// adds from the [`kind`](Self::kind)).
    pub fn message(&self) -> &str {
        &self.msg
    }
}

impl fmt::Display for RttiError {
    /// Renders as `[BSTACK08xx] message`, reproducing the legacy string form — so a
    /// consumer matching on the code text still works. An absorbed `Io` error prints its
    /// own message under the `081B` code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[BSTACK{:04X}] {}", self.kind.code(), self.msg)
    }
}

impl std::error::Error for RttiError {}

impl From<io::Error> for RttiError {
    /// Absorb an underlying [`io::Error`] (a stack read, allocator, WAL, or refcount
    /// failure) into the interpreter's error channel, preserving its kind and message.
    fn from(e: io::Error) -> Self {
        Self {
            kind: RttiErrorKind::Io(e.kind()),
            msg: Cow::Owned(e.to_string()),
        }
    }
}

impl From<RttiError> for io::Error {
    /// Surface an RTTI error under `io::Result` at the persistence boundary. Boxes once
    /// (here, on the error path only), keeping the [`RttiError`] recoverable via
    /// `io::Error::downcast_ref`.
    fn from(e: RttiError) -> Self {
        io::Error::new(e.kind.io_kind(), e)
    }
}

/// Construct an [`RttiError`]: `rtti_err!(Kind, "message")` for a fixed message (a
/// zero-allocation `Cow::Borrowed`), or `rtti_err!(Kind, "fmt {}", value)` for one built
/// from runtime values (`Cow::Owned`). The `[BSTACK08xx]` code comes from `Kind` — do
/// **not** repeat it in the message string.
macro_rules! rtti_err {
    ($kind:ident, $lit:literal $(,)?) => {
        $crate::rtti::RttiError::new(
            $crate::rtti::RttiErrorKind::$kind,
            ::std::borrow::Cow::Borrowed($lit),
        )
    };
    ($kind:ident, $fmt:literal, $($arg:tt)+) => {
        $crate::rtti::RttiError::new(
            $crate::rtti::RttiErrorKind::$kind,
            ::std::borrow::Cow::Owned(::std::format!($fmt, $($arg)+)),
        )
    };
}

pub(crate) use rtti_err;
