//! Bounds-checked little-endian byte cursor over a slice — a generic, error-free
//! reader that reports a short read as `None` (and a failed alignment as `false`),
//! leaving the domain-specific error to the caller. The mirror of
//! [`Writer`](super::writer::Writer). Currently the backing cursor for the RTTI schema
//! decoder, which layers its typed reads and error framing on top; kept here as a
//! reusable primitive with no vocabulary of its own.

/// Bounds-checked little-endian byte cursor over a slice.
pub(crate) struct Reader<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) pos: usize,
}

impl<'a> Reader<'a> {
    #[inline(always)]
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Take the next `n` bytes, advancing the cursor. Returns `None` — leaving the
    /// cursor unmoved — when fewer than `n` bytes remain.
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len())?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Some(s)
    }

    /// Skip zero-padding up to the next `a`-byte boundary. Returns `false` — leaving
    /// the cursor unmoved — when that boundary lies past the end of the buffer.
    #[inline(always)]
    pub(crate) fn skip_pad(&mut self, a: usize) -> bool {
        let aligned = (self.pos + a - 1) & !(a - 1);
        if aligned > self.buf.len() {
            return false;
        }
        self.pos = aligned;
        true
    }
}
