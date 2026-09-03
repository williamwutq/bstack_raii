//! Bounds-checked little-endian byte cursor over a slice — a generic, error-free
//! reader that reports a short read as `None` (and a failed alignment as `false`),
//! leaving the domain-specific error to the caller. The read mirror of
//! [`Writer`](super::writer::Writer): the same typed little-endian primitives
//! (`u8` / `u16` / `u32` / `u64` / `i64` / `eightcc`), inverted. The RTTI schema decoder
//! is its current caller; it layers only its error framing on top, keeping this a
//! reusable primitive with no vocabulary of its own.

use crate::primitives::EightCC;

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

    /// The next byte. `None` on underrun.
    #[inline(always)]
    pub(crate) fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    /// The next little-endian `u16`. `None` on underrun.
    #[inline(always)]
    pub(crate) fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    /// The next little-endian `u32`. `None` on underrun.
    #[inline(always)]
    pub(crate) fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// The next little-endian `u64`. `None` on underrun.
    #[inline(always)]
    pub(crate) fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// The next little-endian `i64`. `None` on underrun.
    #[inline(always)]
    pub(crate) fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// The next 8-byte [`EightCC`]. `None` on underrun.
    #[inline(always)]
    pub(crate) fn eightcc(&mut self) -> Option<EightCC> {
        Some(EightCC(self.take(8)?.try_into().unwrap()))
    }

    /// Skip zero-padding up to the next `a`-byte boundary (`a` a power of two). Returns
    /// `false` — leaving the cursor unmoved — when that boundary lies past the end of
    /// the buffer, or when rounding up would overflow.
    #[inline(always)]
    pub(crate) fn skip_pad(&mut self, a: usize) -> bool {
        let Some(aligned) = self.pos.checked_add(a - 1).map(|p| p & !(a - 1)) else {
            return false;
        };
        if aligned > self.buf.len() {
            return false;
        }
        self.pos = aligned;
        true
    }
}
