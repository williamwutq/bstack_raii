//! Append-only little-endian writer over a growing byte buffer — the crate's
//! hand-rolled serializer for building fixed-layout on-disk records. Pairs with
//! [`Reader`](super::reader::Reader), its bounds-checked deserializer. Currently the
//! backing codec for the RTTI schema stack; kept here (not in `rtti`) as a generic,
//! reusable byte cursor.

use crate::primitives::EightCC;

/// Append-only little-endian writer over a growing byte buffer.
#[derive(Default)]
pub(crate) struct Writer {
    pub(crate) buf: Vec<u8>,
}

impl Writer {
    #[inline(always)]
    pub(crate) fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    #[inline(always)]
    pub(crate) fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    pub(crate) fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    pub(crate) fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    pub(crate) fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    pub(crate) fn eightcc(&mut self, v: EightCC) {
        self.buf.extend_from_slice(&v.0);
    }
    #[inline(always)]
    pub(crate) fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
    /// Pad with zero bytes up to the next `a`-byte boundary (`a` a power of two).
    ///
    /// Requires `a` to be a power of 2
    #[inline(always)]
    pub(crate) fn align(&mut self, a: usize) {
        let mask = a - 1;
        let new_len = (self.buf.len() + mask) & !mask;
        self.buf.resize(new_len, 0);
    }
}
