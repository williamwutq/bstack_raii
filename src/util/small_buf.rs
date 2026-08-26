//! [`SmallBuf`] — a small inline-or-heap byte buffer for write batches.

/// A write payload for the `(u64, SmallBuf)` batches the stdlib collections'
/// atomic mutators (`atomic_update` / `probe_commit` / `BStack::set_batched`)
/// take. Two on-disk shapes recur often enough to inline without a heap
/// allocation: a single `u64` field (every counter/offset/length bump — the
/// overwhelming majority of writes) and a linked-list node's whole image (the
/// 16-byte [`crate::BlockHeader`] plus `prev`/`next`/`val`, 3 `u64`s — 40 bytes).
/// Deliberately **no length field** — each inline variant is exact-size-only
/// (never "up to N bytes"), so there is nothing to track; anything that isn't
/// exactly 8 or 40 bytes (a B-tree node, a bucket-table image, a
/// generic-`K`-sized heap slot, …) goes through [`SmallBuf::Heap`].
#[derive(Hash, Eq, PartialEq, Clone)]
pub(crate) enum SmallBuf {
    Buf8([u8; 8]),
    Buf40([u8; 40]),
    Heap(Box<[u8]>),
}

impl SmallBuf {
    #[inline(always)]
    pub(crate) fn buf_8() -> Self {
        Self::Buf8([0u8; 8])
    }

    #[allow(unused)]
    #[inline(always)]
    pub(crate) fn buf_40() -> Self {
        Self::Buf40([0u8; 40])
    }

    #[inline]
    pub(crate) fn new(len: usize) -> Self {
        if len <= 8 {
            Self::Buf8([0u8; 8])
        } else if len <= 40 {
            Self::Buf40([0u8; 40])
        } else {
            Self::Heap(Box::from(vec![0u8; len]))
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            SmallBuf::Buf8(b) => b.as_slice(),
            SmallBuf::Buf40(b) => b.as_slice(),
            SmallBuf::Heap(b) => b.as_ref(),
        }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            SmallBuf::Buf8(b) => b.as_mut_slice(),
            SmallBuf::Buf40(b) => b.as_mut_slice(),
            SmallBuf::Heap(b) => b.as_mut(),
        }
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            SmallBuf::Buf8(b) => 8,
            SmallBuf::Buf40(b) => 40,
            SmallBuf::Heap(b) => b.len(),
        }
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl AsRef<[u8]> for SmallBuf {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for SmallBuf {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl From<&[u8]> for SmallBuf {
    #[inline]
    fn from(s: &[u8]) -> Self {
        let len = s.len();
        let mut buf = Self::new(len);
        let slice = &mut buf.as_mut_slice()[..len];
        slice.copy_from_slice(s);
        buf
    }
}

impl<const N: usize> From<[u8; N]> for SmallBuf {
    #[inline(always)]
    fn from(arr: [u8; N]) -> Self {
        Self::from(&arr[..])
    }
}

impl From<SmallBuf> for Box<[u8]> {
    #[inline]
    fn from(buf: SmallBuf) -> Self {
        match buf {
            SmallBuf::Buf8(b) => b[..].into(),
            SmallBuf::Buf40(b) => b[..].into(),
            SmallBuf::Heap(b) => b,
        }
    }
}

impl PartialEq<[u8]> for SmallBuf {
    #[inline(always)]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<SmallBuf> for [u8] {
    #[inline(always)]
    fn eq(&self, other: &SmallBuf) -> bool {
        self == other.as_slice()
    }
}

impl PartialEq<Vec<u8>> for SmallBuf {
    #[inline(always)]
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<SmallBuf> for Vec<u8> {
    #[inline(always)]
    fn eq(&self, other: &SmallBuf) -> bool {
        self.as_slice() == other.as_slice()
    }
}

// Debug and Display traits
impl std::fmt::Debug for SmallBuf {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SmallBuf({:?})", self.as_slice())
    }
}

impl std::fmt::Display for SmallBuf {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02x?}", self.as_slice())
    }
}

// Default trait - empty buffer
impl Default for SmallBuf {
    #[inline(always)]
    fn default() -> Self {
        SmallBuf::buf_8()
    }
}

// Borrow traits for HashMap/HashSet keys
impl core::borrow::Borrow<[u8]> for SmallBuf {
    #[inline(always)]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl core::borrow::BorrowMut<[u8]> for SmallBuf {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}
