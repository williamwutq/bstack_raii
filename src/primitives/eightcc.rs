//! [`EightCC`] — the 8-byte per-type tag stamped in every block header.

use bytemuck::{Pod, Zeroable};

/// An 8-byte type tag stored in every [`BlockHeader`](crate::BlockHeader).
///
/// Used instead of a 4-byte `FourCC` because `bstack` offsets are 64-bit, so
/// 8-byte alignment is natural. [`crate::BStackCast`] compares it during safe
/// downcasts, so it must be unique per block type.
///
/// The `#[bstack_block]` macro generates it as a readable ASCII prefix followed
/// by the high-bit-set tail of a hash of the crate + type name (so distinct
/// types stay distinct even with a shared prefix) — see the macro docs.
/// [`EightCC::from_name`] is the simpler truncating form, for manual use.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Pod, Zeroable)]
pub struct EightCC(pub [u8; 8]);

impl EightCC {
    /// Wrap a raw 8-byte tag.
    #[inline(always)]
    pub const fn new(tag: [u8; 8]) -> Self {
        Self(tag)
    }

    /// Derive a tag from a type name: the first 8 bytes, zero-padded (or
    /// truncated). `const` so the macro can emit it in a `const` context.
    #[inline]
    pub const fn from_name(name: &str) -> Self {
        let bytes = name.as_bytes();
        let mut out = [0u8; 8];
        let mut i = 0;
        while i < 8 && i < bytes.len() {
            out[i] = bytes[i];
            i += 1;
        }
        Self(out)
    }

    /// Fold another tag into this one's non-readable (high-bit-set) bytes,
    /// leaving the leading ASCII prefix untouched. A generic `#[bstack_block]`
    /// uses this to give each instantiation a distinct-but-related tag — the
    /// readable prefix stays the outer type's, while the type arguments perturb
    /// the hash bytes so `Foo<A>` and `Foo<B>` never share a discriminant (which
    /// would let `bstack_cast!` confuse them). Deterministic, and composes for
    /// nested generics.
    ///
    /// **Order-sensitive.** The FNV digest is seeded with `self`'s *current* bytes
    /// before folding `other`, so a later `mix` sees the state a prior one produced:
    /// `base.mix(A).mix(B) != base.mix(B).mix(A)`. (Folding only `other` would perturb
    /// via a plain XOR, which is commutative — `Foo<A,B>` and `Foo<B,A>` would then
    /// collide to one tag, and the tag is the *sole* `bstack_cast!` / on-disk type
    /// identity, so the cast could not tell them apart.)
    ///
    /// Note: a fully-specified 8-byte explicit `tag = "…"` leaves no hash bytes,
    /// so every instantiation shares it — don't pin an 8-byte tag on a generic.
    pub const fn mix(self, other: EightCC) -> EightCC {
        // FNV-1a over `self`'s bytes *then* `other`'s — seeding with the running tag
        // makes the fold order-sensitive (see above), not XOR-commutative.
        let mut d: u64 = 0xcbf2_9ce4_8422_2325;
        let sb = self.0;
        let mut i = 0;
        while i < 8 {
            d ^= sb[i] as u64;
            d = d.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        let ob = other.0;
        let mut i = 0;
        while i < 8 {
            d ^= ob[i] as u64;
            d = d.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        let db = d.to_le_bytes();
        let mut out = self.0;
        let mut i = 0;
        while i < 8 {
            if out[i] & 0x80 != 0 {
                out[i] = (out[i] ^ db[i]) | 0x80;
            }
            i += 1;
        }
        Self(out)
    }

    /// Fold an arbitrary string into this tag's non-readable (high-bit-set) hash
    /// bytes, leaving the leading ASCII prefix untouched — the same domain `mix`
    /// perturbs. The generated `eightcc()` uses this to fold the type's
    /// `module_path!()` into the tag, so two same-named types in *different
    /// modules* of one crate get distinct tags (the crate + bare identifier hash
    /// alone does not). Deterministic; FNV-1a over
    /// `self`'s current bytes then the string's, matching `mix`'s seeding.
    pub const fn mix_str(self, s: &str) -> EightCC {
        let mut d: u64 = 0xcbf2_9ce4_8422_2325;
        let sb = self.0;
        let mut i = 0;
        while i < 8 {
            d ^= sb[i] as u64;
            d = d.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            d ^= bytes[i] as u64;
            d = d.wrapping_mul(0x0000_0100_0000_01b3);
            i += 1;
        }
        let db = d.to_le_bytes();
        let mut out = self.0;
        let mut i = 0;
        while i < 8 {
            if out[i] & 0x80 != 0 {
                out[i] = (out[i] ^ db[i]) | 0x80;
            }
            i += 1;
        }
        Self(out)
    }

    /// Derive a control-block tag from this (data) tag by toggling one reserved
    /// bit in the trailing hash byte. The result differs from the data tag in
    /// exactly that bit, so `data_tag != ctrl_tag` is guaranteed **structurally**,
    /// regardless of what the readable prefix contains.
    ///
    /// Bit 6 (`0x40`) is chosen because `mix` / `mix_str` force bit 7 (`0x80`) and
    /// never touch bit 6, so the toggle survives generic mixing.
    #[inline(always)]
    pub const fn with_ctrl_bit(self) -> EightCC {
        let mut out = self.0;
        out[7] ^= 0x40;
        Self(out)
    }
}
