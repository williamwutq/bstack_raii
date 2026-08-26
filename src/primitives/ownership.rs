//! [`OwnershipKind`] — how a reference relates to the block it points at.

/// The **ownership relationship** a reference has with its target — the crate's
/// four in-file reference kinds, the axis that selects teardown / deep-clone /
/// refcount behaviour for a field, a variant payload, or a cross-file
/// [`Foreign`](crate::Foreign) pointer.
///
/// This is the single vocabulary the whole crate branches on; the derive's field
/// classifier, the RTTI interpreter, and the `Foreign` teardown helpers all express
/// the same four cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OwnershipKind {
    /// **Owned** (`#[bstack_owned]`): sole ownership. Torn down (freed) and
    /// deep-cloned with its target.
    Owned = 0,
    /// **Strong** (`#[bstack_strong]`): a shared, refcounted reference. Teardown
    /// decrements the strong count (freeing at zero); clone increments it.
    Strong = 1,
    /// **Weak** (`#[bstack_weak]`): a non-owning refcounted reference to a control
    /// block. Teardown decrements the weak count; never keeps the target alive.
    Weak = 2,
    /// **Ref** (`#[bstack_ref]`): a plain borrow — a bare offset with no ownership.
    /// Teardown and clone leave the target untouched.
    Ref = 3,
}

impl OwnershipKind {
    /// All four kinds, in declaration order.
    pub const ALL: [OwnershipKind; 4] = [
        OwnershipKind::Owned,
        OwnershipKind::Strong,
        OwnershipKind::Weak,
        OwnershipKind::Ref,
    ];

    /// The kind's stable 1-byte discriminant (`Owned=0, Strong=1, Weak=2, Ref=3`) —
    /// the on-disk / wire encoding.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decode a kind from its [`as_u8`](Self::as_u8) discriminant, or `None` for an
    /// out-of-range byte (corrupt / forged data).
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(OwnershipKind::Owned),
            1 => Some(OwnershipKind::Strong),
            2 => Some(OwnershipKind::Weak),
            3 => Some(OwnershipKind::Ref),
            _ => None,
        }
    }

    /// Whether this kind keeps its target alive (frees or refcounts it): [`Owned`] or
    /// [`Strong`], as opposed to the non-owning [`Weak`] / [`Ref`].
    ///
    /// [`Owned`]: Self::Owned
    /// [`Strong`]: Self::Strong
    /// [`Weak`]: Self::Weak
    /// [`Ref`]: Self::Ref
    pub const fn is_owning(self) -> bool {
        matches!(self, OwnershipKind::Owned | OwnershipKind::Strong)
    }
}

impl From<OwnershipKind> for u8 {
    #[inline]
    fn from(k: OwnershipKind) -> u8 {
        k.as_u8()
    }
}

impl TryFrom<u8> for OwnershipKind {
    type Error = InvalidOwnershipKind;

    #[inline]
    fn try_from(v: u8) -> Result<Self, InvalidOwnershipKind> {
        OwnershipKind::from_u8(v).ok_or(InvalidOwnershipKind(v))
    }
}

/// The error [`OwnershipKind::try_from`] returns for a byte outside `0..=3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidOwnershipKind(pub u8);

impl core::fmt::Display for InvalidOwnershipKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid OwnershipKind discriminant: {}", self.0)
    }
}

impl std::error::Error for InvalidOwnershipKind {}
