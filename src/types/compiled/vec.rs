//! Persistent, growable vectors reached through a **fixed-size** field, with the
//! descriptor stored **inline** in the owning struct.
//!
//! A block field can only store a fixed-size value, but a vector's backing store
//! must grow — and `BStackByteVec` **moves** its block on realloc. The solution is a
//! small [`VecDesc`] (`{ data_off, data_size }`) that names the current data
//! block; the field stores it, and on growth only the descriptor is rewritten:
//!
//! ```text
//! struct field: [ data_off, data_size ] ── points to ──▶ data block (may realloc/move)
//! ```
//!
//! Because a struct **uniquely owns** its vector, the descriptor lives *inline*
//! in the field — there is no separate descriptor block, no extra indirection or
//! allocation. A vector not resident in a field (built by [`BStackVec::from_slice`]
//! or handed out by `bstack_move!`) carries its descriptor **in memory** in the
//! handle, and becomes persistent only when written into a field (which stamps
//! the inline descriptor). A field handle remembers its inline location and
//! rewrites it whenever a push reallocates.
//!
//! Elements are `bytemuck::Pod`; bytes are stored/read unaligned. `u8` (`Vec<u8>`
//! / `String` fields) is the common case. Block-element vectors
//! ([`BStackBlockVec`] / [`BStackStrongVec`] / [`BStackWeakVec`] /
//! [`BStackRefVec`]) store a `u64` offset per element in the same way.
//!
//! > **Growth reallocates**, so use a realloc-safe allocator (see the crate
//! > docs) to avoid corruption on a torn realloc.
//!
//! **Concurrency:** a field-resident [`push`](BStackVec::push) runs the whole
//! read → append/grow → commit under the file's WAL lock, re-reading the shared
//! inline descriptor inside it, so concurrent pushes through independent handles
//! (each `get_<field>()` mints a fresh one) never write into a freed block or
//! double-free a displaced ring. A detached vector has no shared
//! descriptor and appends lock-free.

use core::marker::PhantomData;
use core::mem::size_of;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::{BStack, BStackByteVec, BStackOwnedSlice, BStackRange};
use bytemuck::{Pod, Zeroable};

use super::super::traits::block::BStackBlock;
use super::super::traits::drop::BStackDrop;
use super::super::traits::rc::{BStackShared, BStackWeakable};
use super::super::traits::reference::BStackRef;
use super::owned::BStackOwned;
use super::rc::WeakRef;
use super::rc::{BStackRc, BStackWeak};
use crate::clone::ClonePlan;
use crate::handback::ReplaceError;
use crate::io_core::teardown::dealloc_range;
use crate::primitives::{NonNullOffset, Offset};
use crate::util::bytes::{get_u64, put_u64};

/// The on-disk header length of a `BStackByteVec` block: `len: u64` @ 0,
/// `cap: u64` @ 8, elements from offset 16. Fixed by bstack's ABI (stable across
/// `0.4.x`). Used where we build a byte-vec block image by hand to keep a
/// mutation crash-atomic.
pub(crate) const BYTEVEC_HEADER: u64 = 16;

/// Build a `BStackByteVec` block image `[len@0 | cap@8 | data@16]` by hand — the
/// single place that on-disk shape is assembled, shared by a cloned vec
/// ([`crate::ClonePlan::stage_bytevec`]) and a field-resident growth
/// [`push`](BStackVec::push).
// NOTE: suspecious pub(crate). This is because clone plan still use it, we will refactor later
pub(crate) fn bytevec_image(len: u64, cap: u64, data: &[u8]) -> Vec<u8> {
    let mut img = vec![0u8; BYTEVEC_HEADER as usize + data.len()];
    put_u64(&mut img, 0, len);
    put_u64(&mut img, 8, cap);
    img[BYTEVEC_HEADER as usize..].copy_from_slice(data);
    img
}

/// Build a fresh data block holding `offs` (an offset array), register it in
/// `plan` for rollback, and return its descriptor. The shared back end of the
/// block-element vector clones, whose elements are all `u64` offsets.
fn build_offset_desc<A: BStackRaiiAllocator>(
    allocator: &A,
    offs: &[u64],
    plan: &mut ClonePlan,
) -> io::Result<VecDesc> {
    // Fold the offset-array block into the plan: allocated through our machinery,
    // its bytes committed in the plan's single atomic batch.
    plan.stage_bytevec(allocator, bytemuck::cast_slice(offs))
}

/// The inline, fixed-size descriptor of a persistent vector: the current offset
/// and byte size of its (reallocating) data block.
///
/// Stored **inline** in the owning struct's field — there is no separate
/// descriptor block, since the struct uniquely owns the vector. `Pod`, so it
/// embeds directly in a generated `XOnDisk` (`data_off` is an [`Offset`], which is
/// itself `#[repr(transparent)]` over `u64`, so the wire layout is unchanged and
/// `data_off == 0` stays the "no data block" / `Option<Vec>` niche).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VecDesc {
    /// Offset of the current data block, or [`Offset::NULL`] (`0`) when there is
    /// no block yet (an empty / `None` vector).
    pub data_off: Offset,
    pub data_size: u64,
}

impl VecDesc {
    /// Read a [`VecDesc`] from an absolute on-disk location (its inline field slot).
    fn read(stack: &BStack, loc: u64) -> io::Result<VecDesc> {
        let mut buf = [0u8; 16];
        stack.get_into(loc, &mut buf)?;
        Ok(VecDesc::from(buf))
    }
}

impl From<[u8; 16]> for VecDesc {
    /// Decode the 16-byte little-endian on-disk image (`data_off` @0, `data_size` @8).
    #[inline]
    fn from(buf: [u8; 16]) -> VecDesc {
        VecDesc {
            data_off: Offset::from_raw(get_u64(&buf[0..8])),
            data_size: get_u64(&buf[8..16]),
        }
    }
}

impl From<VecDesc> for [u8; 16] {
    /// The 16-byte little-endian on-disk image — the form a field slot stores and
    /// the descriptor CAS in [`BStackVec::push`] compares against.
    #[inline]
    fn from(desc: VecDesc) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&desc.data_off.get().to_le_bytes());
        buf[8..16].copy_from_slice(&desc.data_size.to_le_bytes());
        buf
    }
}

impl From<BStackRange> for VecDesc {
    /// A descriptor naming `range` as the data block.
    #[inline]
    fn from(range: BStackRange) -> VecDesc {
        VecDesc {
            data_off: Offset::from(range),
            data_size: range.len(),
        }
    }
}

impl From<VecDesc> for BStackRange {
    /// The data block range this descriptor names (`[data_off, data_off + data_size)`).
    #[inline]
    fn from(desc: VecDesc) -> BStackRange {
        BStackRange::new(desc.data_off.get(), desc.data_size)
    }
}

// No `Deref`/`AsRef`/`AsMut<BStackRange>`: `VecDesc` does not *contain* a
// `BStackRange` (it is `Offset` + a `u64` size), so there is nothing to borrow —
// the `From` conversions above are the value-level bridge instead.

/// A persistent, growable vector of POD elements. Backs un-annotated `Vec<T>`
/// (`T: Pod`) / `String` fields.
///
/// The handle carries the descriptor in memory (`data`), plus the inline field
/// location to persist it to (`writeback`) when field-resident — `None` for a
/// detached vector (from [`from_slice`](Self::from_slice) or `bstack_move!`).
pub struct BStackVec<'a, T, A: BStackRaiiAllocator> {
    /// The current data block range (the live descriptor).
    data: BStackRange,
    /// Where to persist descriptor changes on realloc (the inline field). `None`
    /// for a detached vector (in-memory descriptor only).
    writeback: Option<BStackRange>,
    allocator: &'a A,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T, A: BStackRaiiAllocator> BStackVec<'a, T, A> {
    /// Reconstruct a **field-resident** handle from its inline descriptor's
    /// absolute on-disk location (what a field accessor passes). Reads the
    /// current descriptor and remembers the location for write-back.
    ///
    /// # Safety
    /// `loc` must be the offset of a live inline [`VecDesc`] written by this type.
    pub unsafe fn from_field(loc: NonNullOffset, allocator: &'a A) -> io::Result<Self> {
        let desc = VecDesc::read(allocator.stack(), loc.as_u64())?;
        Ok(Self {
            data: BStackRange::from(desc),
            writeback: Some(BStackRange::new(loc.as_u64(), size_of::<VecDesc>() as u64)),
            allocator,
            _marker: PhantomData,
        })
    }

    /// Like [`from_field`](Self::from_field), but for a nullable field: a
    /// `data_off` of `0` (the offset-0 niche, since no allocation lives there)
    /// reads as `None`. Backs `Option<Vec<T>>` accessors.
    ///
    /// # Safety
    /// As [`from_field`](Self::from_field).
    pub unsafe fn from_field_opt(loc: NonNullOffset, allocator: &'a A) -> io::Result<Option<Self>> {
        let desc = VecDesc::read(allocator.stack(), loc.as_u64())?;
        if desc.data_off.is_null() {
            return Ok(None);
        }
        Ok(Some(Self {
            data: BStackRange::from(desc),
            writeback: Some(BStackRange::new(loc.as_u64(), size_of::<VecDesc>() as u64)),
            allocator,
            _marker: PhantomData,
        }))
    }

    /// Reconstruct a **detached** handle from a descriptor value (no write-back;
    /// the descriptor lives only in memory). Used by `bstack_move!`.
    ///
    /// # Safety
    ///
    /// `desc` must be a descriptor written by this element type over a live data
    /// block owned by `allocator` that no other live handle will also free —
    /// the same contract as [`from_field`](Self::from_field), asserted about the
    /// descriptor's *contents* rather than its location. A fabricated descriptor
    /// lets safe methods (`bstack_drop`, element reads) free or reinterpret an
    /// arbitrary range.
    pub unsafe fn from_desc(desc: VecDesc, allocator: &'a A) -> Self {
        Self {
            data: BStackRange::from(desc),
            writeback: None,
            allocator,
            _marker: PhantomData,
        }
    }

    /// The current descriptor value — what a field stores inline.
    pub fn descriptor(&self) -> VecDesc {
        VecDesc::from(self.data)
    }

    /// The allocator this vector is bound to.
    pub fn allocator(&self) -> &'a A {
        self.allocator
    }

    /// Persist the current descriptor to the inline field, if field-resident.
    fn persist(&self) -> io::Result<()> {
        if let Some(loc) = self.writeback {
            self.allocator
                .stack()
                .set(loc.start(), <[u8; 16]>::from(self.descriptor()))?;
        }
        Ok(())
    }

    /// Reconstruct the `BStackByteVec` over the current data block.
    fn bytes(&self) -> io::Result<BStackByteVec<'a, A>> {
        let block = unsafe { BStackOwnedSlice::from_raw_range(self.allocator, self.data) };
        Ok(unsafe { BStackByteVec::from_raw_block(block) })
    }

    /// Free the data block. Consumes the handle. (There is no descriptor block;
    /// a field's inline descriptor is freed with the owning struct's block.)
    pub fn bstack_drop(self) -> io::Result<()> {
        unsafe { dealloc_range(self.allocator, self.data) }
    }
}

impl<'a, T: Pod, A: BStackRaiiAllocator> BStackVec<'a, T, A> {
    /// Create a **detached** vector holding `data`, allocating only the data
    /// block. It becomes persistent when written into a struct field.
    pub fn from_slice(allocator: &'a A, data: &[T]) -> io::Result<Self> {
        let data_range = BStackByteVec::from_slice(bytemuck::cast_slice(data), allocator)?
            .into_raw_block()
            .as_range();
        Ok(Self {
            data: data_range,
            writeback: None,
            allocator,
            _marker: PhantomData,
        })
    }

    /// Create an empty detached vector.
    pub fn new(allocator: &'a A) -> io::Result<Self> {
        Self::from_slice(allocator, &[])
    }

    /// Number of elements.
    pub fn len(&self) -> io::Result<u64> {
        let size = size_of::<T>() as u64;
        if size == 0 {
            // A zero-sized element (`T = ()`) occupies no bytes, so an element count
            // is not representable in the byte vector — it is definitionally empty.
            // Guard the division: `byte_len / 0` would panic from a safe, non-misuse
            // call.
            return Ok(0);
        }
        Ok(self.bytes()?.len()? / size)
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Read all elements into a `Vec<T>` (unaligned reads, so any `T` is fine).
    pub fn to_vec(&self) -> io::Result<Vec<T>> {
        let esz = size_of::<T>();
        if esz == 0 {
            // Zero-sized elements are definitionally empty (see `len`); guard the
            // `chunks_exact(0)` panic, matching `len`'s division guard.
            return Ok(Vec::new());
        }
        let bytes = self.bytes()?.read_bytes()?;
        Ok(bytes
            .chunks_exact(esz)
            .map(bytemuck::pod_read_unaligned::<T>)
            .collect())
    }

    /// Append an element, growing the data block if needed.
    ///
    /// A **detached** vec (no live on-disk descriptor) appends in place; the
    /// element is committed atomically (bytes into spare capacity, then one `len`
    /// bump), so a crash never leaves a partial element.
    ///
    /// A **field-resident** vec runs the whole read → append/grow → commit under
    /// the file's WAL lock. The field's inline descriptor is
    /// shared on disk — every `get_<field>()` mints a fresh handle over it, so two
    /// threads can push through independent handles — and without the lock a
    /// within-capacity append could write into a block a concurrent grow just
    /// freed, or a stale-snapshot commit could clobber a concurrent grow's
    /// descriptor (a double free of the displaced block). Holding the lock and
    /// re-reading the descriptor inside it makes the operation atomic; grow itself
    /// is allocate → commit descriptor → free old, so a crash mid-grow leaks at
    /// worst, never dangles.
    pub fn push(&mut self, value: T) -> io::Result<()> {
        if self.writeback.is_some() {
            // Serialize field-resident pushes on this file, and re-read the
            // descriptor inside the lock — our in-memory snapshot may be stale.
            let lock = crate::io_core::wal::wal_lock_for(self.allocator);
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let loc = self.writeback.expect("field-resident vec has a writeback");
            self.data =
                VecDesc::read(self.allocator.stack(), loc.start()).map(BStackRange::from)?;
            self.push_inner(value)
        } else {
            self.push_inner(value)
        }
    }

    /// The append/grow body of [`push`](Self::push). For a field-resident vec the
    /// caller holds the WAL lock and has re-read `self.data`; for a detached vec
    /// there is no shared descriptor to race.
    fn push_inner(&mut self, value: T) -> io::Result<()> {
        let bytevec = self.bytes()?;
        let elem = size_of::<T>() as u64;
        let len = bytevec.len()?;
        let cap = bytevec.capacity()?;
        // `len` is an on-disk field; a corrupted value near `u64::MAX` must not
        // silently wrap the fits-check (which would then size the grow path's new
        // block from the *wrapped* `new_len` while still `stack.copy`-ing the
        // original, huge `len` into it below).
        let new_len = len
            .checked_add(elem)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "vector length overflow"))?;

        if new_len <= cap || self.writeback.is_none() {
            let mut bytevec = bytevec;
            // Append the whole element as **one** crash-atomic unit:
            // `extend_from_slice` reserves (writing into unobservable spare capacity),
            // then commits `len += size_of::<T>()` in a single write — so a crash or
            // `Err` mid-element never commits a `len` that isn't an element multiple
            // (which would misalign every later element and, for the offset-storing
            // block vectors, tear a `u64` child pointer into a garbage range).
            bytevec.extend_from_slice(bytemuck::bytes_of(&value))?;
            self.data = bytevec.into_raw_block().as_range();
            return self.persist();
        }

        // Field-resident growth: allocate a new block, copy the old elements over
        // with the crash-atomic `BStack::copy` (no materialising), append the new
        // element, commit the descriptor, then free the old block.
        let new_cap = core::cmp::max(cap.saturating_mul(2), new_len);
        let old = self.data;

        let slice = self.allocator.alloc(BYTEVEC_HEADER + new_cap)?;
        let new_range = slice.as_range();
        let stack = self.allocator.stack();
        let build = (|| -> io::Result<()> {
            stack.set(new_range.start(), bytevec_image(new_len, new_cap, &[]))?;
            if len > 0 {
                stack.copy(
                    old.start() + BYTEVEC_HEADER,
                    new_range.start() + BYTEVEC_HEADER,
                    len,
                )?;
            }
            stack.set(
                new_range.start() + BYTEVEC_HEADER + len,
                bytemuck::bytes_of(&value),
            )
        })();
        if let Err(e) = build {
            let _ = self.allocator.dealloc(slice);
            return Err(e);
        }

        // Commit: repoint the (in-memory + inline) descriptor at the new block.
        // Safe under the caller's WAL lock — no concurrent handle can observe the
        // old descriptor after this and re-free `old`.
        self.data = new_range;
        self.persist()?;
        // Reclaim the old block (a crash before here leaks it; never dangles).
        unsafe { dealloc_range(self.allocator, old)? };
        Ok(())
    }

    /// Deep-clone this POD vector's data into a fresh block for a [`ClonePlan`]:
    /// copy every element into a new data block, register it for rollback, and
    /// return its descriptor (what the cloned owner stores inline). The new block
    /// is written eagerly by the vector runtime, not staged in the plan's batch.
    pub fn clone_data_into(&self, plan: &mut ClonePlan) -> io::Result<VecDesc> {
        // Fold the data block into the plan: read the source elements, then let
        // the plan allocate + stage a fresh block so it rides the atomic commit.
        // (Staging is intentional — the clone commits as one unit — so this does
        // NOT use `BStack::copy`, which would land outside the batch.)
        let bytes = self.bytes()?.read_bytes()?;
        plan.stage_bytevec(self.allocator, &bytes)
    }
}

// ---------------------------------------------------------------------------
// Block-element vectors: one `u64` offset per element, stored the same way. The
// field annotation states the elements' ownership; the descriptor + offset array
// are always owned by the enclosing struct.
// ---------------------------------------------------------------------------

/// A persistent, growable vector of **owned block children**.
///
/// Each element is a `u64` offset to a separately-allocated `#[bstack_block]`
/// child this vector *owns*; dropping the vector recursively frees every child
/// (post-order) plus the offset array. Backs `#[bstack_owned] Vec<Thing>` fields.
pub struct BStackBlockVec<'a, T: BStackBlock, A: BStackRaiiAllocator> {
    offsets: BStackVec<'a, u64, A>,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: BStackBlock, A: BStackRaiiAllocator> BStackBlockVec<'a, T, A> {
    /// # Safety
    /// `loc` must be a live inline descriptor over an array of data offsets to
    /// live `T` blocks this vector owns.
    pub unsafe fn from_field(loc: NonNullOffset, allocator: &'a A) -> io::Result<Self> {
        Ok(Self {
            offsets: unsafe { BStackVec::from_field(loc, allocator)? },
            _marker: PhantomData,
        })
    }

    /// Like [`from_field`](Self::from_field), but nullable — `None` when the
    /// inline descriptor is the offset-0 niche. Backs `Option<Vec<Thing>>`.
    ///
    /// # Safety
    /// As [`from_field`](Self::from_field).
    pub unsafe fn from_field_opt(loc: NonNullOffset, allocator: &'a A) -> io::Result<Option<Self>> {
        Ok(
            unsafe { BStackVec::from_field_opt(loc, allocator)? }.map(|offsets| Self {
                offsets,
                _marker: PhantomData,
            }),
        )
    }

    /// Reconstruct a detached handle from a descriptor value. Used by `bstack_move!`.
    ///
    /// # Safety
    ///
    /// As [`BStackVec::from_desc`]: `desc` must be a descriptor written by this
    /// element kind over a live data block owned by `allocator` that no other
    /// live handle will also free.
    pub unsafe fn from_desc(desc: VecDesc, allocator: &'a A) -> Self {
        Self {
            offsets: unsafe { BStackVec::from_desc(desc, allocator) },
            _marker: PhantomData,
        }
    }

    /// The current descriptor value — what a field stores inline.
    pub fn descriptor(&self) -> VecDesc {
        self.offsets.descriptor()
    }

    /// Number of elements.
    pub fn len(&self) -> io::Result<u64> {
        self.offsets.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> io::Result<bool> {
        self.offsets.is_empty()
    }

    fn elem_range(off: u64) -> BStackRange {
        BStackRange::new(off, size_of::<T::OnDisk>() as u64)
    }

    /// Read all element handles (non-owning views; the vector still owns them).
    pub fn to_vec(&self) -> io::Result<Vec<T>> {
        Ok(self
            .offsets
            .to_vec()?
            .into_iter()
            .map(|off| unsafe { T::from_range(Self::elem_range(off)) })
            .collect())
    }

    /// The element at index `i`, or `None` if out of range.
    pub fn get(&self, i: u64) -> io::Result<Option<T>> {
        Ok(self
            .offsets
            .to_vec()?
            .get(i as usize)
            .map(|&off| unsafe { T::from_range(Self::elem_range(off)) }))
    }

    /// Reconstruct the owned children from the offsets a failed build collected,
    /// so they can be handed back rather than freed.
    fn recover_owned(offs: &[u64]) -> Vec<BStackOwned<T>> {
        offs.iter()
            // SAFETY: each `off` names a live child this vec just consumed and did
            // not free; reconstructing one owner apiece is sound.
            .map(|&off| unsafe { BStackOwned::from_raw(T::from_range(Self::elem_range(off))) })
            .collect()
    }

    /// Build a detached vector from a list of owned children (each consumed).
    ///
    /// On failure the children are handed back through
    /// [`ReplaceError`](crate::ReplaceError) (reconstructed from the offsets the
    /// build collected) rather than freed, so the caller keeps their subtree data
    /// to retry or free.
    pub fn from_handles(
        allocator: &'a A,
        children: Vec<BStackOwned<T>>,
    ) -> Result<Self, ReplaceError<Vec<BStackOwned<T>>>> {
        let offs: Vec<u64> = children
            .into_iter()
            .map(|c| c.into_inner().range().start())
            .collect();
        match BStackVec::from_slice(allocator, &offs) {
            Ok(offsets) => Ok(Self {
                offsets,
                _marker: PhantomData,
            }),
            Err(e) => Err(ReplaceError::recovered(e, Self::recover_owned(&offs))),
        }
    }

    /// Create an empty detached vector.
    pub fn new(allocator: &'a A) -> io::Result<Self> {
        // No children to hand back, so a failure is a plain I/O error.
        Self::from_handles(allocator, Vec::new()).map_err(ReplaceError::into_source)
    }

    /// Append an owned child, transferring its ownership into the vector.
    ///
    /// On failure the child is handed back through
    /// [`ReplaceError`](crate::ReplaceError) rather than freed.
    pub fn push_owned(
        &mut self,
        child: BStackOwned<T>,
    ) -> Result<(), ReplaceError<BStackOwned<T>>> {
        let off = child.into_inner().range().start();
        if let Err(e) = self.offsets.push(off) {
            // Push failed after `child`'s ownership was moved in (`into_inner`
            // defused its RAII drop). Hand the child back rather than freeing it.
            // SAFETY: `off` names the live child whose ownership just moved in.
            let child = unsafe { BStackOwned::from_raw(T::from_range(Self::elem_range(off))) };
            return Err(ReplaceError::recovered(e, child));
        }
        Ok(())
    }

    /// Recursively free every owned child (post-order), then the offset array.
    /// Consumes the handle.
    pub fn bstack_drop(self) -> io::Result<()> {
        let allocator = self.offsets.allocator();
        for off in self.offsets.to_vec()? {
            // SAFETY: each stored offset names a live child this vec owns.
            unsafe {
                crate::types::traits::drop::drop_block::<T, A>(allocator, Self::elem_range(off))?
            };
        }
        self.offsets.bstack_drop()
    }

    /// Deep-clone this owned vector into a fresh block for a [`ClonePlan`]:
    /// deep-clone every child via `clone_elem` (which recurses the child into the
    /// plan and returns its new block range), then build a new offset array over
    /// the fresh children. The per-element callback is supplied by codegen so it
    /// can name the concrete child type's `__bstack_clone_into`.
    pub fn clone_into<F>(&self, plan: &mut ClonePlan, mut clone_elem: F) -> io::Result<VecDesc>
    where
        F: FnMut(BStackRange, &mut ClonePlan) -> io::Result<BStackRange>,
    {
        let allocator = self.offsets.allocator();
        let mut new_offs = Vec::new();
        for off in self.offsets.to_vec()? {
            let new_block = clone_elem(Self::elem_range(off), plan)?;
            new_offs.push(new_block.start());
        }
        build_offset_desc(allocator, &new_offs, plan)
    }
}

/// A persistent, growable vector of **strong references** to shared block
/// children (`(rc)` / `(rc, weak)` blocks).
///
/// Each element holds one strong reference; dropping the vector releases every
/// one (freeing a child when its count hits zero) and frees the offset array.
/// Backs `#[bstack_strong] Vec<Thing>` fields.
pub struct BStackStrongVec<'a, T: BStackShared, A: BStackRaiiAllocator> {
    offsets: BStackVec<'a, u64, A>,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: BStackShared, A: BStackRaiiAllocator> BStackStrongVec<'a, T, A> {
    /// # Safety
    /// `loc` must be a live inline descriptor over an array of data offsets to
    /// live `T` blocks, each accounting for one strong reference this vector owns.
    pub unsafe fn from_field(loc: NonNullOffset, allocator: &'a A) -> io::Result<Self> {
        Ok(Self {
            offsets: unsafe { BStackVec::from_field(loc, allocator)? },
            _marker: PhantomData,
        })
    }

    /// Like [`from_field`](Self::from_field), but nullable — `None` when the
    /// inline descriptor is the offset-0 niche. Backs `Option<Vec<Thing>>`.
    ///
    /// # Safety
    /// As [`from_field`](Self::from_field).
    pub unsafe fn from_field_opt(loc: NonNullOffset, allocator: &'a A) -> io::Result<Option<Self>> {
        Ok(
            unsafe { BStackVec::from_field_opt(loc, allocator)? }.map(|offsets| Self {
                offsets,
                _marker: PhantomData,
            }),
        )
    }

    /// Reconstruct a detached handle from a descriptor value. Used by `bstack_move!`.
    ///
    /// # Safety
    ///
    /// As [`BStackVec::from_desc`]: `desc` must be a descriptor written by this
    /// element kind over a live data block owned by `allocator` that no other
    /// live handle will also free.
    pub unsafe fn from_desc(desc: VecDesc, allocator: &'a A) -> Self {
        Self {
            offsets: unsafe { BStackVec::from_desc(desc, allocator) },
            _marker: PhantomData,
        }
    }

    /// The current descriptor value — what a field stores inline.
    pub fn descriptor(&self) -> VecDesc {
        self.offsets.descriptor()
    }

    /// Number of elements.
    pub fn len(&self) -> io::Result<u64> {
        self.offsets.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> io::Result<bool> {
        self.offsets.is_empty()
    }

    fn elem_range(off: u64) -> BStackRange {
        BStackRange::new(off, size_of::<T::OnDisk>() as u64)
    }

    /// Read all element handles (non-owning views — the vector still holds the
    /// strong references).
    pub fn to_vec(&self) -> io::Result<Vec<T>> {
        Ok(self
            .offsets
            .to_vec()?
            .into_iter()
            .map(|off| unsafe { T::from_range(Self::elem_range(off)) })
            .collect())
    }

    /// The element at index `i`, or `None` if out of range.
    pub fn get(&self, i: u64) -> io::Result<Option<T>> {
        Ok(self
            .offsets
            .to_vec()?
            .get(i as usize)
            .map(|&off| unsafe { T::from_range(Self::elem_range(off)) }))
    }

    /// Build a detached vector from a list of strong handles (each consumed, its
    /// strong count moved into the vector).
    ///
    /// On failure the strong handles are handed back through
    /// [`ReplaceError`](crate::ReplaceError) (reconstructed from the consumed data
    /// offset + control range, so no count is dropped) rather than released.
    pub fn from_handles(
        allocator: &'a A,
        elems: Vec<BStackRc<'a, T, A>>,
    ) -> Result<Self, ReplaceError<Vec<BStackRc<'a, T, A>>>> {
        // Keep each element's `(data offset, control range)` so a failed build can
        // reconstruct the exact strong handle it consumed.
        let parts: Vec<(u64, Option<BStackRange>)> = elems
            .into_iter()
            .map(|rc| {
                let (data, ctrl) = rc.into_raw();
                (data.into_range().start(), ctrl)
            })
            .collect();
        let offs: Vec<u64> = parts.iter().map(|(off, _)| *off).collect();
        match BStackVec::from_slice(allocator, &offs) {
            Ok(offsets) => Ok(Self {
                offsets,
                _marker: PhantomData,
            }),
            Err(e) => {
                let recovered = parts
                    .into_iter()
                    .map(|(off, ctrl)| {
                        let data = unsafe { BStackRef::<T>::from_range(Self::elem_range(off)) };
                        // SAFETY: `data`/`ctrl` name the block whose strong count
                        // this element still holds (`into_raw` transferred it in).
                        unsafe { BStackRc::from_raw(data, ctrl, allocator) }
                    })
                    .collect();
                Err(ReplaceError::recovered(e, recovered))
            }
        }
    }

    /// Append a strong reference (consumed, its count moved into the vector).
    ///
    /// On failure the strong handle is handed back through
    /// [`ReplaceError`](crate::ReplaceError) rather than released.
    pub fn push_strong(
        &mut self,
        elem: BStackRc<'a, T, A>,
    ) -> Result<(), ReplaceError<BStackRc<'a, T, A>>> {
        let (data, ctrl) = elem.into_raw();
        let off = data.into_range().start();
        if let Err(e) = self.offsets.push(off) {
            // Push failed after `elem` was consumed (its count moved in by
            // `into_raw`). Hand the strong handle back rather than release it.
            let data = unsafe { BStackRef::<T>::from_range(Self::elem_range(off)) };
            // SAFETY: `data`/`ctrl` name the block whose strong count `elem` still holds.
            let rc = unsafe { BStackRc::from_raw(data, ctrl, self.offsets.allocator()) };
            return Err(ReplaceError::recovered(e, rc));
        }
        Ok(())
    }

    /// Release every strong reference (freeing children that reach zero), then
    /// free the offset array. Consumes the handle.
    pub fn bstack_drop(self) -> io::Result<()> {
        let allocator = self.offsets.allocator();
        for off in self.offsets.to_vec()? {
            let data = unsafe { BStackRef::<T>::from_range(Self::elem_range(off)) };
            T::drop_strong_ref(data, allocator)?;
        }
        self.offsets.bstack_drop()
    }

    /// Clone this strong vector into a fresh block for a [`ClonePlan`]: the shared
    /// children are re-referenced, not copied — bump each element's strong count
    /// and keep its data offset, then build a new offset array over the same
    /// targets.
    pub fn clone_into(&self, plan: &mut ClonePlan) -> io::Result<VecDesc> {
        let allocator = self.offsets.allocator();
        let offs = self.offsets.to_vec()?;
        for &off in &offs {
            let data = unsafe { BStackRef::<T>::from_range(Self::elem_range(off)) };
            plan.bump_strong(data, allocator)?;
        }
        build_offset_desc(allocator, &offs, plan)
    }
}

/// A persistent, growable vector of **weak references** to `(rc, weak)` block
/// children.
///
/// Each element holds one weak reference (a stored control-block offset).
/// Dropping the vector releases every weak count (freeing a control block when
/// it reaches zero) and frees the offset array. Backs `#[bstack_weak] Vec<Thing>`
/// fields.
pub struct BStackWeakVec<'a, T: BStackWeakable, A: BStackRaiiAllocator> {
    offsets: BStackVec<'a, u64, A>,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: BStackWeakable, A: BStackRaiiAllocator> BStackWeakVec<'a, T, A> {
    /// # Safety
    /// `loc` must be a live inline descriptor over an array of control-block
    /// offsets, each accounting for one weak reference this vector owns.
    pub unsafe fn from_field(loc: NonNullOffset, allocator: &'a A) -> io::Result<Self> {
        Ok(Self {
            offsets: unsafe { BStackVec::from_field(loc, allocator)? },
            _marker: PhantomData,
        })
    }

    /// Like [`from_field`](Self::from_field), but nullable — `None` when the
    /// inline descriptor is the offset-0 niche. Backs `Option<Vec<Thing>>`.
    ///
    /// # Safety
    /// As [`from_field`](Self::from_field).
    pub unsafe fn from_field_opt(loc: NonNullOffset, allocator: &'a A) -> io::Result<Option<Self>> {
        Ok(
            unsafe { BStackVec::from_field_opt(loc, allocator)? }.map(|offsets| Self {
                offsets,
                _marker: PhantomData,
            }),
        )
    }

    /// Reconstruct a detached handle from a descriptor value. Used by `bstack_move!`.
    ///
    /// # Safety
    ///
    /// As [`BStackVec::from_desc`]: `desc` must be a descriptor written by this
    /// element kind over a live data block owned by `allocator` that no other
    /// live handle will also free.
    pub unsafe fn from_desc(desc: VecDesc, allocator: &'a A) -> Self {
        Self {
            offsets: unsafe { BStackVec::from_desc(desc, allocator) },
            _marker: PhantomData,
        }
    }

    /// The current descriptor value — what a field stores inline.
    pub fn descriptor(&self) -> VecDesc {
        self.offsets.descriptor()
    }

    /// Number of elements.
    pub fn len(&self) -> io::Result<u64> {
        self.offsets.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> io::Result<bool> {
        self.offsets.is_empty()
    }

    fn ctrl_ref(off: u64) -> BStackRef<T::Control> {
        unsafe { BStackRef::from_range(BStackRange::new(off, size_of::<T::Control>() as u64)) }
    }

    /// Attempt to upgrade the element at index `i` to a strong handle. `None` if
    /// out of range, or if the target's strong count has already reached zero.
    pub fn upgrade(&self, i: u64) -> io::Result<Option<BStackRc<'a, T, A>>> {
        let offs = self.offsets.to_vec()?;
        let Some(&off) = offs.get(i as usize) else {
            return Ok(None);
        };
        // Borrow a weak over the element's control ref just long enough to
        // upgrade; consume it via `into_raw` so the vector's own weak count is
        // untouched.
        let allocator = self.offsets.allocator();
        let weak = unsafe { BStackWeak::from_raw(Self::ctrl_ref(off), allocator) };
        let result = weak.upgrade();
        let _ = weak.into_raw();
        result
    }

    /// Build a detached vector from a list of weak handles (each consumed, its
    /// weak count moved into the vector).
    ///
    /// On failure the weak handles are handed back through
    /// [`ReplaceError`](crate::ReplaceError) (reconstructed from the consumed
    /// control offsets, so no count is dropped) rather than released.
    pub fn from_handles(
        allocator: &'a A,
        elems: Vec<BStackWeak<'a, T, A>>,
    ) -> Result<Self, ReplaceError<Vec<BStackWeak<'a, T, A>>>> {
        let offs: Vec<u64> = elems
            .into_iter()
            .map(|w| w.into_raw().into_range().start())
            .collect();
        match BStackVec::from_slice(allocator, &offs) {
            Ok(offsets) => Ok(Self {
                offsets,
                _marker: PhantomData,
            }),
            Err(e) => {
                let recovered = offs
                    .iter()
                    // SAFETY: each `off` carries the weak count its consumed
                    // `BStackWeak` transferred in above.
                    .map(|&off| unsafe { BStackWeak::from_raw(Self::ctrl_ref(off), allocator) })
                    .collect();
                Err(ReplaceError::recovered(e, recovered))
            }
        }
    }

    /// Append a weak reference (consumed, its count moved into the vector).
    ///
    /// On failure the weak handle is handed back through
    /// [`ReplaceError`](crate::ReplaceError) rather than released.
    pub fn push_weak(
        &mut self,
        elem: BStackWeak<'a, T, A>,
    ) -> Result<(), ReplaceError<BStackWeak<'a, T, A>>> {
        let ctrl = elem.into_raw();
        let off = ctrl.into_range().start();
        if let Err(e) = self.offsets.push(off) {
            // Push failed after `elem` was consumed (its decrement defused by
            // `into_raw`). Hand the weak handle back rather than release its count.
            // SAFETY: `off` carries the weak count `elem` transferred in.
            let weak =
                unsafe { BStackWeak::from_raw(Self::ctrl_ref(off), self.offsets.allocator()) };
            return Err(ReplaceError::recovered(e, weak));
        }
        Ok(())
    }

    /// Release every weak reference (freeing control blocks that reach zero),
    /// then free the offset array. Consumes the handle.
    pub fn bstack_drop(self) -> io::Result<()> {
        let allocator = self.offsets.allocator();
        for off in self.offsets.to_vec()? {
            // SAFETY: each stored offset carries one weak count owned by this vec.
            unsafe { WeakRef::<T>::new(Self::ctrl_ref(off)) }.bstack_drop(allocator)?;
        }
        self.offsets.bstack_drop()
    }

    /// Clone this weak vector into a fresh block for a [`ClonePlan`]: bump each
    /// element's weak count and keep its control offset, then build a new offset
    /// array over the same control blocks.
    pub fn clone_into(&self, plan: &mut ClonePlan) -> io::Result<VecDesc> {
        let allocator = self.offsets.allocator();
        let offs = self.offsets.to_vec()?;
        for &off in &offs {
            plan.bump_weak(off)?;
        }
        build_offset_desc(allocator, &offs, plan)
    }
}

/// A persistent, growable vector of **raw references** to block children.
///
/// Elements carry no ownership: dropping the vector frees only the offset array,
/// never the targets. Backs `#[bstack_ref] Vec<Thing>` fields.
pub struct BStackRefVec<'a, T: BStackBlock, A: BStackRaiiAllocator> {
    offsets: BStackVec<'a, u64, A>,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: BStackBlock, A: BStackRaiiAllocator> BStackRefVec<'a, T, A> {
    /// # Safety
    /// `loc` must be a live inline descriptor over an array of offsets to `T`
    /// blocks (which this vector does not own).
    pub unsafe fn from_field(loc: NonNullOffset, allocator: &'a A) -> io::Result<Self> {
        Ok(Self {
            offsets: unsafe { BStackVec::from_field(loc, allocator)? },
            _marker: PhantomData,
        })
    }

    /// Like [`from_field`](Self::from_field), but nullable — `None` when the
    /// inline descriptor is the offset-0 niche. Backs `Option<Vec<Thing>>`.
    ///
    /// # Safety
    /// As [`from_field`](Self::from_field).
    pub unsafe fn from_field_opt(loc: NonNullOffset, allocator: &'a A) -> io::Result<Option<Self>> {
        Ok(
            unsafe { BStackVec::from_field_opt(loc, allocator)? }.map(|offsets| Self {
                offsets,
                _marker: PhantomData,
            }),
        )
    }

    /// Reconstruct a detached handle from a descriptor value. Used by `bstack_move!`.
    ///
    /// # Safety
    ///
    /// As [`BStackVec::from_desc`]: `desc` must be a descriptor written by this
    /// element kind over a live data block owned by `allocator` that no other
    /// live handle will also free.
    pub unsafe fn from_desc(desc: VecDesc, allocator: &'a A) -> Self {
        Self {
            offsets: unsafe { BStackVec::from_desc(desc, allocator) },
            _marker: PhantomData,
        }
    }

    /// The current descriptor value — what a field stores inline.
    pub fn descriptor(&self) -> VecDesc {
        self.offsets.descriptor()
    }

    /// Number of elements.
    pub fn len(&self) -> io::Result<u64> {
        self.offsets.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> io::Result<bool> {
        self.offsets.is_empty()
    }

    fn elem_range(off: u64) -> BStackRange {
        BStackRange::new(off, size_of::<T::OnDisk>() as u64)
    }

    /// Read all element handles (non-owning views).
    pub fn to_vec(&self) -> io::Result<Vec<T>> {
        Ok(self
            .offsets
            .to_vec()?
            .into_iter()
            .map(|off| unsafe { T::from_range(Self::elem_range(off)) })
            .collect())
    }

    /// The element at index `i`, or `None` if out of range.
    pub fn get(&self, i: u64) -> io::Result<Option<T>> {
        Ok(self
            .offsets
            .to_vec()?
            .get(i as usize)
            .map(|&off| unsafe { T::from_range(Self::elem_range(off)) }))
    }

    /// Build a detached vector from a list of raw references.
    pub fn from_handles(allocator: &'a A, elems: Vec<BStackRef<T>>) -> io::Result<Self> {
        let offs: Vec<u64> = elems.into_iter().map(|r| r.into_range().start()).collect();
        Ok(Self {
            offsets: BStackVec::from_slice(allocator, &offs)?,
            _marker: PhantomData,
        })
    }

    /// Append a raw reference.
    pub fn push_ref(&mut self, elem: BStackRef<T>) -> io::Result<()> {
        self.offsets.push(elem.into_range().start())
    }

    /// Free only the offset array (elements are not owned). Consumes the handle.
    pub fn bstack_drop(self) -> io::Result<()> {
        self.offsets.bstack_drop()
    }

    /// Clone this ref vector into a fresh block for a [`ClonePlan`]: the elements
    /// are non-owning, so copy the offset array verbatim (the clone aliases the
    /// same targets).
    pub fn clone_into(&self, plan: &mut ClonePlan) -> io::Result<VecDesc> {
        let allocator = self.offsets.allocator();
        let offs = self.offsets.to_vec()?;
        build_offset_desc(allocator, &offs, plan)
    }
}
