//! [`BStackDeque<T>`]: an owned double-ended queue over a contiguous ring.
//!
//! The on-disk answer to [`std::collections::VecDeque`], and the container most
//! callers reaching for [`crate::BStackLinkedList`] actually want. Its elements'
//! references live in **one contiguous ring block** — `[u64; cap]` slots indexed
//! circularly — so traversing the structure is a single sequential scan rather
//! than a pointer chase per element (each *value* still lives in its own block, so
//! resolving a value seeks once, but finding the next element does not).
//!
//! Push/pop at **both** ends are O(1) amortized. The ring's `head`/`len`/`cap`
//! and its data pointer live in the fixed handle block, so the handle never
//! moves; growth reallocates only the ring.
//!
//! # Single-ref slots, non-generic ring
//!
//! Like [`crate::BStackLinkedList`], each slot is a **single `u64` reference** to
//! an ordinary `T` block the deque owns — not `T` inlined. So the ring's on-disk
//! shape is the same for every `T` (a plain `u64` array), the handle layout
//! [`DequeOnDisk`] is non-generic, and only the tag varies by element type.
//!
//! # Atomicity
//!
//! Every push/pop is atomic per call and external-lock-free on the fast path: the
//! `head`/`len`/`cap`/`data` metadata and the target slot are read *and* written
//! inside one [`bstack::BStack::inplace_gen`] run (see
//! [`atomic_update`](super::util::atomic_update)), so a concurrent writer never
//! observes a torn ring and a crash never corrupts it. **Growth** is also atomic
//! and consistent: the new ring is allocated first (an orphan), then one
//! `inplace_gen` snapshots every live element, copies it into the new ring, and
//! swaps the descriptor — all under bstack's write lock, so it composes correctly
//! with concurrent pushes (a push that finds the ring full simply grows and
//! retries). A crash mid-growth leaks a ring, never tears the deque.

use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::size_of;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::{BStack, BStackRange};
use bytemuck::{Pod, Zeroable};

use super::util::{WriteBuf, alloc_image, atomic_update, read_fields, read_u64, w8};
use crate::clone::{ClonePlan, TryCloneIn};
use crate::handback::ReplaceError;
use crate::io_core::teardown::dealloc_range;
use crate::primitives::EightCC;
use crate::types::compiled::block::{BlockHeader, HEADER_SIZE};
use crate::types::compiled::owned::BStackOwned;
use crate::types::traits::block::{BStackBlock, BStackCast};
use crate::types::traits::drop::BStackDrop;
use crate::util::small_buf::SmallBuf;

/// The on-disk image of a [`BStackDeque`]: the block header, a pointer to the
/// ring data block (`0` = none), its capacity in slots, and the circular
/// `head`/`len`. `#[repr(C)]` with only `u64` fields after the header, so it is
/// padding-free and **non-generic** — the same layout for every element type.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DequeOnDisk {
    /// The 16-byte block header (size + type tag).
    pub header: BlockHeader,
    /// Offset of the ring data block (`[u64; cap]`), or `0` when unallocated.
    pub data: u64,
    /// Number of slots in the ring.
    pub cap: u64,
    /// Index (into the ring) of the front element.
    pub head: u64,
    /// Number of elements currently held.
    pub len: u64,
}

// Field offsets within the handle block.
const DATA_OFF: u64 = HEADER_SIZE; // 16
const CAP_OFF: u64 = HEADER_SIZE + 8; // 24
const HEAD_OFF: u64 = HEADER_SIZE + 16; // 32
const LEN_OFF: u64 = HEADER_SIZE + 24; // 40

const DEQUE_SIZE: u64 = size_of::<DequeOnDisk>() as u64;
/// The capacity a freshly grown empty ring starts at.
const MIN_CAP: u64 = 4;

fn overflow_err() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "deque offset overflow")
}

/// An owned double-ended queue of `T` blocks.
///
/// A typed handle (a newtype over a [`BStackRange`]); [`new`](Self::new) returns
/// a bare [`BStackOwned<BStackDeque<T>>`] that frees nothing on scope exit — free
/// it with [`bstack_drop`](BStackDrop::bstack_drop) or wrap it
/// ([`AutoDrop`] / [`crate::BStackCow`]).
///
/// The deque owns its elements' blocks: pushing takes a [`BStackOwned<T>`],
/// popping hands one back, and teardown recursively frees every element and the
/// ring.
pub struct BStackDeque<T: BStackBlock> {
    range: BStackRange,
    _marker: PhantomData<fn() -> T>,
}

impl<T: BStackBlock> BStackDeque<T> {
    fn value_size() -> u64 {
        size_of::<<T as BStackBlock>::OnDisk>() as u64
    }

    /// A `T`-value handle over the block at `off` (fixed-size-block model).
    fn value_at(off: u64) -> T {
        unsafe { <T as BStackBlock>::from_range(BStackRange::new(off, Self::value_size())) }
    }

    /// Read the four `(head, len, cap, data)` metadata fields of the handle at
    /// `handle` in a single I/O (the on-disk order is `data, cap, head, len`).
    fn read_meta(stack: &BStack, handle: u64) -> io::Result<(u64, u64, u64, u64)> {
        let [data, cap, head, len] = read_fields::<4>(stack, handle + DATA_OFF)?;
        Ok((head, len, cap, data))
    }

    /// The ring index for logical position `head + x` (mod `cap`), rejecting
    /// `cap == 0` — a corrupted on-disk capacity, otherwise a `%` panic
    /// unconditionally reachable from a pure read (`front`/`back`/`to_vec`/
    /// iteration) regardless of build mode. `head + x` wrapping before the
    /// reduction is intentional: only the reduced, `cap`-bounded index is ever
    /// used for an address.
    fn ring_index(head: u64, x: u64, cap: u64) -> io::Result<u64> {
        if cap == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt deque: zero capacity with a nonzero length",
            ));
        }
        Ok(head.wrapping_add(x) % cap)
    }

    /// The absolute ring-slot address for a (already `cap`-bounded) index,
    /// rejecting overflow — `data` can originate from a corrupted on-disk
    /// pointer.
    fn slot_addr(data: u64, idx: u64) -> io::Result<u64> {
        let delta = idx
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "deque offset overflow"))?;
        data.checked_add(delta)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "deque offset overflow"))
    }

    /// Allocate an empty deque (no ring is allocated until the first push).
    pub fn new<A: BStackRaiiAllocator>(allocator: &A) -> io::Result<BStackOwned<Self>> {
        Self::with_image(allocator, 0, 0)
    }

    /// Allocate an empty deque with room for `cap` elements pre-reserved (so the
    /// first `cap` pushes never grow). `cap == 0` behaves like [`new`](Self::new).
    pub fn with_capacity<A: BStackRaiiAllocator>(
        allocator: &A,
        cap: u64,
    ) -> io::Result<BStackOwned<Self>> {
        if cap == 0 {
            return Self::new(allocator);
        }
        // Allocate the ring first (an orphan); its slots are empty (len == 0), so
        // their contents are never read before being written.
        let size = cap.saturating_mul(8);
        let ring = allocator.alloc(size)?.as_range().start();
        match Self::with_image(allocator, ring, cap) {
            Ok(owned) => Ok(owned),
            Err(e) => {
                // SAFETY: the ring was just allocated and linked to nothing.
                let _ = unsafe { dealloc_range(allocator, BStackRange::new(ring, size)) };
                Err(e)
            }
        }
    }

    fn with_image<A: BStackRaiiAllocator>(
        allocator: &A,
        data: u64,
        cap: u64,
    ) -> io::Result<BStackOwned<Self>> {
        let od = DequeOnDisk {
            header: BlockHeader {
                size: DEQUE_SIZE,
                tag: Self::eightcc(),
            },
            data,
            cap,
            head: 0,
            len: 0,
        };
        let range = alloc_image(allocator, bytemuck::bytes_of(&od))?;
        // SAFETY: a freshly allocated block owned by no other handle.
        Ok(unsafe { BStackOwned::from_raw(Self::from_range(range)) })
    }

    /// Number of elements.
    pub fn len(&self, stack: &BStack) -> io::Result<u64> {
        read_u64(stack, self.range.start() + LEN_OFF)
    }

    /// Whether the deque has no elements.
    pub fn is_empty(&self, stack: &BStack) -> io::Result<bool> {
        Ok(self.len(stack)? == 0)
    }

    /// Current ring capacity (elements storable before the next growth).
    pub fn capacity(&self, stack: &BStack) -> io::Result<u64> {
        read_u64(stack, self.range.start() + CAP_OFF)
    }

    /// Append a value to the back, taking ownership of its block. Grows the ring
    /// (once) if it is full, then commits the slot write + length bump atomically.
    pub fn push_back<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        value: BStackOwned<T>,
    ) -> Result<(), ReplaceError<BStackOwned<T>>> {
        let handle = self.range.start();
        // Guard the value block so a stray return can't orphan it; on a failed
        // push [`finish_handback`] returns it to the caller rather than freeing it
        //, and defuses the guard on success once it is linked.
        let value = value.auto(allocator);
        let val_off = value.range().start();
        let outcome: io::Result<()> = (|| {
            loop {
                let full = Cell::new(false);
                let mut w: WriteBuf<2> = WriteBuf::new();
                atomic_update(
                    allocator,
                    &[
                        handle + HEAD_OFF,
                        handle + LEN_OFF,
                        handle + CAP_OFF,
                        handle + DATA_OFF,
                    ],
                    |_v1| Ok(Vec::new()),
                    |v1, _v2| {
                        let (head, len, cap, data) = (v1[0], v1[1], v1[2], v1[3]);
                        if len < cap {
                            // `len < cap` (unsigned) already implies `cap > 0`.
                            let idx = head.wrapping_add(len) % cap;
                            let slot = Self::slot_addr(data, idx)?;
                            w.push(w8(slot, val_off));
                            w.push(w8(handle + LEN_OFF, len + 1));
                        } else {
                            full.set(true);
                        }
                        Ok(w.as_slice())
                    },
                )?;
                if !full.get() {
                    return Ok(());
                }
                self.grow(allocator)?;
            }
        })();
        value.finish_handback(outcome)
    }

    /// Prepend a value to the front, taking ownership of its block.
    pub fn push_front<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        value: BStackOwned<T>,
    ) -> Result<(), ReplaceError<BStackOwned<T>>> {
        let handle = self.range.start();
        // Guard the value block so a stray return can't orphan it; on a failed
        // push [`finish_handback`] returns it to the caller rather than freeing it
        //, and defuses the guard on success once it is linked.
        let value = value.auto(allocator);
        let val_off = value.range().start();
        let outcome: io::Result<()> = (|| {
            loop {
                let full = Cell::new(false);
                let mut w: WriteBuf<3> = WriteBuf::new();
                atomic_update(
                    allocator,
                    &[
                        handle + HEAD_OFF,
                        handle + LEN_OFF,
                        handle + CAP_OFF,
                        handle + DATA_OFF,
                    ],
                    |_v1| Ok(Vec::new()),
                    |v1, _v2| {
                        let (head, len, cap, data) = (v1[0], v1[1], v1[2], v1[3]);
                        if len < cap {
                            // `len < cap` (unsigned) already implies `cap > 0`.
                            let idx = head.wrapping_add(cap - 1) % cap;
                            let slot = Self::slot_addr(data, idx)?;
                            w.push(w8(slot, val_off));
                            w.push(w8(handle + HEAD_OFF, idx));
                            w.push(w8(handle + LEN_OFF, len + 1));
                        } else {
                            full.set(true);
                        }
                        Ok(w.as_slice())
                    },
                )?;
                if !full.get() {
                    return Ok(());
                }
                self.grow(allocator)?;
            }
        })();
        value.finish_handback(outcome)
    }

    /// Remove and return the last element (as an owned value block), or `None` if
    /// empty. Atomic: the slot is read and the length decremented in one commit;
    /// the value block's ownership transfers to the caller (its ring slot is left
    /// stale and reused by a later push).
    pub fn pop_back<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
    ) -> io::Result<Option<BStackOwned<T>>> {
        let handle = self.range.start();
        let got = Cell::new(false);
        let val = Cell::new(0u64);
        let mut w: WriteBuf<1> = WriteBuf::new();
        atomic_update(
            allocator,
            &[
                handle + HEAD_OFF,
                handle + LEN_OFF,
                handle + CAP_OFF,
                handle + DATA_OFF,
            ],
            |v1| {
                let (head, len, cap, data) = (v1[0], v1[1], v1[2], v1[3]);
                if len == 0 {
                    Ok(Vec::new())
                } else {
                    Ok(vec![Self::slot_addr(
                        data,
                        Self::ring_index(head, len - 1, cap)?,
                    )?])
                }
            },
            |v1, v2| {
                let len = v1[1];
                if len != 0 {
                    got.set(true);
                    val.set(v2[0]);
                    w.push(w8(handle + LEN_OFF, len - 1));
                }
                Ok(w.as_slice())
            },
        )?;
        if !got.get() {
            return Ok(None);
        }
        // SAFETY: the value block's ownership transfers to the caller.
        Ok(Some(unsafe {
            BStackOwned::from_raw(Self::value_at(val.get()))
        }))
    }

    /// Remove and return the first element (as an owned value block), or `None`
    /// if empty.
    pub fn pop_front<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
    ) -> io::Result<Option<BStackOwned<T>>> {
        let handle = self.range.start();
        let got = Cell::new(false);
        let val = Cell::new(0u64);
        let mut w: WriteBuf<2> = WriteBuf::new();
        atomic_update(
            allocator,
            &[
                handle + HEAD_OFF,
                handle + LEN_OFF,
                handle + CAP_OFF,
                handle + DATA_OFF,
            ],
            |v1| {
                let (head, len, cap, data) = (v1[0], v1[1], v1[2], v1[3]);
                if len == 0 {
                    Ok(Vec::new())
                } else {
                    Ok(vec![Self::slot_addr(
                        data,
                        Self::ring_index(head, 0, cap)?,
                    )?])
                }
            },
            |v1, v2| {
                let (head, len, cap) = (v1[0], v1[1], v1[2]);
                if len != 0 {
                    got.set(true);
                    val.set(v2[0]);
                    w.push(w8(handle + HEAD_OFF, Self::ring_index(head, 1, cap)?));
                    w.push(w8(handle + LEN_OFF, len - 1));
                }
                Ok(w.as_slice())
            },
        )?;
        if !got.get() {
            return Ok(None);
        }
        // SAFETY: the value block's ownership transfers to the caller.
        Ok(Some(unsafe {
            BStackOwned::from_raw(Self::value_at(val.get()))
        }))
    }

    /// Grow the ring to at least double its capacity, atomically snapshotting and
    /// re-basing the live elements. A no-op (beyond a wasted allocation, freed
    /// again) if another thread already made room.
    fn grow<A: BStackRaiiAllocator>(&self, allocator: &A) -> io::Result<()> {
        let handle = self.range.start();
        let cap0 = read_u64(allocator.stack(), handle + CAP_OFF)?;
        let newcap = if cap0 == 0 {
            MIN_CAP
        } else {
            cap0.saturating_mul(2)
        };
        // Allocate the new ring up front (an orphan until the commit swaps to it).
        let new_size = newcap
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "deque capacity overflow"))?;
        let newring = allocator.alloc(new_size)?.as_range().start();

        let grown = Cell::new(false);
        let old_ring = Cell::new(0u64);
        let old_cap = Cell::new(0u64);

        // Abort if, at commit time, the ring already has room or is already at
        // least this big (another thread grew it) — then our `newring` is wasted.
        let abort = |_head: u64, len: u64, cap: u64| (cap != 0 && len < cap) || newcap <= cap;
        let mut w: Vec<(u64, SmallBuf)> = Vec::new();

        atomic_update(
            allocator,
            &[
                handle + HEAD_OFF,
                handle + LEN_OFF,
                handle + CAP_OFF,
                handle + DATA_OFF,
            ],
            |v1| {
                let (head, len, cap, data) = (v1[0], v1[1], v1[2], v1[3]);
                if abort(head, len, cap) {
                    Ok(Vec::new())
                } else {
                    // The live elements, in logical order.
                    (0..len)
                        .map(|i| Self::slot_addr(data, Self::ring_index(head, i, cap)?))
                        .collect::<io::Result<Vec<u64>>>()
                }
            },
            |v1, v2| {
                let (head, len, cap, data) = (v1[0], v1[1], v1[2], v1[3]);
                if !abort(head, len, cap) {
                    grown.set(true);
                    old_ring.set(data);
                    old_cap.set(cap);
                    w.reserve(v2.len() + 3);
                    // Copy every live element to the front of the new ring.
                    for (i, &r) in v2.iter().enumerate() {
                        let off = newring
                            .checked_add((i as u64).checked_mul(8).ok_or_else(overflow_err)?)
                            .ok_or_else(overflow_err)?;
                        w.push(w8(off, r));
                    }
                    // Swap the descriptor to the new ring, re-based at head 0.
                    w.push(w8(handle + DATA_OFF, newring));
                    w.push(w8(handle + CAP_OFF, newcap));
                    w.push(w8(handle + HEAD_OFF, 0u64));
                }
                Ok(w.as_slice())
            },
        )?;

        if grown.get() {
            // The old ring is now unreferenced; free it (leak-only on crash).
            if old_cap.get() > 0 {
                // SAFETY: the descriptor no longer points at the old ring.
                let _ = unsafe {
                    dealloc_range(
                        allocator,
                        BStackRange::new(old_ring.get(), old_cap.get().saturating_mul(8)),
                    )
                };
            }
        } else {
            // Growth was unnecessary; reclaim the unused new ring.
            // SAFETY: `newring` was never linked into the descriptor.
            let _ = unsafe { dealloc_range(allocator, BStackRange::new(newring, new_size)) };
        }
        Ok(())
    }

    /// A **borrowed** handle to the front value (no ownership), or `None` if empty.
    pub fn front(&self, stack: &BStack) -> io::Result<Option<T>> {
        let (head, len, cap, data) = Self::read_meta(stack, self.range.start())?;
        if len == 0 {
            return Ok(None);
        }
        let off = Self::slot_addr(data, Self::ring_index(head, 0, cap)?)?;
        Ok(Some(Self::value_at(read_u64(stack, off)?)))
    }

    /// A **borrowed** handle to the back value (no ownership), or `None` if empty.
    pub fn back(&self, stack: &BStack) -> io::Result<Option<T>> {
        let (head, len, cap, data) = Self::read_meta(stack, self.range.start())?;
        if len == 0 {
            return Ok(None);
        }
        let off = Self::slot_addr(data, Self::ring_index(head, len - 1, cap)?)?;
        Ok(Some(Self::value_at(read_u64(stack, off)?)))
    }

    /// Collect **borrowed** handles to every value, front to back. The handles
    /// alias the deque's blocks — do not free them; they stay valid only while the
    /// deque does.
    pub fn to_vec(&self, stack: &BStack) -> io::Result<Vec<T>> {
        let (head, len, cap, data) = Self::read_meta(stack, self.range.start())?;
        // Untrusted `len`: each element occupies 8 ring bytes, so bound it by the
        // stack before sizing an allocation with it (as `string.rs` does).
        let ring_bytes = len
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "deque length overflow"))?;
        if ring_bytes > stack.len()? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "deque length larger than the stack",
            ));
        }
        let mut out = Vec::with_capacity(len as usize);
        for i in 0..len {
            let off = Self::slot_addr(data, Self::ring_index(head, i, cap)?)?;
            out.push(Self::value_at(read_u64(stack, off)?));
        }
        Ok(out)
    }

    /// A lazy iterator over the elements, front to back, yielding `io::Result`
    /// value handles. A read snapshot: do not mutate the deque while iterating.
    pub fn iter<'a>(&self, stack: &'a BStack) -> io::Result<DequeIter<'a, T>> {
        let (head, len, cap, data) = Self::read_meta(stack, self.range.start())?;
        Ok(DequeIter {
            stack,
            block_off: self.range.start(),
            data,
            cap,
            head,
            len,
            pos: 0,
            _marker: PhantomData,
        })
    }
}

impl<T: BStackBlock> BStackCast for BStackDeque<T> {
    /// A `"Deq"` prefix over hash bytes perturbed by `T`'s tag, so deques of
    /// different element types never share a discriminant despite the identical
    /// on-disk layout.
    fn eightcc() -> EightCC {
        const BASE: EightCC = EightCC::new([b'D', b'e', b'q', 0x80, 0x81, 0x82, 0x83, 0x84]);
        BASE.mix(<T as BStackCast>::eightcc())
    }
}

// Self-contained (no separate control block): may be `#[embed]`ded.
impl<T: BStackBlock> crate::types::traits::embed::BStackEmbeddable for BStackDeque<T> {}

impl<T: BStackBlock> BStackBlock for BStackDeque<T> {
    type OnDisk = DequeOnDisk;

    unsafe fn from_range(range: BStackRange) -> Self {
        BStackDeque {
            range,
            _marker: PhantomData,
        }
    }

    fn range(&self) -> BStackRange {
        self.range
    }

    /// Recursively free every element block and the ring, **without** freeing the
    /// handle block itself.
    fn __bstack_drop_children<A: BStackRaiiAllocator>(
        allocator: &A,
        range: BStackRange,
    ) -> io::Result<()> {
        let (head, len, cap, data) = Self::read_meta(allocator.stack(), range.start())?;
        for i in 0..len {
            let off = Self::slot_addr(data, Self::ring_index(head, i, cap)?)?;
            let r = read_u64(allocator.stack(), off)?;
            if r != 0 {
                // SAFETY: the deque solely owns each element block.
                let owned = unsafe { BStackOwned::from_raw(Self::value_at(r)) };
                owned.bstack_drop(allocator)?;
            }
        }
        if data != 0 {
            let size = cap.checked_mul(8).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "deque capacity overflow")
            })?;
            // SAFETY: the deque solely owns its ring block.
            unsafe { dealloc_range(allocator, BStackRange::new(data, size))? };
        }
        Ok(())
    }

    /// Deep-clone the deque into `plan`: every element is deep-cloned (via `T`'s
    /// own clone hook) and packed into a fresh, compacted ring (`head = 0`,
    /// `cap = len`); the handle block is staged — all in the parent plan's single
    /// atomic commit.
    fn __bstack_clone_children_inplace<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        plan: &mut ClonePlan,
    ) -> io::Result<Self::OnDisk> {
        let (head, len, cap, data) = Self::read_meta(allocator.stack(), self.range.start())?;

        // Untrusted `len`: checked math + stack bound before sizing allocations
        // (see `to_vec`); `ring_bytes` is also the fresh ring's size below.
        let ring_bytes = len
            .checked_mul(8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "deque length overflow"))?;
        if ring_bytes > allocator.stack().len()? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "deque length larger than the stack",
            ));
        }
        // Deep-clone each element (in logical order) into the plan.
        let mut dsts = Vec::with_capacity(len as usize);
        for i in 0..len {
            let off = Self::slot_addr(data, Self::ring_index(head, i, cap)?)?;
            let r = read_u64(allocator.stack(), off)?;
            let dst = if r != 0 {
                Self::value_at(r)
                    .__bstack_clone_into(allocator, plan)?
                    .start()
            } else {
                0
            };
            dsts.push(dst);
        }

        // Pack the cloned refs into a fresh, exactly-sized ring.
        let (new_data, new_cap) = if len > 0 {
            let ring = plan.alloc_raw(allocator, ring_bytes)?;
            let mut bytes = Vec::with_capacity(dsts.len() * 8);
            for d in &dsts {
                bytes.extend_from_slice(&d.to_le_bytes());
            }
            plan.write(ring.start(), bytes);
            (ring.start(), len)
        } else {
            (0, 0)
        };

        // Return the descriptor pointing at the freshly-cloned ring; the caller
        // places it — the default `__bstack_clone_into` in its own fresh block, or an
        // `#[embed]`ding parent inline in its payload. (Returning the descriptor
        // *verbatim* — the trait default — would alias the source's ring, so an
        // embedded collection MUST override this.)
        Ok(DequeOnDisk {
            header: BlockHeader {
                size: DEQUE_SIZE,
                tag: Self::eightcc(),
            },
            data: new_data,
            cap: new_cap,
            head: 0,
            len,
        })
    }
}

impl<T: BStackBlock> TryCloneIn for BStackDeque<T> {
    fn try_clone_in<A: BStackRaiiAllocator>(&self, allocator: &A) -> io::Result<BStackOwned<Self>> {
        let mut plan = ClonePlan::new();
        let dst = match self.__bstack_clone_into(allocator, &mut plan) {
            Ok(range) => range,
            Err(e) => {
                plan.rollback(allocator);
                return Err(e);
            }
        };
        plan.commit(allocator)?;
        // SAFETY: `dst` is a fresh block owned by nobody else.
        Ok(unsafe { BStackOwned::from_raw(Self::from_range(dst)) })
    }
}

/// A front-to-back iterator over a [`BStackDeque`], yielding `io::Result<T>`
/// value handles. Created by [`BStackDeque::iter`].
pub struct DequeIter<'a, T: BStackBlock> {
    stack: &'a BStack,
    /// The deque handle block, re-read each step to detect mutation.
    block_off: u64,
    data: u64,
    cap: u64,
    head: u64,
    len: u64,
    pos: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T: BStackBlock> Iterator for DequeIter<'a, T> {
    type Item = io::Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        // Fail fast if the deque was mutated during iteration. A `grow` frees
        // the old ring and repoints `data`, but a `push`/`pop` mutates *in place*:
        // it advances `head` / changes `len` and hands out (or frees) an element
        // block **without moving the ring or clearing the vacated slot**, so
        // checking `data`/`cap` alone would miss it and later read a stale offset a
        // `T` handle could then free (use-after-free). Compare all four snapshot
        // fields — pure iteration never touches them, so any change is a mutation —
        // and turn it into a clean `InvalidData` error.
        match BStackDeque::<T>::read_meta(self.stack, self.block_off) {
            Ok((head, len, cap, data))
                if data == self.data && cap == self.cap && head == self.head && len == self.len => {
            }
            Ok(_) => {
                self.pos = self.len;
                return Some(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "BStackDeque was mutated during iteration (its head/len/ring \
                     changed); the iterator is invalidated",
                )));
            }
            Err(e) => {
                self.pos = self.len;
                return Some(Err(e));
            }
        }
        let slot = match BStackDeque::<T>::ring_index(self.head, self.pos, self.cap)
            .and_then(|idx| BStackDeque::<T>::slot_addr(self.data, idx))
        {
            Ok(off) => off,
            Err(e) => {
                self.pos = self.len; // stop after an error
                return Some(Err(e));
            }
        };
        self.pos += 1;
        match read_u64(self.stack, slot) {
            Ok(vref) => Some(Ok(BStackDeque::<T>::value_at(vref))),
            Err(e) => {
                self.pos = self.len; // stop after an error
                Some(Err(e))
            }
        }
    }
}
