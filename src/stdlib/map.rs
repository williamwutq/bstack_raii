//! [`BStackHashMap<K, V>`]: an owned open-addressing hash map.
//!
//! The on-disk answer to [`std::collections::HashMap`], and the way to look a
//! value up by key without a linear scan. Keys are **`Pod`** (`K: Pod`), stored
//! inline in the bucket and hashed by their raw bytes; values are blocks
//! (`V: BStackBlock`) the map owns, referenced by a single `u64` per bucket.
//!
//! # Layout
//!
//! The fixed handle block ([`MapOnDisk`]) holds a pointer to a **contiguous
//! bucket block**, the bucket count `cap` (a power of two), the live-entry count
//! `len`, and `used` (occupied + tombstone slots, which drives growth). Each
//! bucket is `state: u64` (`EMPTY` / `OCCUPIED` / `TOMBSTONE`), then the inline
//! key `K`, then a `u64` value reference — a stride of `16 + size_of::<K>()`
//! bytes. Probing is linear (`cap` a power of two, `mask = cap - 1`), so the
//! whole probe sequence is a contiguous scan, not a pointer chase.
//!
//! # Atomicity
//!
//! Each `insert` / `remove` is atomic per call and external-lock-free: the entire
//! probe *and* the resulting bucket + metadata writes run inside one
//! [`bstack::BStack::inplace_gen`] (see [`probe_commit`]), so a concurrent writer
//! never observes a torn table and a crash never corrupts it. **Growth / rehash**
//! is likewise atomic: a bigger bucket block is allocated first (an orphan), then
//! one `inplace_gen` snapshots every live bucket, rebuilds the table into the new
//! block (dropping tombstones), and swaps the descriptor — all under bstack's
//! write lock, composing with concurrent inserts (an insert that can't place
//! grows and retries). A crash mid-rehash leaks a bucket block, never tears the
//! map.
//!
//! `get` / `contains_key` are plain probes (no write lock); each bucket read is
//! atomic, but the probe is not linearized against a concurrent mutation, so a
//! borrowed value handle it returns is valid only while that entry is not removed.

use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::size_of;
use std::io;

use crate::BStackRaiiAllocator;
use bstack::{BStack, BStackRange};
use bytemuck::{Pod, Zeroable};

use super::hash::fnv1a;
use super::util::{
    Meta, ProbeStep, Scratch, alloc_image, grow_table, probe_commit, read_fields, w8,
};
use crate::handback::ReplaceError;
use crate::io_core::{ClonePlan, TryCloneIn, dealloc_range};
use crate::primitives::EightCC;
use crate::types::compiled::{BStackOwned, BlockHeader, HEADER_SIZE};
use crate::types::traits::{BStackBlock, BStackCast, BStackDrop};
use crate::util::{SmallBuf, get_u64, io_error, read_u64};

/// The on-disk image of a [`BStackHashMap`]: header, bucket-block pointer (`0` =
/// none), bucket count `cap`, live-entry count `len`, and `used` (occupied +
/// tombstone). `#[repr(C)]`, only `u64` fields after the header — non-generic.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MapOnDisk {
    /// The 16-byte block header (size + type tag).
    pub header: BlockHeader,
    /// Offset of the bucket block, or `0` when unallocated.
    pub table: u64,
    /// Number of buckets (a power of two).
    pub cap: u64,
    /// Number of live entries.
    pub len: u64,
    /// Occupied + tombstone slots (drives growth).
    pub used: u64,
}

// Field offsets within the handle block. `table..used` are contiguous so all
// four load in one 32-byte read.
const TABLE_OFF: u64 = HEADER_SIZE; // 16
const CAP_OFF: u64 = HEADER_SIZE + 8; // 24
const LEN_OFF: u64 = HEADER_SIZE + 16; // 32
const USED_OFF: u64 = HEADER_SIZE + 24; // 40

const MAP_SIZE: u64 = size_of::<MapOnDisk>() as u64;
/// Bucket count of a freshly grown empty table (a power of two).
const MIN_CAP: u64 = 4;

// Bucket states.
const EMPTY: u64 = 0;
const OCCUPIED: u64 = 1;
const TOMBSTONE: u64 = 2;

/// The per-insert invariants shared by the probe closures that place the new
/// entry: where the map lives, its bucket geometry, and the entry to write.
struct NewEntry<'a> {
    handle: u64,
    stride: u64,
    ksz: usize,
    key_bytes: &'a [u8],
    val_ref: u64,
}

/// Build the writes that place a *new* entry (state, key, value) at bucket
/// `target`, bumping `len` (and `used` when the slot was previously `EMPTY`).
fn new_bucket_writes(
    e: &NewEntry,
    m: &Meta,
    target: u64,
    slot_was_empty: bool,
) -> io::Result<Vec<(u64, SmallBuf)>> {
    let mut img = Vec::with_capacity(16 + e.ksz);
    img.extend_from_slice(&OCCUPIED.to_le_bytes());
    img.extend_from_slice(e.key_bytes);
    img.extend_from_slice(&e.val_ref.to_le_bytes());

    // `m.table` is an on-disk pointer that can be corrupted/forged.
    let off = target
        .checked_mul(e.stride)
        .and_then(|d| m.table.checked_add(d))
        .ok_or_else(|| io_error!("corrupt bucket table offset"))?;
    let mut w = vec![
        (off, SmallBuf::Heap(img.into_boxed_slice())),
        w8(e.handle + LEN_OFF, m.len + 1),
    ];
    if slot_was_empty {
        w.push(w8(e.handle + USED_OFF, m.used + 1));
    }
    Ok(w)
}

/// An owned open-addressing hash map from a `Pod` key to a block value.
///
/// A typed handle (a newtype over a [`BStackRange`]); [`new`](Self::new) returns
/// a bare [`BStackOwned<BStackHashMap<K, V>>`] that frees nothing on scope exit —
/// free it with [`bstack_drop`](BStackDrop::bstack_drop) or wrap it
/// ([`AutoDrop`] / [`crate::BStackCow`]).
///
/// The map owns its values' blocks: [`insert`](Self::insert) takes a
/// [`BStackOwned<V>`], [`remove`](Self::remove) hands one back (as does an
/// overwriting `insert`, returning the replaced value), and teardown recursively
/// frees every value and the bucket block.
pub struct BStackHashMap<K: Pod, V: BStackBlock> {
    range: BStackRange,
    _marker: PhantomData<fn() -> (K, V)>,
}

impl<K: Pod, V: BStackBlock> BStackHashMap<K, V> {
    /// Bytes of an inline key.
    fn ksize() -> usize {
        size_of::<K>()
    }

    /// Bytes of one bucket: `state (8) + key (ksize) + value ref (8)`.
    fn stride() -> u64 {
        16 + Self::ksize() as u64
    }

    /// On-disk size of one `V` value block.
    fn value_size() -> u64 {
        size_of::<<V as BStackBlock>::OnDisk>() as u64
    }

    /// A `V`-value handle over the block at `off`.
    fn value_at(off: u64) -> V {
        unsafe { <V as BStackBlock>::from_range(BStackRange::new(off, Self::value_size())) }
    }

    /// Allocate an empty map (no bucket block until the first insert).
    pub fn new<A: BStackRaiiAllocator>(allocator: &A) -> io::Result<BStackOwned<Self>> {
        let od = MapOnDisk {
            header: BlockHeader {
                size: MAP_SIZE,
                tag: Self::eightcc(),
            },
            table: 0,
            cap: 0,
            len: 0,
            used: 0,
        };
        let range = alloc_image(allocator, bytemuck::bytes_of(&od))?;
        // SAFETY: a freshly allocated block owned by no other handle.
        Ok(unsafe { BStackOwned::from_raw(Self::from_range(range)) })
    }

    /// Number of live entries.
    pub fn len(&self, stack: &BStack) -> io::Result<u64> {
        read_u64(stack, self.range.start() + LEN_OFF)
    }

    /// Whether the map has no entries.
    pub fn is_empty(&self, stack: &BStack) -> io::Result<bool> {
        Ok(self.len(stack)? == 0)
    }

    /// Insert `key -> value`, taking ownership of the value block. Returns the
    /// previously-mapped value (owned) if `key` was already present, else `None`.
    ///
    /// Atomic and external-lock-free: grows the table first if the load factor
    /// would be exceeded, then probes and commits the bucket + metadata writes in
    /// one [`probe_commit`].
    pub fn insert<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        key: K,
        value: BStackOwned<V>,
    ) -> Result<Option<BStackOwned<V>>, ReplaceError<BStackOwned<V>>> {
        let handle = self.range.start();
        let stride = Self::stride();
        let ksz = Self::ksize();
        let key_bytes = bytemuck::bytes_of(&key).to_vec();
        // Guard the value block: any fallible step below (`read_fields`, `grow`,
        // `probe_commit`) that errors must not orphan it. On failure
        // [`finish_handback`] returns it to the caller rather than freeing it
        //; on success it is defused once linked into the map.
        let value = value.auto(allocator);
        let val_ref = value.range().start();
        let hash = fnv1a(&key_bytes);
        let entry = NewEntry {
            handle,
            stride,
            ksz,
            key_bytes: &key_bytes,
            val_ref,
        };

        let outcome: io::Result<Option<BStackOwned<V>>> = (|| loop {
            // Proactively keep the load factor under 3/4 (also clears tombstones).
            let [cap, _len, used] = read_fields::<3>(allocator.stack(), handle + CAP_OFF)?;
            if cap == 0 || (used + 1) * 4 > cap * 3 {
                self.grow(allocator)?;
                continue;
            }

            let first_tomb: Cell<Option<u64>> = Cell::new(None);
            let is_new = Cell::new(false);
            let need_grow = Cell::new(false);
            let old_value = Cell::new(0u64);

            probe_commit(
                allocator,
                handle,
                stride,
                hash,
                |m, idx, buf| {
                    let state = get_u64(&buf[0..8]);
                    if state == EMPTY {
                        let target = first_tomb.get().unwrap_or(idx);
                        let slot_was_empty = first_tomb.get().is_none();
                        is_new.set(true);
                        ProbeStep::Stop(new_bucket_writes(&entry, m, target, slot_was_empty))
                    } else if state == OCCUPIED && buf[8..8 + ksz] == key_bytes[..] {
                        // Overwrite: replace the value ref, hand back the old one.
                        old_value.set(get_u64(&buf[8 + ksz..8 + ksz + 8]));
                        is_new.set(false);
                        // `m.table` is an on-disk pointer that can be corrupted/forged.
                        let step = idx
                            .checked_mul(stride)
                            .and_then(|d| m.table.checked_add(d))
                            .and_then(|b| b.checked_add(8 + ksz as u64))
                            .ok_or_else(|| io_error!("corrupt bucket table offset"))
                            .map(|value_off| vec![w8(value_off, val_ref)]);
                        ProbeStep::Stop(step)
                    } else {
                        if state == TOMBSTONE && first_tomb.get().is_none() {
                            first_tomb.set(Some(idx));
                        }
                        ProbeStep::Continue
                    }
                },
                |m| {
                    if let Some(t) = first_tomb.get() {
                        is_new.set(true);
                        new_bucket_writes(&entry, m, t, false)
                    } else {
                        need_grow.set(true);
                        Ok(Vec::new())
                    }
                },
            )?;

            if need_grow.get() {
                self.grow(allocator)?;
                continue;
            }
            // Committed: the value is now linked into the map.
            return if is_new.get() {
                Ok(None)
            } else {
                // SAFETY: the replaced value block's ownership transfers to the caller.
                Ok(Some(unsafe {
                    BStackOwned::from_raw(Self::value_at(old_value.get()))
                }))
            };
        })();
        value.finish_handback(outcome)
    }

    /// Remove `key`, returning its value (owned) if present, else `None`. The
    /// bucket becomes a tombstone; the value block's ownership transfers out.
    pub fn remove<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        key: &K,
    ) -> io::Result<Option<BStackOwned<V>>> {
        let handle = self.range.start();
        let stride = Self::stride();
        let ksz = Self::ksize();
        let key_bytes = bytemuck::bytes_of(key).to_vec();
        let hash = fnv1a(&key_bytes);

        let found = Cell::new(false);
        let old_value = Cell::new(0u64);

        probe_commit(
            allocator,
            handle,
            stride,
            hash,
            |m, idx, buf| {
                let state = get_u64(&buf[0..8]);
                if state == EMPTY {
                    ProbeStep::Stop(Ok(Vec::new())) // absent: commit nothing
                } else if state == OCCUPIED && buf[8..8 + ksz] == key_bytes[..] {
                    found.set(true);
                    old_value.set(get_u64(&buf[8 + ksz..8 + ksz + 8]));
                    // `m.table` is an on-disk pointer that can be corrupted/forged.
                    let step = idx
                        .checked_mul(stride)
                        .and_then(|d| m.table.checked_add(d))
                        .ok_or_else(|| io_error!("corrupt bucket table offset"))
                        .map(|off| vec![w8(off, TOMBSTONE), w8(handle + LEN_OFF, m.len - 1)]);
                    ProbeStep::Stop(step)
                } else {
                    ProbeStep::Continue
                }
            },
            |_m| Ok(Vec::new()),
        )?;

        if found.get() {
            // SAFETY: the removed value block's ownership transfers to the caller.
            Ok(Some(unsafe {
                BStackOwned::from_raw(Self::value_at(old_value.get()))
            }))
        } else {
            Ok(None)
        }
    }

    /// A **borrowed** handle to the value mapped by `key` (no ownership; valid
    /// only while the entry is not removed), or `None` if absent.
    ///
    /// A plain probe: correct with external synchronization or a single writer,
    /// but not linearized against a concurrent mutation.
    pub fn get(&self, stack: &BStack, key: &K) -> io::Result<Option<V>> {
        let handle = self.range.start();
        let stride = Self::stride();
        let ksz = Self::ksize();
        let [table, cap] = read_fields::<2>(stack, handle + TABLE_OFF)?;
        if cap == 0 {
            return Ok(None);
        }
        let key_bytes = bytemuck::bytes_of(key);
        let mask = cap - 1;
        let mut idx = fnv1a(key_bytes) & mask;
        // One read per probed bucket (state + key + value in a single get_into).
        let mut scratch = Scratch::new();
        for _ in 0..cap {
            // `table` is an on-disk pointer that can be corrupted/forged.
            let bucket = idx
                .checked_mul(stride)
                .and_then(|d| table.checked_add(d))
                .ok_or_else(|| io_error!("corrupt bucket table offset"))?;
            let buf = scratch.buf(stride as usize);
            stack.get_into(bucket, buf)?;
            let state = get_u64(&buf[0..8]);
            if state == EMPTY {
                return Ok(None);
            }
            if state == OCCUPIED && buf[8..8 + ksz] == *key_bytes {
                return Ok(Some(Self::value_at(get_u64(&buf[8 + ksz..8 + ksz + 8]))));
            }
            idx = (idx + 1) & mask;
        }
        Ok(None)
    }

    /// Whether `key` is present.
    pub fn contains_key(&self, stack: &BStack, key: &K) -> io::Result<bool> {
        Ok(self.get(stack, key)?.is_some())
    }

    /// Get the value for `key`, inserting one produced by `f` if absent — the
    /// fused entry operation. Returns `(value handle, was_newly_inserted)`.
    ///
    /// If `key` is already present this is a single probe: `f` is **not** called
    /// and nothing is allocated. Existing values are never replaced (use
    /// [`insert`](Self::insert) for that). The returned handle is mutable in
    /// place, and the `bool` distinguishes a fresh insert from a hit (e.g. to
    /// increment an existing counter). **Single-writer** — do not mutate the map
    /// concurrently across this call.
    pub fn get_or_insert_with<A, F>(&self, allocator: &A, key: K, f: F) -> io::Result<(V, bool)>
    where
        A: BStackRaiiAllocator,
        F: FnOnce() -> io::Result<BStackOwned<V>>,
    {
        if let Some(v) = self.get(allocator.stack(), &key)? {
            return Ok((v, false));
        }
        let value = f()?;
        let vref = value.handle().range().start();
        // Absent per the probe above, so `insert` returns no prior value; reclaim
        // one defensively if a race produced it.
        if let Some(old) = self
            .insert(allocator, key, value)
            .map_err(|e| e.discard_freeing(allocator))?
        {
            old.bstack_drop(allocator)?;
        }
        Ok((Self::value_at(vref), true))
    }

    /// Like [`get_or_insert_with`](Self::get_or_insert_with) but with an eager
    /// `default`. If `key` is present, `default` is **freed** (its block is
    /// dropped) — prefer the `_with` form to avoid allocating a value you may not
    /// use.
    pub fn get_or_insert<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        key: K,
        default: BStackOwned<V>,
    ) -> io::Result<(V, bool)> {
        if let Some(v) = self.get(allocator.stack(), &key)? {
            default.bstack_drop(allocator)?;
            return Ok((v, false));
        }
        let vref = default.handle().range().start();
        if let Some(old) = self
            .insert(allocator, key, default)
            .map_err(|e| e.discard_freeing(allocator))?
        {
            old.bstack_drop(allocator)?;
        }
        Ok((Self::value_at(vref), true))
    }

    /// A lazy iterator over all `(key, value)` entries in **unspecified** order,
    /// yielding `io::Result`. A read snapshot: do not mutate the map while
    /// iterating (mutating a yielded value block is fine).
    pub fn iter<'a>(&self, stack: &'a BStack) -> io::Result<HashMapIter<'a, K, V>> {
        let [table, cap, len] = read_fields::<3>(stack, self.range.start() + TABLE_OFF)?;
        Ok(HashMapIter {
            stack,
            block_off: self.range.start(),
            table,
            cap,
            len,
            stride: Self::stride(),
            ksz: Self::ksize(),
            idx: 0,
            scratch: Scratch::new(),
            _marker: PhantomData,
        })
    }

    /// Grow the table to at least double its capacity, rehashing every live entry
    /// (and dropping tombstones) atomically. A no-op (beyond a freed spare block)
    /// if another thread already grew it.
    fn grow<A: BStackRaiiAllocator>(&self, allocator: &A) -> io::Result<()> {
        grow_table(
            allocator,
            self.range.start(),
            Self::stride(),
            Self::ksize(),
            OCCUPIED,
            MIN_CAP,
        )
    }
}

impl<K: Pod, V: BStackBlock> BStackCast for BStackHashMap<K, V> {
    /// A `"Map"` prefix over hash bytes perturbed by the key size and the value
    /// type's tag, so maps of different key/value types never share a
    /// discriminant despite the identical handle layout.
    fn eightcc() -> EightCC {
        const BASE: EightCC = EightCC::new([b'M', b'a', b'p', 0x80, 0x81, 0x82, 0x83, 0x84]);
        BASE.mix(EightCC::new((size_of::<K>() as u64).to_le_bytes()))
            .mix(<V as BStackCast>::eightcc())
    }
}

// Self-contained (no separate control block): may be `#[embed]`ded.
impl<K: Pod, V: BStackBlock> crate::types::traits::BStackEmbeddable for BStackHashMap<K, V> {}

impl<K: Pod, V: BStackBlock> BStackBlock for BStackHashMap<K, V> {
    type OnDisk = MapOnDisk;

    unsafe fn from_range(range: BStackRange) -> Self {
        BStackHashMap {
            range,
            _marker: PhantomData,
        }
    }

    fn range(&self) -> BStackRange {
        self.range
    }

    /// Recursively free every value block and the bucket block, **without**
    /// freeing the handle block itself.
    fn __bstack_drop_children<A: BStackRaiiAllocator>(
        allocator: &A,
        range: BStackRange,
    ) -> io::Result<()> {
        let stride = Self::stride();
        let ksz = Self::ksize();
        let handle = range.start();
        let [table, cap] = read_fields::<2>(allocator.stack(), handle + TABLE_OFF)?;
        if table == 0 {
            return Ok(());
        }
        // Read the whole bucket block once, then free values from memory.
        // `cap` is an untrusted on-disk field: checked math (a wrap would size the
        // image and the later `dealloc_range` wrong) and a stack bound (fail
        // before allocating, not after).
        let table_size = cap
            .checked_mul(stride)
            .ok_or_else(|| io_error!("hash map capacity overflow"))?;
        if table_size > allocator.stack().len()? {
            return Err(io_error!("hash map bucket block larger than the stack"));
        }
        let mut image = vec![0u8; table_size as usize];
        allocator.stack().get_into(table, &mut image)?;
        for j in 0..cap as usize {
            let lo = j * stride as usize;
            if get_u64(&image[lo..lo + 8]) == OCCUPIED {
                let vref = get_u64(&image[lo + 8 + ksz..lo + 16 + ksz]);
                if vref != 0 {
                    // SAFETY: the map solely owns each value block.
                    let owned = unsafe { BStackOwned::from_raw(Self::value_at(vref)) };
                    owned.bstack_drop(allocator)?;
                }
            }
        }
        // SAFETY: the map solely owns its bucket block.
        unsafe { dealloc_range(allocator, BStackRange::new(table, table_size))? };
        Ok(())
    }

    /// Deep-clone the map into `plan`: copy the bucket block verbatim (keeping
    /// every key's position), deep-clone each occupied value (via `V`'s clone
    /// hook) and swap in the clone's ref; stage the handle — all in the parent
    /// plan's single atomic commit.
    fn __bstack_clone_children_inplace<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        plan: &mut ClonePlan,
    ) -> io::Result<Self::OnDisk> {
        let stride = Self::stride();
        let ksz = Self::ksize();
        let handle = self.range.start();
        let [table, cap, len, used] = read_fields::<4>(allocator.stack(), handle + TABLE_OFF)?;

        let (new_table, new_cap, new_used) = if cap == 0 {
            (0, 0, 0)
        } else {
            // Copy the whole bucket block, then deep-clone the occupied values.
            // Untrusted `cap`: checked math + stack bound (see drop_children).
            let table_size = cap
                .checked_mul(stride)
                .ok_or_else(|| io_error!("hash map capacity overflow"))?;
            if table_size > allocator.stack().len()? {
                return Err(io_error!("hash map bucket block larger than the stack"));
            }
            let mut image = vec![0u8; table_size as usize];
            allocator.stack().get_into(table, &mut image)?;
            for j in 0..cap as usize {
                let lo = j * stride as usize;
                if get_u64(&image[lo..lo + 8]) != OCCUPIED {
                    continue;
                }
                let vref = get_u64(&image[lo + 8 + ksz..lo + 16 + ksz]);
                let cloned = Self::value_at(vref)
                    .__bstack_clone_into(allocator, plan)?
                    .start();
                image[lo + 8 + ksz..lo + 16 + ksz].copy_from_slice(&cloned.to_le_bytes());
            }
            let dst = plan.alloc_raw(allocator, table_size)?;
            plan.write(dst.start(), image);
            (dst.start(), cap, used)
        };

        let od = MapOnDisk {
            header: BlockHeader {
                size: MAP_SIZE,
                tag: Self::eightcc(),
            },
            table: new_table,
            cap: new_cap,
            len,
            used: new_used,
        };
        Ok(od)
    }
}

impl<K: Pod, V: BStackBlock> TryCloneIn for BStackHashMap<K, V> {
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

/// An unordered iterator over a [`BStackHashMap`]'s live entries, yielding
/// `io::Result<(K, V)>`. Created by [`BStackHashMap::iter`]; scans the buckets.
pub struct HashMapIter<'a, K: Pod, V: BStackBlock> {
    stack: &'a BStack,
    /// The map handle block, re-read each step to detect mutation.
    block_off: u64,
    table: u64,
    cap: u64,
    /// Snapshot of the entry count; a same-table insert/remove (no rehash) leaves
    /// `table`/`cap` unchanged but changes `len`, so comparing it too makes the
    /// mutation check catch every mutation, not only a growth.
    len: u64,
    stride: u64,
    ksz: usize,
    idx: u64,
    scratch: Scratch,
    _marker: PhantomData<fn() -> (K, V)>,
}

impl<'a, K: Pod, V: BStackBlock> Iterator for HashMapIter<'a, K, V> {
    type Item = io::Result<(K, V)>;

    fn next(&mut self) -> Option<Self::Item> {
        // Fail fast if the map was mutated during iteration: a growth/rehash
        // frees the old table and repoints `table`; a same-table insert/remove
        // leaves `table`/`cap` but changes `len` (a remove's backward-shift can
        // relocate an unvisited entry into a visited slot, silently skipping it).
        // Comparing all three catches every mutation.
        if self.idx < self.cap {
            match read_fields::<3>(self.stack, self.block_off + TABLE_OFF) {
                Ok([table, cap, len])
                    if table == self.table && cap == self.cap && len == self.len => {}
                Ok(_) => {
                    self.idx = self.cap;
                    return Some(Err(io_error!(
                        "BStackHashMap was mutated during iteration (its table/len \
                         changed); the iterator is invalidated"
                    )));
                }
                Err(e) => {
                    self.idx = self.cap;
                    return Some(Err(e));
                }
            }
        }
        while self.idx < self.cap {
            let i = self.idx;
            self.idx += 1;
            let buf = self.scratch.buf(self.stride as usize);
            if let Err(e) = self.stack.get_into(self.table + i * self.stride, buf) {
                self.idx = self.cap;
                return Some(Err(e));
            }
            if get_u64(&buf[0..8]) == OCCUPIED {
                let k = bytemuck::pod_read_unaligned::<K>(&buf[8..8 + self.ksz]);
                let vref = get_u64(&buf[8 + self.ksz..8 + self.ksz + 8]);
                return Some(Ok((k, BStackHashMap::<K, V>::value_at(vref))));
            }
        }
        None
    }
}
