//! Write-ahead log data model for atomic multi-slice transactions.
//!
//! ## The categories
//!
//! An on-disk **slice** `S = (ptr, len)` is allocated/freed by the adjoint
//! functors `Alloc`/`Dealloc`. Because `Alloc ∘ Dealloc` and `Dealloc ∘ Alloc`
//! are identities, slices form a groupoid — which would license a reduction
//! optimisation (reuse a freed slice for an equal-length allocation instead of
//! freeing then re-allocating); it is not currently implemented (see the note near
//! `finish` on why no task presents both polarities at once).
//!
//! But `Alloc` is *non-deterministic in address*: two requests for the same
//! length yield different slices. So we split the "slice" notion into a
//! requirement category `R = (len)` and a slice category `S`, bridged by an
//! address-preserving `R' = (id, len)`:
//!
//! ```text
//! R --Choice--> R' --Alloc--> S       S --Dealloc--> R' --ForgetAddress--> R
//! ```
//!
//! `Alloc: R' → S` and `Dealloc: S → R'` are adjoint, and equal-length slices are
//! interchangeable (`ForgetAddress ∘ Dealloc ∘ Alloc ∘ Choice = id`).
//!
//! ## The log
//!
//! A traditional durable-monoid WAL can't work here: `Alloc`/`Dealloc` aren't
//! idempotent, so operations can't be replayed. Instead each operation carries a
//! [`WalStatus`] from the ordered set `{None < Pending < Complete}` (plus the
//! recovery sink `Abandon`), with two monotonic maps:
//!
//! * normal progress [`advance`](WalStatus::advance): `None → Pending → Complete`;
//! * [`recover`](WalStatus::recover): `Pending → Abandon`, else identity.
//!
//! An operation is thus `(slice, Alloc|Dealloc) × Status`, and the WAL is the
//! functor `wal_append` mapping operations into disk state ([`WalLog`]). On disk
//! **both** an `Alloc` and a `Dealloc` store their slice `S = (ptr, len)`; the
//! `op` marks the *recovery polarity* (which outcome orphans the slice). Each slice
//! also carries a **file identity** (`file_id`): `0` = the WAL's own file
//! ([`FileId::SELF`](crate::registry::FileId::SELF), the common case), non-zero = a
//! foreign file whose orphan is reclaimed through the [registry](crate::registry) on
//! recovery — the on-disk half of the cross-file (`Foreign<T>`) atomicity story.
//!
//! Recovery semantics ([`finish`]), driven by the transaction-level
//! `txn_status`, **reclaims** rather than merely staying consistent:
//!
//! * **committed** → free each `Pending` `Dealloc` (the old blocks the op
//!   unlinked) — roll forward;
//! * **abandoned** → free each `Pending` `Alloc` (the new blocks a crashed op
//!   allocated but never linked) — reclaim the orphans.
//!
//! Each entry self-brackets (`persist Complete → free`), so a second crash mid-
//! completion never double-frees; no separate cursor is needed.

use core::mem::size_of;
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use bstack::{BStackGenOp, BStackRange};
use bytemuck::{Pod, Zeroable};

use crate::BStackRaiiAllocator;
use crate::io_core::dealloc_range;
use crate::primitives::{NonNullOffset, Offset};
use crate::registry::{self, FileId};
use crate::util::bytes::read_u64;
use crate::util::io_errorfn;

// --- On-disk constants ------------------------------------------------------

/// Magic word at the head of a persistent WAL block ("bstackWA").
const WAL_MAGIC: u64 = 0x6273_7461_636b_5741;

/// Minimum entry capacity of the persistent WAL block; larger transactions grow
/// it to the next power of two.
const WAL_MIN_CAP: u64 = 8;

/// Ceiling on a persisted WAL header's `capacity` (and, transitively, `count`)
/// that this crate will ever trust. `header.capacity`/`header.count` are read
/// straight from disk with only the `magic` field validating the record, so a
/// corrupted header could otherwise claim billions of entries — driving an
/// unbounded allocation in [`HeldLock::load`] (`handle_alloc_error` aborts the
/// process) or, worse, letting [`HeldLock::ensure_block`] skip a real reallocation
/// because a forged `capacity` looks "already big enough" while the real block stays
/// its old (small) size, so a later [`HeldLock::append_alloc`] writes past it. No real
/// transaction logs anywhere near this many allocations.
const WAL_MAX_CAP: u64 = 1 << 20;

/// Offset of the header `count` field (the `u64` after the magic + `txn_status`).
const WAL_COUNT_OFFSET: u64 = 16;

/// Anchor offset for the bstack-provided freeing allocators: the second `u64`
/// word of the user-reserved region every one of them keeps at payload offset 0
/// and never hands out (FirstFit reserves 16 B there, GhostTree 32 B, Slab and
/// CheckedSlab 24 B — all ≥ 16). Payload offset 0 is left as `bstack_raii`'s null
/// niche, so the anchor is the *next* word, `[8, 16)`.
pub const STD_WAL_ANCHOR: NonNullOffset =
    // SAFETY: 8 is non-zero.
    unsafe { NonNullOffset::new_unchecked(Offset::from_raw(8)) };

/// The status lifecycle of a WAL operation: the ordered set
/// `{None < Pending < Complete}` plus the recovery sink `Abandon`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum WalStatus {
    None = 0,
    Pending = 1,
    Complete = 2,
    Abandon = 3,
}

impl WalStatus {
    /// Normal monotonic progress: `None → Pending → Complete` (idempotent at
    /// `Complete`; `Abandon` is terminal).
    #[cfg(test)]
    pub fn advance(self) -> Self {
        match self {
            WalStatus::None => WalStatus::Pending,
            WalStatus::Pending => WalStatus::Complete,
            other => other,
        }
    }

    /// Recovery monotonic map: `Pending → Abandon` (an in-flight op is abandoned,
    /// its slice leaked rather than re-run); `None` and `Complete` are unchanged.
    #[cfg(test)]
    pub fn recover(self) -> Self {
        match self {
            WalStatus::Pending => WalStatus::Abandon,
            other => other,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => WalStatus::None,
            1 => WalStatus::Pending,
            2 => WalStatus::Complete,
            // Any other byte (incl. 3 and corruption) is treated as `Abandon` —
            // the safe sink: never run, leak.
            _ => WalStatus::Abandon,
        }
    }
}

/// The two morphisms of the slice groupoid, as recorded in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum WalOp {
    Alloc = 0,
    Dealloc = 1,
}

impl WalOp {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(WalOp::Alloc),
            1 => Some(WalOp::Dealloc),
            // A corrupt byte decodes to *neither* op — the safe sink, matching
            // `WalStatus`: an unrecognised entry is never acted on, so recovery
            // can at most leak. (Defaulting to `Alloc` was unsafe on the
            // abandoned path: a `Dealloc` misread as `Alloc` freed a slice that
            // was staged-but-never-committed, i.e. still live and linked.)
            _ => None,
        }
    }
}

/// One on-disk WAL entry: `(status, op, file_id, payload)`.
///
/// `status` and `op` are **separate** fields — never packed into one byte. The two
/// payload words are `R' = (id, len)` for an [`Alloc`](WalOp::Alloc) and
/// `S = (ptr, len)` for a [`Dealloc`](WalOp::Dealloc). `file_id` names the file the
/// slice lives in — `0` = the WAL's own file ([`FileId::SELF`], the common case),
/// non-zero = a foreign [`FileId`] reclaimed through the [registry](crate::registry)
/// on recovery. 32 bytes, 8-aligned, `Pod`.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct WalEntry {
    status: u8,
    op: u8,
    _pad: [u8; 6],
    /// The file the recorded slice lives in: [`SELF`](FileId::SELF) = this file,
    /// else a foreign [`FileId`] resolved through the registry on recovery.
    ///
    /// A [`FileId`] is a `#[repr(transparent)]` `u32`, paired with the reserved
    /// `_obj_id` word below so the two together fill the same 8 bytes the field used
    /// to be a single `u64`. On little-endian it keeps its offset (it was the `u64`'s
    /// low word), so the on-disk image is unchanged.
    file_id: FileId,
    /// Reserved for a future intra-file object id (e.g. RTTI / sub-object
    /// addressing). Currently always `0` and unread — split out now so the split is
    /// a no-op on disk rather than a later breaking widening.
    _obj_id: u32,
    /// `Alloc`: requirement `id`. `Dealloc`: slice `ptr` (start offset).
    word_a: u64,
    /// `Alloc` / `Dealloc`: `len`.
    word_b: u64,
}

impl WalEntry {
    /// An `Alloc` entry recording a freshly allocated **local** slice `S = (ptr, len)`
    /// (`file_id 0`, [`FileId::SELF`]). Recovery frees it iff the transaction is
    /// **abandoned** (the block is an orphan of a crashed op); a committed
    /// transaction keeps it. See [`alloc_in`](Self::alloc_in) for a foreign slice.
    pub fn alloc(status: WalStatus, slice: BStackRange) -> Self {
        Self::alloc_in(status, FileId::SELF, slice)
    }

    /// An `Alloc` entry for a slice in file `file` (foreign-aware). `file = `
    /// [`FileId::SELF`] is the WAL's own file; any other id names a foreign file
    /// whose orphan is reclaimed through the [registry](crate::registry) on recovery.
    pub fn alloc_in(status: WalStatus, file: FileId, slice: BStackRange) -> Self {
        WalEntry {
            status: status as u8,
            op: WalOp::Alloc as u8,
            _pad: [0; 6],
            file_id: file,
            _obj_id: 0,
            word_a: slice.start(),
            word_b: slice.len(),
        }
    }

    /// A `Dealloc` entry recording a concrete **local** slice `S = (ptr, len)`
    /// (`file_id 0`, [`FileId::SELF`]). See [`dealloc_in`](Self::dealloc_in) for a
    /// slice in a foreign file.
    #[cfg(test)]
    pub fn dealloc(status: WalStatus, slice: BStackRange) -> Self {
        Self::dealloc_in(status, FileId::SELF, slice)
    }

    /// A `Dealloc` entry for a slice in file `file` (foreign-aware).
    pub fn dealloc_in(status: WalStatus, file: FileId, slice: BStackRange) -> Self {
        WalEntry {
            status: status as u8,
            op: WalOp::Dealloc as u8,
            _pad: [0; 6],
            file_id: file,
            _obj_id: 0,
            word_a: slice.start(),
            word_b: slice.len(),
        }
    }

    pub fn status(&self) -> WalStatus {
        WalStatus::from_u8(self.status)
    }

    /// `None` for a corrupt `op` byte — such an entry is inert on both
    /// recovery paths (see [`WalOp::from_u8`]).
    pub fn op(&self) -> Option<WalOp> {
        WalOp::from_u8(self.op)
    }

    /// The file the recorded slice lives in — [`SELF`](FileId::SELF) = this file,
    /// else a foreign [`FileId`] reclaimed through the registry on recovery. Widened
    /// to `u64` for the recovery path (`FileId::from_u64`).
    pub fn file_id(&self) -> u64 {
        self.file_id.as_u64()
    }

    /// The recorded slice `S`, if this is an `Alloc` entry (to be freed on abandon).
    pub fn as_alloc(&self) -> Option<BStackRange> {
        match self.op() {
            Some(WalOp::Alloc) => Some(BStackRange::new(self.word_a, self.word_b)),
            _ => None,
        }
    }

    /// The recorded slice `S`, if this is a `Dealloc` entry.
    pub fn as_dealloc(&self) -> Option<BStackRange> {
        match self.op() {
            Some(WalOp::Dealloc) => Some(BStackRange::new(self.word_a, self.word_b)),
            _ => None,
        }
    }
}

/// In-memory write-ahead log: a vector of [`WalEntry`] plus the `R'` id counter.
///
/// [`with_capacity`](Self::with_capacity) pre-reserves to the known operation
/// count (a transaction knows how many allocs/deallocs it will log up front), so
/// [`append`](Self::append) never reallocates mid-transaction.
pub(crate) struct WalLog {
    entries: Vec<WalEntry>,
    #[cfg(test)]
    next_id: u64,
}

impl WalLog {
    /// A log pre-reserved for `ops` entries.
    pub fn with_capacity(ops: usize) -> Self {
        WalLog {
            entries: Vec::with_capacity(ops),
            #[cfg(test)]
            next_id: 0,
        }
    }

    /// The next `R'` identity — a wrapping autoincrement.
    #[cfg(test)]
    pub fn fresh_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    /// Append an operation to the log (`wal_append`).
    pub fn append(&mut self, entry: WalEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[WalEntry] {
        &self.entries
    }

    /// The log's on-disk image (a packed array of [`WalEntry`]).
    #[cfg(test)]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.entries)
    }

    /// Parse a log image back into entries (unaligned-safe, so a raw disk buffer
    /// works). Trailing bytes shorter than one entry are ignored.
    pub fn entries_from_bytes(bytes: &[u8]) -> Vec<WalEntry> {
        let sz = size_of::<WalEntry>();
        let count = bytes.len() / sz;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(bytemuck::pod_read_unaligned(&bytes[i * sz..(i + 1) * sz]));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// On-disk WAL block + completion runtime.
//
// The WAL block is **persistent and reused** (Vec-like), not allocated per
// transaction: `[WalHeader | WalEntry × capacity]`, reached through a stable
// anchor slot (see [`BStackRaiiAllocator`]) that holds the block's offset (`0` = not
// yet created — lazily allocated on first use). The header's `txn_status` doubles
// as the in-use flag:
//
//   * `None`     — idle: no transaction in flight (the block is free to reuse);
//   * `Pending`  — a transaction is staged but not committed (abandon on recover);
//   * `Complete` — committed, deallocs may be unfinished (roll forward on recover).
//
// A transaction reuses the block in place (growing it — free old, alloc bigger —
// only when it needs more than `capacity` entries), sets its status back to
// `None` when done, and never frees the block. Concurrent transactions on the
// same file are serialized by an in-memory mutex ([`wal_lock_for`]); the on-disk
// `txn_status` is purely for crash recovery, not live mutual exclusion.
// ---------------------------------------------------------------------------

io_errorfn!(
    corrupt_wal_capacity,
    InvalidData,
    "corrupt persistent WAL block: capacity/count out of range"
);

/// On-disk header of the persistent WAL block. `txn_status` is both the
/// transaction-level commit marker and the idle/in-use flag (`None` = idle);
/// `capacity` is the number of [`WalEntry`] slots the block was allocated for.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct WalHeader {
    magic: u64,
    txn_status: u8,
    _pad: [u8; 7],
    count: u64,
    capacity: u64,
}

impl WalHeader {
    fn txn_status(&self) -> WalStatus {
        WalStatus::from_u8(self.txn_status)
    }
}

impl WalLog {
    /// The used prefix image `[WalHeader | entries]` at the given transaction
    /// status, for a block allocated with `capacity` entry slots. Only the header
    /// and the `count` live entries are written; the spare capacity is left as-is.
    pub fn block_image(&self, txn_status: WalStatus, capacity: u64) -> Vec<u8> {
        let header = WalHeader {
            magic: WAL_MAGIC,
            txn_status: txn_status as u8,
            _pad: [0; 7],
            count: self.entries.len() as u64,
            capacity,
        };
        let mut img =
            Vec::with_capacity(size_of::<WalHeader>() + self.entries.len() * size_of::<WalEntry>());
        img.extend_from_slice(bytemuck::bytes_of(&header));
        img.extend_from_slice(bytemuck::cast_slice(&self.entries));
        img
    }
}

// ---------------------------------------------------------------------------
// In-memory serialization of WAL transactions.
//
// The persistent WAL block + anchor slot are single-writer per file: one
// transaction may be staged there at a time. Since automatic teardown/clone run
// concurrently, an in-memory mutex (keyed by the file's `BStack` identity)
// serializes the whole staging→commit→finish critical section. Different files
// use different locks and never contend. The lock is process-local, matching
// bstack's single-process write model; the on-disk `txn_status` handles the
// orthogonal job of crash recovery across a restart.
// ---------------------------------------------------------------------------

/// Per-file WAL mutex registry, keyed by the address of the file's [`BStack`].
/// Stores [`Weak`](std::sync::Weak) so an entry dies with its last outstanding
/// guard — the map would otherwise grow by one mutex for every distinct `BStack`
/// address the process ever uses. Dead entries are swept on each insert.
static WAL_LOCKS: OnceLock<Mutex<HashMap<usize, std::sync::Weak<Mutex<()>>>>> = OnceLock::new();

/// The WAL mutex for `allocator`'s file (created on first use). Hold its guard
/// across a whole WAL transaction.
pub(crate) fn wal_lock_for<A: BStackRaiiAllocator>(allocator: &A) -> Arc<Mutex<()>> {
    let key = core::ptr::from_ref(allocator.stack()) as usize;
    let reg = WAL_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = reg.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(live) = map.get(&key).and_then(std::sync::Weak::upgrade) {
        return live;
    }
    // Miss (or dead entry): sweep expired weaks so the map stays bounded by the
    // number of *live* locks, then insert a fresh one.
    map.retain(|_, w| w.strong_count() > 0);
    let fresh = Arc::new(Mutex::new(()));
    map.insert(key, Arc::downgrade(&fresh));
    fresh
}

thread_local! {
    /// Stack-pointer keys of the per-file WAL locks this thread currently holds for
    /// in-flight clone transactions. A clone holds a file's lock for its *whole*
    /// descent, so a nested clone that re-enters the **same** file — an owned cross-file
    /// cycle A→B→A, or a `Foreign` with an explicit id that resolves to the home file —
    /// would re-acquire the same non-reentrant `Mutex` the outer clone still holds, a
    /// self-deadlock. Detecting the key here turns that hang into a clean error rather
    /// than progress: the two would otherwise share (and clobber) the one per-file WAL
    /// block, and an owned cycle is unclonable regardless (issue F4).
    static HELD_CLONE_LOCKS: core::cell::RefCell<Vec<usize>> = const { core::cell::RefCell::new(Vec::new()) };
}

/// The file's WAL [`Mutex`] held across a whole clone / RTTI-clone transaction. It
/// owns the `Arc` so the lifetime-extended guard can never outlive the mutex it
/// borrows; `Drop` releases the guard *before* the `Arc` is dropped. Shared by
/// [`crate::io_core::ClonePlan`] and the RTTI `clone_value` interpreter.
pub(crate) struct HeldLock {
    /// `Some` while held; taken in `Drop` so the guard releases before `_arc`.
    guard: Option<MutexGuard<'static, ()>>,
    /// Keeps the mutex alive for as long as `guard` borrows it.
    _arc: Arc<Mutex<()>>,
    /// The file's stack-pointer key, removed from [`HELD_CLONE_LOCKS`] on drop.
    key: usize,
}

impl HeldLock {
    /// Acquire `allocator`'s file WAL lock for a clone. Returns `Err` — instead of
    /// deadlocking — if this thread already holds it (a same-file clone re-entry).
    pub(crate) fn acquire<A: BStackRaiiAllocator>(allocator: &A) -> io::Result<Self> {
        let key = core::ptr::from_ref(allocator.stack()) as usize;
        if HELD_CLONE_LOCKS.with(|h| h.borrow().contains(&key)) {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "clone re-entered a file already being cloned on this thread (an owned \
                 cross-file cycle, or a `Foreign` that resolves to the home file); this \
                 would deadlock on the per-file WAL lock",
            ));
        }
        let arc = wal_lock_for(allocator);
        let guard = arc.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the guard borrows the `Mutex` owned by `arc`, which this struct
        // keeps alive; `Drop` releases the guard before `arc` is dropped, so the
        // borrow never dangles. The transmute only extends the guard's lifetime to
        // `'static` to store it alongside its owning `Arc`.
        let guard: MutexGuard<'static, ()> = unsafe { core::mem::transmute(guard) };
        HELD_CLONE_LOCKS.with(|h| h.borrow_mut().push(key));
        Ok(HeldLock {
            guard: Some(guard),
            _arc: arc,
            key,
        })
    }
}

impl Drop for HeldLock {
    fn drop(&mut self) {
        // Release the guard before `_arc` drops (which would free the `Mutex`).
        self.guard = None;
        HELD_CLONE_LOCKS.with(|h| {
            let mut v = h.borrow_mut();
            if let Some(pos) = v.iter().rposition(|&k| k == self.key) {
                v.swap_remove(pos);
            }
        });
    }
}

// Every persistent-block operation lives here, as a method on [`HeldLock`]: they all
// require the file's WAL lock to be held, so a caller can only reach them through a
// lock it actually holds — the precondition is a **type-state**, not a doc comment,
// and there are no free functions to call by mistake.
impl HeldLock {
    /// Read the anchor slot: the persistent WAL block's offset, or `None` if the
    /// allocator opts out of reclamation ([`wal_anchor`](BStackRaiiAllocator::wal_anchor)
    /// is `None`) or no block has been created yet.
    fn read_anchor<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
    ) -> io::Result<Option<NonNullOffset>> {
        let slot = match allocator.wal_anchor() {
            Some(s) => s,
            None => return Ok(None),
        };
        // `0` is "no block yet"; a real WAL block always sits at a non-zero offset.
        let raw = read_u64(allocator.stack(), slot.as_u64())?;
        Ok(Offset::from_raw(raw).to_non_null())
    }

    /// Ensure the persistent WAL block exists and holds at least `needed` entry
    /// slots, returning it as a [`WalBlock`]. Lazily allocates it on first use;
    /// grows it (free old, allocate a larger one — its contents are transient between
    /// transactions) when a transaction needs more capacity. The header is
    /// (re)initialized `None` (idle) whenever the block is created or grown. Errors if
    /// the allocator names no anchor slot (callers gate on `wal_anchor().is_some()`).
    fn ensure_block<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        needed: u64,
    ) -> io::Result<WalBlock> {
        let slot = allocator
            .wal_anchor()
            .ok_or_else(|| io::Error::other("allocator names no WAL anchor slot"))?;
        let stack = allocator.stack();
        let hsz = size_of::<WalHeader>() as u64;
        let esz = size_of::<WalEntry>() as u64;

        let mut old_to_free: Option<BStackRange> = None;
        if let Some(off) = self.read_anchor(allocator)? {
            let off = off.as_u64();
            let mut hbuf = [0u8; size_of::<WalHeader>()];
            stack.get_into(off, &mut hbuf)?;
            let header: WalHeader = bytemuck::pod_read_unaligned(&hbuf);
            if header.magic == WAL_MAGIC {
                // A forged/corrupted `capacity` is never trusted past this bound — used
                // unchecked it could make `header.capacity >= needed` skip a real
                // reallocation while the block's real (small) size stays put, so a later
                // append writes past it; or size `old_to_free`'s dealloc from thin air.
                // Neither is safe to attempt, so a wildly out-of-range capacity is
                // reported rather than acted on.
                if header.capacity > WAL_MAX_CAP {
                    return Err(corrupt_wal_capacity());
                }
                if header.capacity >= needed {
                    return Ok(WalBlock {
                        range: BStackRange::new(off, hsz + header.capacity * esz),
                        capacity: header.capacity,
                    });
                }
                // Too small: reclaim the old block *after* the new one is allocated and the
                // anchor repointed — link-before-free, so a crash never leaves the anchor
                // naming a freed offset. Its content is transient across the grow.
                old_to_free = Some(BStackRange::new(off, hsz + header.capacity * esz));
            }
        }

        let capacity = needed.max(WAL_MIN_CAP).next_power_of_two();
        let mut slice = allocator.alloc(hsz + capacity * esz)?;
        let off = slice.as_range().start();
        let header = WalHeader {
            magic: WAL_MAGIC,
            txn_status: WalStatus::None as u8,
            _pad: [0; 7],
            count: 0,
            capacity,
        };
        if let Err(e) = slice.write_range(0, bytemuck::bytes_of(&header)) {
            let _ = allocator.dealloc(slice);
            return Err(e);
        }
        // Commit the new block by repointing the anchor. Before this, a crash leaks the
        // fresh block and the anchor still names the valid old one; after it, the old block
        // is safe to reclaim (a crash there merely leaks the old block).
        stack.set(slot.as_u64(), off.to_le_bytes())?;
        if let Some(old) = old_to_free {
            unsafe { dealloc_range(allocator, old)? };
        }
        Ok(WalBlock {
            range: BStackRange::new(off, hsz + capacity * esz),
            capacity,
        })
    }

    /// Stage `log` into the file's persistent WAL block (lazily creating / growing
    /// it) at transaction status `status`, returning the block's range.
    pub(crate) fn persist<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        log: &WalLog,
        status: WalStatus,
    ) -> io::Result<BStackRange> {
        let block = self.ensure_block(allocator, log.entries().len() as u64)?;
        let image = log.block_image(status, block.capacity);
        allocator.stack().set(block.range.start(), &image)?;
        Ok(block.range)
    }

    /// Read the staged transaction in the file's persistent WAL block, if any.
    /// Returns the block range, header, and the `count` live entries; `None` if no
    /// block exists (or the allocator opts out of reclamation).
    fn load<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
    ) -> io::Result<Option<(BStackRange, WalHeader, Vec<WalEntry>)>> {
        let stack = allocator.stack();
        let wal_off = match self.read_anchor(allocator)? {
            Some(off) => off.as_u64(),
            None => return Ok(None),
        };
        let mut hbuf = [0u8; size_of::<WalHeader>()];
        stack.get_into(wal_off, &mut hbuf)?;
        let header: WalHeader = bytemuck::pod_read_unaligned(&hbuf);
        if header.magic != WAL_MAGIC {
            return Ok(None);
        }
        // `count`/`capacity` are on-disk fields validated only by `magic` above; an
        // unbounded `count` would size a `Vec` allocation an attacker fully controls
        // (`handle_alloc_error` aborts the process on failure — worse than a panic),
        // and this is the recovery path `wal::finish` runs on every open.
        if header.capacity > WAL_MAX_CAP || header.count > header.capacity {
            return Err(corrupt_wal_capacity());
        }
        let ebytes = header.count as usize * size_of::<WalEntry>();
        let mut ebuf = vec![0u8; ebytes];
        stack.get_into(wal_off + size_of::<WalHeader>() as u64, &mut ebuf)?;
        let entries = WalLog::entries_from_bytes(&ebuf);
        let block_size =
            size_of::<WalHeader>() as u64 + header.capacity * size_of::<WalEntry>() as u64;
        Ok(Some((
            BStackRange::new(wal_off, block_size),
            header,
            entries,
        )))
    }

    /// Mark the file's persistent WAL block idle (`txn_status := None`) — a
    /// transaction is complete and the block is free to reuse. The block itself is
    /// **not** freed.
    pub(crate) fn set_idle<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        block_off: u64,
    ) -> io::Result<()> {
        // `txn_status` is the byte right after the u64 magic (offset 8).
        allocator
            .stack()
            .set(block_off + 8, [WalStatus::None as u8])
    }

    /// Flip the block's `txn_status` to `Complete` in one [`inplace_gen`](bstack::BStack)
    /// — the **commit point** for a staged transaction (a crash before it abandons;
    /// after it, [`finish`](Self::finish) rolls the frees forward).
    ///
    /// The single-byte write is atomic on its own, so a plain `set` would be *durably*
    /// equivalent — but the commit is deliberately its own `inplace_gen` op (matching
    /// the clone commit): the fault-injection harness's [`FaultPolicy`] sees only op
    /// *names*, not offsets, so this distinct op is the only way a crash test can
    /// pinpoint "just committed" and inject a fault right after it. Collapsing it into
    /// a `set` would make the commit indistinguishable from the ordinary writes around
    /// it.
    fn mark_complete<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        block_off: u64,
    ) -> io::Result<()> {
        let flip = [WalStatus::Complete as u8];
        let mut done = false;
        allocator.stack().inplace_gen(|_feedback| {
            if done {
                None
            } else {
                done = true;
                // SAFETY: `flip` outlives the call; the `txn_status` byte follows the
                // u64 magic at offset 8 in `WalHeader`.
                let data: &[u8] = unsafe { core::mem::transmute::<&[u8], _>(&flip[..]) };
                Some(BStackGenOp::Write {
                    offset: block_off + 8,
                    data,
                })
            }
        })
    }

    /// Append one `Pending` `Alloc` entry to an already-`Pending` WAL block at slot
    /// `index` (0-based, `< capacity`), then **publish** it by bumping the header
    /// `count` to `index + 1`. The entry payload is written *before* the count bump,
    /// so a crash between the two leaves the new entry unseen (recovery reads only the
    /// `count` live entries) — the incremental, intention-first form of [`persist`](Self::persist)
    /// used by a deep clone to log each allocation the instant it is made. The caller
    /// guarantees the block has a slot free at `index`.
    fn append_alloc<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        block_off: u64,
        index: u64,
        slice: BStackRange,
    ) -> io::Result<()> {
        let stack = allocator.stack();
        let hsz = size_of::<WalHeader>() as u64;
        let esz = size_of::<WalEntry>() as u64;
        let entry = WalEntry::alloc(WalStatus::Pending, slice);
        // Write the entry first; only then advance `count` to make it live.
        stack.set(block_off + hsz + index * esz, bytemuck::bytes_of(&entry))?;
        stack.set(block_off + WAL_COUNT_OFFSET, (index + 1).to_le_bytes())?;
        Ok(())
    }

    /// Free one WAL-recorded slice during recovery, in whichever file it lives in.
    ///
    /// * `file_id == 0` ([`FileId::SELF`]) — the WAL's own file: free through the local
    ///   `allocator` (the overwhelmingly common path).
    /// * `file_id != 0` — a foreign file: resolve it through the [registry](crate::registry)
    ///   and free via its live [`BStackRaiiHost`]. If that file is **not currently attached**
    ///   (or the registry is not up), the orphan cannot be reclaimed here — it is *left to
    ///   leak*, which the crate's atomicity contract explicitly permits (a cross-file
    ///   orphan whose file is unavailable at recovery degrades to a leak, exactly as if
    ///   the WAL did not cover it). A malformed id is likewise ignored (leak, not error).
    ///
    /// Errors only propagate a genuine I/O failure from an *attempted* free.
    fn free_recorded<A: BStackRaiiAllocator>(
        &self,
        allocator: &A,
        file_id: u64,
        slice: BStackRange,
    ) -> io::Result<()> {
        if file_id == 0 {
            return unsafe { dealloc_range(allocator, slice) };
        }
        let Some(id) = FileId::from_u64(file_id) else {
            return Ok(()); // malformed id: cannot resolve → leak (permitted)
        };
        match registry::with_host(id, |host| unsafe { host.dealloc(slice) }) {
            Some(res) => res.map_err(|e| e.source),
            None => Ok(()), // file not attached / registry down → leak (permitted)
        }
    }

    /// Complete the file's staged transaction by reclaiming exactly the slices its
    /// outcome orphaned, then marking the block idle. Returns the number freed.
    ///
    /// * **Committed** (`txn_status == Complete`): roll forward — free each still-
    ///   `Pending` `Dealloc` (the old blocks the committed op unlinked).
    /// * **Uncommitted** (`txn_status == Pending`): abandon — free each still-
    ///   `Pending` `Alloc` (the new blocks a crashed op allocated but never linked).
    ///
    /// Each freed entry is persisted `Complete` **before** its slice is freed, so a
    /// second crash mid-completion never double-frees.
    pub(crate) fn finish<A: BStackRaiiAllocator>(&self, allocator: &A) -> io::Result<usize> {
        let (wal_range, header, entries) = match self.load(allocator)? {
            Some(x) => x,
            None => return Ok(0),
        };
        let stack = allocator.stack();
        let txn = header.txn_status();
        if txn == WalStatus::None {
            // Idle block, nothing staged.
            return Ok(0);
        }
        let committed = txn == WalStatus::Complete;
        let base = wal_range.start() + size_of::<WalHeader>() as u64;
        let esz = size_of::<WalEntry>() as u64;
        let mut completed = 0usize;

        // Each orphan entry (committed ⇒ its `Dealloc`, abandoned ⇒ its `Alloc`) is
        // persisted `Complete` *before* its slice is freed, so a second crash can never
        // re-free it.
        if allocator.atomic_bulk() {
            // Bulk allocator: reverse the whole batch of local orphans with one atomic
            // `dealloc_bulk` (a clone's `alloc_bulk`'d region is *not* reclaimed cleanly by
            // freeing its split slices one at a time). Mark every entry `Complete` first,
            // then bulk-free the local slices; foreign slices still go one by one through
            // the registry.
            let mut local: Vec<BStackRange> = Vec::new();
            let mut foreign: Vec<(u64, BStackRange)> = Vec::new();
            for (i, e) in entries.iter().enumerate() {
                if e.status() != WalStatus::Pending {
                    continue;
                }
                let slice = if committed {
                    e.as_dealloc()
                } else {
                    e.as_alloc()
                };
                if let Some(slice) = slice {
                    stack.set(base + i as u64 * esz, [WalStatus::Complete as u8])?;
                    if e.file_id() == 0 {
                        local.push(slice);
                    } else {
                        foreign.push((e.file_id(), slice));
                    }
                    completed += 1;
                }
            }
            if !local.is_empty() {
                // SAFETY: recovery replays ranges the WAL recorded from owned frees.
                unsafe { allocator.free_many(local)? };
            }
            for (fid, s) in foreign {
                self.free_recorded(allocator, fid, s)?;
            }
        } else {
            for (i, e) in entries.iter().enumerate() {
                if e.status() != WalStatus::Pending {
                    continue;
                }
                // Committed: the `Dealloc`s (old blocks) must go. Abandoned: the `Alloc`s
                // (new orphans) must go. Everything else is kept.
                let slice = if committed {
                    e.as_dealloc()
                } else {
                    e.as_alloc()
                };
                if let Some(slice) = slice {
                    let entry_off = base + i as u64 * esz;
                    stack.set(entry_off, [WalStatus::Complete as u8])?;
                    self.free_recorded(allocator, e.file_id(), slice)?;
                    completed += 1;
                }
            }
        }

        // Mark the persistent block idle (kept for reuse); do not free it.
        self.set_idle(allocator, wal_range.start())?;
        Ok(completed)
    }
}

/// An in-flight **intention-first WAL transaction**: the file's WAL lock held for
/// the whole descent, plus the persistent block's offset, entry-slot capacity, and
/// how many entries have been published so far.
///
/// It is the single home for the "log each fresh allocation the instant it is made,
/// growing the block when it fills" logic that [`crate::io_core::ClonePlan`] and the
/// RTTI `clone_value` interpreter both drive — they used to carry a private copy of
/// this state (`CloneWal`) and a duplicated `wal_log_alloc`.
pub(crate) struct WalTxn {
    held: HeldLock,
    block_off: u64,
    capacity: u64,
    logged: u64,
}

impl WalTxn {
    /// Begin a transaction logging the single allocation `first`. Takes the file's
    /// WAL lock for the whole descent; errors (rather than deadlocks) on a same-file
    /// re-entry (see [`HeldLock::acquire`]).
    pub(crate) fn begin<A: BStackRaiiAllocator>(
        allocator: &A,
        first: BStackRange,
    ) -> io::Result<Self> {
        Self::begin_many(allocator, &[first])
    }

    /// Begin a transaction staging `ranges` all at once — the bulk-alloc path, where
    /// every block was allocated up front by one atomic `alloc_bulk`.
    pub(crate) fn begin_many<A: BStackRaiiAllocator>(
        allocator: &A,
        ranges: &[BStackRange],
    ) -> io::Result<Self> {
        let held = HeldLock::acquire(allocator)?;
        let mut log = WalLog::with_capacity(ranges.len());
        for &r in ranges {
            log.append(WalEntry::alloc(WalStatus::Pending, r));
        }
        let block = held.persist(allocator, &log, WalStatus::Pending)?;
        Ok(WalTxn {
            held,
            block_off: block.start(),
            capacity: Self::capacity_of(block),
            logged: ranges.len() as u64,
        })
    }

    /// Log one more allocation `range`: a cheap append while the block has a free
    /// slot, else a full re-persist that grows the block. `logged_so_far` are the
    /// ranges already published (**not** including `range`) — re-logged on the grow
    /// path.
    pub(crate) fn append<A: BStackRaiiAllocator>(
        &mut self,
        allocator: &A,
        logged_so_far: &[BStackRange],
        range: BStackRange,
    ) -> io::Result<()> {
        if self.logged < self.capacity {
            self.held
                .append_alloc(allocator, self.block_off, self.logged, range)?;
            self.logged += 1;
        } else {
            let mut log = WalLog::with_capacity(logged_so_far.len() + 1);
            for &r in logged_so_far {
                log.append(WalEntry::alloc(WalStatus::Pending, r));
            }
            log.append(WalEntry::alloc(WalStatus::Pending, range));
            let block = self.held.persist(allocator, &log, WalStatus::Pending)?;
            self.block_off = block.start();
            self.capacity = Self::capacity_of(block);
            self.logged = logged_so_far.len() as u64 + 1;
        }
        Ok(())
    }

    /// Entry-slot capacity of a persistent WAL block, from its full range.
    fn capacity_of(block: BStackRange) -> u64 {
        (block.len() - size_of::<WalHeader>() as u64) / size_of::<WalEntry>() as u64
    }

    /// Byte offset of the block's `txn_status` marker (right after the u64 magic) —
    /// where a commit flips the transaction `Pending → Complete`.
    pub(crate) fn commit_marker_off(&self) -> u64 {
        self.block_off + 8
    }

    /// Mark the block idle — a committed transaction keeps its block for reuse.
    pub(crate) fn set_idle<A: BStackRaiiAllocator>(&self, allocator: &A) -> io::Result<()> {
        self.held.set_idle(allocator, self.block_off)
    }

    /// Abandon (or complete) the transaction via the recovery path, freeing exactly
    /// the orphans it logged. Used on the rollback / failed-commit paths.
    pub(crate) fn finish<A: BStackRaiiAllocator>(&self, allocator: &A) -> io::Result<usize> {
        self.held.finish(allocator)
    }
}

/// The persistent WAL block [`HeldLock::ensure_block`] guarantees: its on-disk byte
/// [`range`](Self::range) (header + `capacity` entry slots) and its
/// [`capacity`](Self::capacity) in [`WalEntry`] slots.
struct WalBlock {
    range: BStackRange,
    capacity: u64,
}

/// **Complete** a crash-left transaction in the allocator's WAL block: reclaim the
/// slices its outcome orphaned and mark the persistent block idle. Returns the
/// number of slices reclaimed. This is what a caller runs once after `open` — a
/// *completion*, not a leaky recovery. Acquires the file's WAL lock. An allocator
/// that opts out of reclamation ([`wal_anchor`](BStackRaiiAllocator::wal_anchor)
/// is `None`) has no WAL to complete, so this is a no-op returning `0`.
pub fn finish<A: BStackRaiiAllocator>(allocator: &A) -> io::Result<usize> {
    HeldLock::acquire(allocator)?.finish(allocator)
}

/// Commit a batch of frees as one atomic unit, choosing the cheapest safe path:
///
/// * **bulk-capable allocator, all home-file (`SELF`) ranges** — `dealloc_bulk` is
///   itself atomic and self-recovering, so free the whole batch at once and **skip
///   the WAL** (wrapping an already-atomic bulk free in the WAL is redundant and
///   unsound — the allocator's recovery direction is opaque, so a WAL retry could
///   double-free). A crash mid-bulk is reclaimed by the allocator's own recovery.
/// * **otherwise** (a sequential allocator, or a mixed cross-file batch) — stage a
///   `Complete` `Dealloc` transaction and execute it, so a crash mid-free leaves a
///   `Complete` WAL that [`finish`] rolls forward on reopen rather than leaking.
///   Foreign (`file_id != 0`) frees are routed to their file via the registry on
///   `finish` (see [`HeldLock::free_recorded`]).
///
/// The staging→commit→finish critical section runs under the file's WAL lock, so
/// concurrent commits on the same file serialize here. This is the single home for
/// the "bulk-or-WAL" free dispatch that `teardown::wal_teardown` and the RTTI
/// interpreter (via [`commit_home_frees`]) both drive.
pub(crate) fn commit_frees<A: BStackRaiiAllocator>(
    allocator: &A,
    slices: Vec<(FileId, BStackRange)>,
) -> io::Result<()> {
    if slices.is_empty() {
        return Ok(());
    }
    if allocator.atomic_bulk() && slices.iter().all(|(fid, _)| *fid == FileId::SELF) {
        // SAFETY: each range was collected from an owned handle's own teardown (or a
        // caller that asserts the same), so it is a live allocation no other handle
        // will also free.
        return unsafe { allocator.free_many(slices.into_iter().map(|(_, r)| r)) };
    }
    let held = HeldLock::acquire(allocator)?;
    let mut log = WalLog::with_capacity(slices.len());
    for (fid, s) in &slices {
        // `fid == SELF` ⇒ a local free (this file); a foreign id ⇒ reclaimed in that
        // file through the registry on `finish`.
        log.append(WalEntry::dealloc_in(WalStatus::Pending, *fid, *s));
    }
    // Stage `Pending`, then commit by flipping `txn_status` to `Complete` — the single
    // commit point. With the transaction `Complete`, `finish` executes the frees and
    // marks the block idle for reuse.
    let wal_range = held.persist(allocator, &log, WalStatus::Pending)?;
    held.mark_complete(allocator, wal_range.start())?;
    held.finish(allocator)?;
    Ok(())
}

/// Commit a batch of **home-file** owned-range frees for crash recovery, routed
/// exactly as [`commit_frees`] (a bulk-capable allocator frees them atomically; a
/// sequential one goes through the WAL, so a crash mid-free rolls forward on the
/// next open rather than leaking permanently). The RTTI interpreter's free sink, so
/// its collected frees no longer bypass the WAL their static counterparts
/// (`wal_teardown` / `bstack_move!`) go through.
///
/// # Safety
/// Each range must be a live allocation owned by `allocator` that no other live
/// handle will also free (as for [`crate::io_core::teardown::dealloc_range`]).
pub(crate) unsafe fn commit_home_frees<A: BStackRaiiAllocator>(
    allocator: &A,
    ranges: Vec<BStackRange>,
) -> io::Result<()> {
    commit_frees(
        allocator,
        ranges.into_iter().map(|r| (FileId::SELF, r)).collect(),
    )
}

// The groupoid-reduction optimisation (cancel an allocation requirement against an
// equal-length freed slice in the same file, reusing its storage) has been removed:
// no commit path ever presents allocs *and* deallocs together for it to cancel. Each
// task is single-polarity — a **clone** only allocates, a **teardown** only frees, and
// a **vec** growth reallocs (its own atomic op) — so there is never a freed slice on
// hand to repurpose for a concurrent allocation. If a mixed-polarity batch is ever
// introduced, resurrect `reduce` (and its `AllocReq` / `Reduced` planning shapes) then.

// --- Other test-only helpers ------------------------------------------------

/// Test-only: hold `alloc`'s clone WAL lock, then attempt to re-acquire it on the same
/// thread — exactly what a same-file nested clone (owned A→B→A cycle, or a `Foreign`
/// resolving to the home file) does. The re-entry must return `Err` instead of
/// deadlocking on the non-reentrant lock (issue F4). Returns whether it was rejected;
/// the outer lock is released when this returns, so the state is left clean.
#[cfg(test)]
pub(crate) fn test_reentrant_acquire_is_rejected<A: BStackRaiiAllocator>(alloc: &A) -> bool {
    let _outer = HeldLock::acquire(alloc).expect("first acquire should succeed");
    HeldLock::acquire(alloc).is_err()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wal_status_maps() {
        // Normal progress: N → P → C → C; Abandon terminal.
        assert_eq!(WalStatus::None.advance(), WalStatus::Pending);
        assert_eq!(WalStatus::Pending.advance(), WalStatus::Complete);
        assert_eq!(WalStatus::Complete.advance(), WalStatus::Complete);
        assert_eq!(WalStatus::Abandon.advance(), WalStatus::Abandon);
        // Recovery: only Pending → Abandon.
        assert_eq!(WalStatus::None.recover(), WalStatus::None);
        assert_eq!(WalStatus::Pending.recover(), WalStatus::Abandon);
        assert_eq!(WalStatus::Complete.recover(), WalStatus::Complete);
    }

    #[test]
    fn wal_entry_roundtrip() {
        let a = WalEntry::alloc(WalStatus::Pending, BStackRange::new(0x1000, 256));
        assert_eq!(a.op(), Some(WalOp::Alloc));
        assert_eq!(a.status(), WalStatus::Pending);
        assert_eq!(a.as_alloc(), Some(BStackRange::new(0x1000, 256)));
        assert_eq!(a.as_dealloc(), None);
        assert_eq!(a.file_id(), 0); // local convenience ctor ⇒ SELF

        let d = WalEntry::dealloc(WalStatus::Complete, BStackRange::new(0x6CD4, 256));
        assert_eq!(d.op(), Some(WalOp::Dealloc));
        assert_eq!(d.as_dealloc(), Some(BStackRange::new(0x6CD4, 256)));
        assert_eq!(d.as_alloc(), None);
        assert_eq!(d.file_id(), 0);

        // Foreign-aware ctors carry the file id.
        let fa = WalEntry::alloc_in(
            WalStatus::Pending,
            FileId::from_u64(7).unwrap(),
            BStackRange::new(0x20, 48),
        );
        assert_eq!(fa.file_id(), 7);
        assert_eq!(fa.as_alloc(), Some(BStackRange::new(0x20, 48)));
        let fd = WalEntry::dealloc_in(
            WalStatus::Pending,
            FileId::from_u64(3).unwrap(),
            BStackRange::new(0x40, 16),
        );
        assert_eq!(fd.file_id(), 3);
        assert_eq!(fd.as_dealloc(), Some(BStackRange::new(0x40, 16)));
    }

    #[test]
    fn wal_entry_is_32_bytes_and_pod_roundtrips() {
        assert_eq!(size_of::<WalEntry>(), 32);
        let mut log = WalLog::with_capacity(2);
        log.append(WalEntry::alloc(
            WalStatus::Pending,
            BStackRange::new(8192, 64),
        ));
        log.append(WalEntry::dealloc_in(
            WalStatus::Pending,
            FileId::from_u64(9).unwrap(),
            BStackRange::new(4096, 64),
        ));
        let bytes = log.as_bytes().to_vec();
        let back = WalLog::entries_from_bytes(&bytes);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].as_alloc(), Some(BStackRange::new(8192, 64)));
        assert_eq!(back[0].file_id(), 0);
        assert_eq!(back[1].as_dealloc(), Some(BStackRange::new(4096, 64)));
        assert_eq!(back[1].file_id(), 9); // file id survives the on-disk round trip
    }

    #[test]
    fn wal_fresh_id_wraps() {
        let mut log = WalLog::with_capacity(0);
        assert_eq!(log.fresh_id(), 0);
        assert_eq!(log.fresh_id(), 1);
        log.next_id = u64::MAX;
        assert_eq!(log.fresh_id(), u64::MAX);
        assert_eq!(log.fresh_id(), 0); // wrapped
    }
}
