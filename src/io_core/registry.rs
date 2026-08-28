//! Process-wide registry of bstack files — the stateful foundation for
//! `Foreign<T>` (cross-file pointers), built before `Foreign<T>` itself.
//!
//! A `Foreign<T>` is "a slice with a file identity attached": a [`FileId`] plus a
//! [`BStackRange`]. Paths are variable-length and awkward to embed on disk, so the
//! registry maps each file's **persistent path** to a small, **stable** numeric
//! [`FileId`] that a `Foreign` can store as a plain integer. Resolving that id
//! back to the file's live allocator (to read/write/allocate in it) happens here.
//!
//! ## Two layers
//!
//! * **Persistent** — a dedicated bstack file (its own [`FirstFitBStackAllocator`])
//!   holding the **append-only** path table. Ids are just indices into it, so a
//!   `Foreign` written to disk with `FileId(5)` means the same path on every future
//!   run. Paths are *never removed* (that would renumber ids and dangle stored
//!   `Foreign`s); only appended.
//! * **In-memory** — mirrors the path↔id maps and adds `id -> live host`: the
//!   *open* allocator for a file, type-erased behind [`BStackRaiiHost`]. Guarded by a
//!   [`parking_lot::RwLock`].
//!
//! ## Why an `RwLock` (and why `parking_lot`)
//!
//! Resolving a foreign file to run an op on it is **hot** and concurrent (many
//! readers); *detaching* a live file is **cold** and must not race an in-flight op
//! (an exclusive writer). That is exactly a read-write lock — and since the read
//! side sits on the bstack io hot path, [`parking_lot::RwLock`] (cheaper, no
//! poisoning) is preferred over `std`. [`FileRegistry::with_host`] holds the read
//! lock for the whole duration of the caller's closure, so a concurrent
//! [`detach`](FileRegistry::detach) blocks until the op finishes — the "stop
//! token" that keeps a live file from vanishing mid-operation.
//!
//! ## Optional and zero-cost when unused
//!
//! The registry is a lazily-created global: a program that never registers a file
//! never instantiates it, so ordinary single-file ops pay nothing. It is brought
//! up explicitly with [`init`].

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use bstack::{BStack, BStackAllocError, BStackAllocator, BStackOwnedSlice};
use parking_lot::RwLock;

use crate::primitives::{NonNullOffset, WidePtr};
use crate::util::io_error;
use crate::{BStackRaiiAllocator, get_u64};

/// The allocator capability's cross-file projection, defined among the
/// [semantic types](crate::types::alloc); re-exported here because resolving a
/// [`FileId`] to a live host is the registry's job.
pub use crate::types::alloc::SyncBStackRaiiAllocator;
pub use crate::types::alloc::{BStackRaiiAllocError, BStackRaiiHost};

/// The stable per-file identity underlying `Foreign<T>`. Defined among the wide
/// pointer's [components](crate::primitives); re-exported here as its resolution
/// (path table, live hosts) is the registry's job.
pub use crate::primitives::FileId;

/// An **allocator adapter over a live foreign host** — the bridge that lets the
/// crate's entirely generic teardown / clone machinery (`OwnedRef`, `StrongRef`,
/// `WeakRef`, `dealloc_range`, `__bstack_drop_children`, …), all written against
/// `A: BStackRaiiAllocator`, run **against another file** without duplicating any of
/// it. A `Foreign<T>` field's teardown resolves the target file's host through the
/// [registry](self), wraps it here, and runs the same `T` teardown it would run
/// locally — every read/write lands in the foreign file (via [`stack`](BStackAllocator::stack)),
/// and every free is [tagged](BStackRaiiAllocator::wal_file_id) with the foreign
/// [`FileId`] so the home file's WAL reclaims it *there*.
///
/// It owns an `Arc<dyn BStackRaiiHost>` (not a borrow) precisely so it can be
/// `'static`, which [`BStackRaiiAllocator`]'s `'static` supertrait
/// ([`BStackOwnedSliceAllocator`]) demands — and so the host stays alive for the
/// whole teardown even if it is concurrently [`detach`](FileRegistry::detach)ed.
///
/// [`into_stack`](BStackAllocator::into_stack) is unsupported (the adapter does not
/// own its `BStack` — the host does); it panics if ever called. The teardown path
/// only ever uses `stack` / `dealloc` / the refcount primitives, never `into_stack`.
pub struct ForeignHostAllocator {
    host: Arc<dyn BStackRaiiHost + 'static>,
    file_id: FileId,
}

impl ForeignHostAllocator {
    /// Adapt a live foreign `host` (as an owned `Arc`, e.g. from
    /// [`host_arc`](FileRegistry::host_arc)) into an allocator whose frees are tagged
    /// with `file_id`.
    pub fn new(host: Arc<dyn BStackRaiiHost + 'static>, file_id: FileId) -> Self {
        Self { host, file_id }
    }
}

impl BStackAllocator for ForeignHostAllocator {
    type Error = io::Error;
    type Allocated<'a> = BStackOwnedSlice<'a, Self>;

    fn stack(&self) -> &BStack {
        self.host.stack()
    }

    fn into_stack(self) -> BStack {
        unreachable!(
            "ForeignHostAllocator is a cross-file adapter over a shared host and cannot \
             be consumed into its BStack"
        )
    }

    fn alloc(&self, len: u64) -> io::Result<BStackOwnedSlice<'_, Self>> {
        let r = self.host.alloc(len)?;
        // SAFETY: `r` is a fresh live allocation in the host's file; we wrap it as the
        // owned handle bound to this adapter, which forwards frees to the same host.
        Ok(unsafe { BStackOwnedSlice::from_raw_range(self, r) })
    }

    fn realloc<'a>(
        &'a self,
        handle: BStackOwnedSlice<'a, Self>,
        new_len: u64,
    ) -> Result<BStackOwnedSlice<'a, Self>, BStackAllocError<'a, Self>> {
        let r = handle.as_range();
        // SAFETY: `handle` is a live allocation owned here; `r` names the same region.
        match unsafe { self.host.realloc(r, new_len) } {
            Ok(nr) => Ok(unsafe { BStackOwnedSlice::from_raw_range(self, nr) }),
            Err(e) => Err(foreign_to_alloc_error(self, e)),
        }
    }

    fn dealloc<'a>(
        &'a self,
        handle: BStackOwnedSlice<'a, Self>,
    ) -> Result<(), BStackAllocError<'a, Self>> {
        let r = handle.as_range();
        // SAFETY: `handle` is a live allocation owned here; `r` names the same region.
        match unsafe { self.host.dealloc(r) } {
            Ok(()) => Ok(()),
            Err(e) => Err(foreign_to_alloc_error(self, e)),
        }
    }
}

// SAFETY: (1) null niche — the adapter forwards to a real host allocator, which
// upholds it. (2) `wal_anchor` mirrors the host's; `wal_file_id` names the foreign
// file so its frees are reclaimed there.
unsafe impl BStackRaiiAllocator for ForeignHostAllocator {
    fn wal_anchor(&self) -> Option<NonNullOffset> {
        self.host.wal_anchor()
    }
    fn wal_file_id(&self) -> FileId {
        self.file_id
    }
}

/// Convert a host-level [`BStackRaiiAllocError`] (range-based) into the allocator-level
/// [`BStackAllocError`] the `BStackAllocator` trait speaks, re-wrapping any surviving
/// range as an owned handle bound to `alloc`.
fn foreign_to_alloc_error(
    alloc: &ForeignHostAllocator,
    e: BStackRaiiAllocError,
) -> BStackAllocError<'_, ForeignHostAllocator> {
    match e.handle {
        // SAFETY: `h` is the region the failed op left intact in the host's file.
        Some(h) => BStackAllocError::with_handle(e.source, unsafe {
            BStackOwnedSlice::from_raw_range(alloc, h)
        }),
        None => BStackAllocError::lost(e.source),
    }
}

/// Persistent backing: an append-only log on the registry's own bstack file.
///
/// No allocator needed — the path table is append-only, and a `BStack` *is* a
/// durable stack, so we just `push` one record per path and read them back from the
/// bottom. Each record is `[len: u64 | path bytes]`; the record's index (order of
/// pushing) is its `FileId`. Each `push` is crash-atomic (bstack contract), so a
/// crash leaves whole records only — a partial trailing record is impossible.
struct RegistryStore {
    stack: BStack,
}

impl RegistryStore {
    /// Open (or create) the registry file and load its path table into memory.
    fn open(path: &Path) -> io::Result<(Self, Vec<PathBuf>)> {
        let stack = BStack::open(path)?;
        let paths = Self::load(&stack)?;
        Ok((RegistryStore { stack }, paths))
    }

    /// Load the append-only path table from the registry's bstack file into memory.
    fn load(stack: &BStack) -> io::Result<Vec<PathBuf>> {
        let total = stack.len()?;
        if total == 0 {
            return Ok(Vec::new());
        }
        let buf = stack.get(0, total)?;
        let mut paths = Vec::new();
        let mut cur = 0usize;
        while cur + 8 <= buf.len() {
            let len = get_u64(&buf[cur..]) as usize;
            cur += 8;
            // `checked_add`, not `cur + len > buf.len()`: a forged `len` near
            // `usize::MAX` would wrap that unchecked sum below `cur`, passing
            // the bounds check and then slicing `buf[cur..cur+len]` with
            // `start > end`, which panics.
            match cur.checked_add(len) {
                Some(end) if end <= buf.len() => {
                    paths.push(bytes_to_path(&buf[cur..end]));
                    cur = end;
                }
                _ => {
                    // Truncated or forged trailing record — shouldn't happen
                    // (push is atomic), but stop rather than misparse.
                    break;
                }
            }
        }
        Ok(paths)
    }

    /// Append one `[len | path]` record to the log (one atomic `push`).
    fn append(&self, path: &Path) -> io::Result<()> {
        let bytes = path_to_bytes(path);
        let mut rec = Vec::with_capacity(8 + bytes.len());
        rec.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        rec.extend_from_slice(&bytes);
        self.stack.push(rec)?;
        Ok(())
    }
}

/// The registry itself.
///
/// See [`FileRegistry`] for the public interface.
struct RegistryInner<'h> {
    /// `id -> path` (append-only; index is the `FileId`).
    paths: Vec<PathBuf>,
    /// `path -> id`, for idempotent registration.
    by_path: HashMap<PathBuf, FileId>,
    /// `id -> live host` (the open allocator), or `None` when the file is not
    /// currently attached. In-memory only.
    live: Vec<Option<Arc<dyn BStackRaiiHost + 'h>>>,
    /// Reverse map `host BStack address -> id`, for turning a live handle back into
    /// its [`FileId`] (`bstack_cast!(slice as Foreign<T>)`). In-memory only;
    /// populated on [`attach`](FileRegistry::attach), pruned on
    /// [`detach`](FileRegistry::detach).
    by_stack: HashMap<usize, FileId>,
    store: RegistryStore,
}

/// The file registry (see the [module docs](self)).
///
/// The in-memory mirror of the persistent path table, plus the live host for each
/// file. Guarded by a [`parking_lot::RwLock`] for concurrent reads and exclusive writes.
///
/// The `'h` lifetime is how long an attached host must live: a host need only
/// outlive *its attachment*, not the whole program, so a scoped `FileRegistry<'a>`
/// can hold hosts borrowing local data. The process-wide singleton behind [`init`]
/// is a `FileRegistry<'static>` (a `static` can hold nothing shorter), which is why
/// the free-function [`attach`] requires `'static` while [`FileRegistry::attach`]
/// does not.
pub struct FileRegistry<'h> {
    inner: RwLock<RegistryInner<'h>>,
}

impl<'h> FileRegistry<'h> {
    /// Open (or create) a registry backed by the file at `path`. The process-wide
    /// [`init`] wraps this; a standalone instance is mainly useful for tests
    /// (the global is a one-shot singleton).
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let (store, paths) = RegistryStore::open(path)?;
        let by_path = paths
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), FileId::from_raw(i as u32 + 1))) // ids are 1-based (0 = SELF)
            .collect();
        let live = (0..paths.len()).map(|_| None).collect();
        Ok(FileRegistry {
            inner: RwLock::new(RegistryInner {
                paths,
                by_path,
                live,
                by_stack: HashMap::new(),
                store,
            }),
        })
    }

    /// Assign (or look up) the stable [`FileId`] for `path`, persisting a new path
    /// to the append-only table. Idempotent: an already-registered path returns its
    /// existing id without touching disk.
    pub fn register_path(&self, path: &Path) -> io::Result<FileId> {
        let mut g = self.inner.write();
        if let Some(&id) = g.by_path.get(path) {
            return Ok(id);
        }
        let next = g.paths.len() as u32 + 1; // ids are 1-based; 0 is reserved for SELF
        if next >= FileId::SPECIAL_FLOOR {
            return Err(io_error!(
                OutOfMemory,
                "file registry exhausted the ordinary id space"
            ));
        }
        let id = FileId::from_raw(next);
        // Persist first (append the record); only mutate memory once the disk write
        // succeeds, so a failed append leaves us consistent.
        g.store.append(path)?;
        g.paths.push(path.to_path_buf());
        g.by_path.insert(path.to_path_buf(), id);
        g.live.push(None);
        Ok(id)
    }

    /// Register `path` (if needed) and mark it **live**, storing `host` as the open
    /// allocator resolved by [`with_host`](Self::with_host). Returns the file's id.
    ///
    /// `host` need only live as long as `'h` (until this registry — or the host's
    /// [`detach`](Self::detach) — drops it), not `'static`.
    pub fn attach(&self, path: &Path, host: Arc<dyn BStackRaiiHost + 'h>) -> io::Result<FileId> {
        let id = self.register_path(path)?;
        let stack_key = core::ptr::from_ref(host.stack()) as usize;
        let mut g = self.inner.write();
        // `register_path` returns a concrete, 1-based id, so it always has a table slot.
        let idx = id
            .table_index()
            .expect("register_path returns a concrete 1-based id");
        // One host may be live under only one id: attaching the same host under a
        // second path would alias it (`live[]` holding it twice while `by_stack`
        // can name only one id), leaving the reverse map inconsistent with the
        // live table. Re-attaching under the *same* id is the ordinary idempotent
        // case and falls through.
        if let Some(&prev) = g.by_stack.get(&stack_key)
            && prev != id
            && prev.table_index().is_some_and(|p| g.live[p].is_some())
        {
            return Err(io_error!(
                InvalidInput,
                "attach: this host is already attached under a different FileId"
            ));
        }
        // Re-attaching an id that is already live (e.g. the same path with a different
        // host): drop the *replaced* host's reverse-map entry, else its stale
        // `by_stack` key survives — two stacks would then map to one id, and the old
        // key would outlive even a later `detach` (which only prunes the current host).
        if let Some(old) = g.live[idx].take() {
            let old_key = core::ptr::from_ref(old.stack()) as usize;
            g.by_stack.remove(&old_key);
        }
        g.live[idx] = Some(host);
        g.by_stack.insert(stack_key, id);
        Ok(id)
    }

    /// Drop the live host for `id` (the file's *path* stays registered forever).
    /// Takes the write lock, so it waits for any in-flight [`with_host`] op to
    /// finish and cannot run *during* one.
    pub fn detach(&self, id: FileId) {
        let Some(idx) = id.table_index() else { return };
        let mut g = self.inner.write();
        // Take the host out (this is the detach) and drop its reverse-map entry.
        let stack_key = g
            .live
            .get_mut(idx)
            .and_then(|slot| slot.take())
            .map(|host| core::ptr::from_ref(host.stack()) as usize);
        // Prune the reverse entry only if it still names *this* id — after an
        // address reuse (or a historical aliasing), the key may belong to another
        // live id, whose entry must survive this detach.
        if let Some(k) = stack_key
            && g.by_stack.get(&k) == Some(&id)
        {
            g.by_stack.remove(&k);
        }
    }

    /// Run `f` against `id`'s live host under a (recursive) read lock, so the file
    /// cannot be [`detach`](Self::detach)ed while `f` runs. Returns `None` if `id`
    /// is unknown or not currently live.
    ///
    /// Uses `read_recursive`, so a foreign op whose `f` itself resolves *another*
    /// foreign file (nesting `with_host`) never deadlocks behind a queued writer:
    /// readers are admitted even while a `detach` waits. The trade-off is that a
    /// `detach` can be starved by a continuous stream of readers — acceptable, since
    /// detaching is cold and a perpetually-in-use file cannot be safely detached
    /// anyway.
    pub fn with_host<R>(&self, id: FileId, f: impl FnOnce(&dyn BStackRaiiHost) -> R) -> Option<R> {
        // `SELF` / special ids name no registry entry, so return without ever taking
        // the lock — the caller resolves `SELF` against its own local allocator.
        let idx = id.table_index()?;
        let g = self.inner.read_recursive();
        let host = g.live.get(idx)?.as_ref()?;
        Some(f(&**host))
    }

    /// Clone out `id`'s live host as an owned [`Arc`], or `None` if `id` is unknown /
    /// not currently live. Unlike [`with_host`](Self::with_host) (which lends a
    /// `&dyn BStackRaiiHost` only for the span of a closure), this hands back an owned
    /// handle that keeps the host alive independently of the registry — the basis for
    /// the `'static` [`ForeignHostAllocator`], which needs to outlive the lock and
    /// survive a concurrent [`detach`](Self::detach) mid-teardown.
    pub fn host_arc(&self, id: FileId) -> Option<Arc<dyn BStackRaiiHost + 'h>> {
        let idx = id.table_index()?;
        self.inner.read_recursive().live.get(idx)?.clone()
    }

    /// The path registered for `id`, if any (`None` for `SELF` / special ids).
    pub fn path_of(&self, id: FileId) -> Option<PathBuf> {
        let idx = id.table_index()?;
        self.inner.read().paths.get(idx).cloned()
    }

    /// The id registered for `path`, if any.
    pub fn id_of(&self, path: &Path) -> Option<FileId> {
        self.inner.read().by_path.get(path).copied()
    }

    /// Whether `id` currently has a live host attached (always `false` for `SELF` /
    /// special ids).
    pub fn is_live(&self, id: FileId) -> bool {
        let Some(idx) = id.table_index() else {
            return false;
        };
        self.inner.read().live.get(idx).is_some_and(Option::is_some)
    }

    /// The [`FileId`] of the currently-attached file whose backing stack is `stack`,
    /// if any — the reverse of [`with_host`](Self::with_host). Lets a live handle be
    /// turned back into a `Foreign` (`bstack_cast!(slice as Foreign<T>)`).
    pub fn id_of_host(&self, stack: &BStack) -> Option<FileId> {
        let key = core::ptr::from_ref(stack) as usize;
        self.inner.read().by_stack.get(&key).copied()
    }
}

/// The lazily-instantiated process-wide singleton + free-function front door.
/// A `static` holds nothing shorter than `'static`, so the global registry's hosts
/// are `'static` (see [`FileRegistry`] for the scoped, shorter-lived alternative).
static REGISTRY: OnceLock<FileRegistry<'static>> = OnceLock::new();

/// Bring up the process-wide registry, backed by the bstack file at
/// `registry_path` (created if absent, its path table loaded if present). Call
/// once, before any [`attach`]/[`register_path`]. Errors if already initialized.
pub fn init(registry_path: impl AsRef<Path>) -> io::Result<()> {
    let reg = FileRegistry::open(registry_path.as_ref())?;
    REGISTRY
        .set(reg)
        .map_err(|_| io_error!(AlreadyExists, "file registry already initialized"))
}

/// The initialized registry, or `None` if [`init`] has not run. `Foreign`
/// resolution uses this so an unregistered/opt-out program pays nothing.
pub fn get() -> Option<&'static FileRegistry<'static>> {
    REGISTRY.get()
}

fn require() -> io::Result<&'static FileRegistry<'static>> {
    REGISTRY.get().ok_or_else(|| {
        io_error!(
            NotFound,
            "file registry not initialized, init with `bstack_raii::registry::init` first"
        )
    })
}

/// [`FileRegistry::register_path`] on the process-wide registry.
pub fn register_path(path: impl AsRef<Path>) -> io::Result<FileId> {
    require()?.register_path(path.as_ref())
}

/// [`FileRegistry::attach`] on the process-wide registry, taking an owned
/// allocator (any [`SyncBStackRaiiAllocator`]) as the file's live host.
///
/// The `'static` bound is inherent to the *global* registry — a `static` cannot
/// hold a shorter-lived host. It is not a constraint of the machinery: bstack's own
/// allocators own their file and are `'static` anyway, and a host that borrows must
/// go through a scoped [`FileRegistry`] instance (whose `attach` accepts any `'h`).
pub fn attach<A>(path: impl AsRef<Path>, allocator: A) -> io::Result<FileId>
where
    A: SyncBStackRaiiAllocator + 'static,
{
    require()?.attach(path.as_ref(), Arc::new(allocator))
}

/// [`FileRegistry::detach`] on the process-wide registry (no-op if uninitialized).
pub fn detach(id: FileId) {
    if let Some(reg) = REGISTRY.get() {
        reg.detach(id);
    }
}

/// [`FileRegistry::with_host`] on the process-wide registry (`None` if
/// uninitialized or `id` is not live).
pub fn with_host<R>(id: FileId, f: impl FnOnce(&dyn BStackRaiiHost) -> R) -> Option<R> {
    REGISTRY.get()?.with_host(id, f)
}

/// The path registered for `id`, if any.
pub fn path_of(id: FileId) -> Option<PathBuf> {
    REGISTRY.get()?.path_of(id)
}

/// [`FileRegistry::id_of_host`] on the process-wide registry.
pub fn id_of_host(stack: &BStack) -> Option<FileId> {
    REGISTRY.get()?.id_of_host(stack)
}

/// **Read-side `SELF` resolution**. A stored [`WidePtr`] whose file is
/// [`SELF`](FileId::SELF) is only meaningful *relative to the file it was read from*;
/// handed out unchanged, an in-memory `SELF` pointer stored into a *different* file
/// would later free or copy the wrong file's block. This rebinds a `SELF` pointer to
/// the reading file's **registered** [`FileId`], so the escaped pointer is *explicit*
/// and routes correctly wherever it is later stored. An explicit pointer (and the null
/// niche) pass through unchanged.
///
/// **When `home` is not registered the `SELF` pointer passes through as-is** (not an
/// error): a `SELF` pointer that reached storage through *safe* code was minted by
/// [`Foreign::from_local`](crate::Foreign::from_local) / `bstack_cast!`, both of which
/// require the file to be registered — so the safe surface is always resolvable here.
/// An unregistered file's `SELF` can only have been minted through `unsafe`
/// [`Foreign::new`](crate::Foreign::new), whose contract already forbids moving it to
/// another file. Returns `io::Result` for signature symmetry with the fallible read
/// paths that call it; it currently never errors. The inverse is
/// [`home_relative_repr`], applied on write.
pub fn resolve_self_repr(repr: WidePtr, home: &BStack) -> io::Result<WidePtr> {
    if !repr.is_self() || repr.is_null() {
        return Ok(repr);
    }
    match id_of_host(home) {
        // Rebind the `SELF` pointer to the reading file's explicit id, keeping its
        // type tag and address.
        Some(id) => Ok(WidePtr::with_parts(id, repr.type_id(), repr.offset())),
        // Unregistered home: no id to resolve against. A safe SELF can't reach here
        // (its minting required registration); leave an unsafe-minted one as-is.
        None => Ok(repr),
    }
}

/// **Write-side `SELF` re-encoding**, the inverse of [`resolve_self_repr`]. An explicit
/// pointer whose target file **is** `home` is stored back as [`SELF`](FileId::SELF), so
/// the on-disk encoding stays portable across re-attaches (file ids are assigned per
/// attach). A pointer to any other file stays explicit; an already-`SELF` pointer (only
/// mintable through `unsafe`) is left as-is — after resolve-on-read a legitimate
/// in-memory `SELF` means "home", so writing it as `SELF` is correct.
pub fn home_relative_repr(repr: WidePtr, home: &BStack) -> WidePtr {
    if repr.is_self() {
        return repr;
    }
    if let Some(id) = id_of_host(home)
        && repr.file() == id
    {
        // Target is the home file ⇒ re-encode as `SELF`, keeping type + address.
        return WidePtr::with_parts(FileId::SELF, repr.type_id(), repr.offset());
    }
    repr
}

/// [`FileRegistry::host_arc`] on the process-wide registry — the owned-`Arc` host
/// lookup that cross-file teardown/clone use to build a [`ForeignHostAllocator`].
pub fn host_arc(id: FileId) -> Option<Arc<dyn BStackRaiiHost + 'static>> {
    REGISTRY.get()?.host_arc(id)
}

/// The id registered for `path`, if any.
pub fn id_of(path: impl AsRef<Path>) -> Option<FileId> {
    REGISTRY.get()?.id_of(path.as_ref())
}

// Path <-> bytes (exact round-trip on unix; lossy elsewhere).

#[cfg(unix)]
fn path_to_bytes(p: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    p.as_os_str().as_bytes().to_vec()
}

#[cfg(unix)]
fn bytes_to_path(b: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::OsStr::from_bytes(b).to_owned().into()
}

#[cfg(not(unix))]
fn path_to_bytes(p: &Path) -> Vec<u8> {
    p.to_string_lossy().into_owned().into_bytes()
}

#[cfg(not(unix))]
fn bytes_to_path(b: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(b).into_owned())
}
