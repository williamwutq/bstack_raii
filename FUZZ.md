# Fuzzing plan for `bstack_raii`

Fuzzing the typed object layer, modeled on `bstack`'s allocator fuzz/fault suite but
adapted to a typed, macro-generated, file-backed crate. Grounded in the generalized
shapes in [`FAILPATTERNS.md`](FAILPATTERNS.md).

## Why

`bstack`'s fuzzer works because an allocator is a flat op-stream with a cheap total
invariant (live ranges disjoint; every free hits a live range once). `bstack_raii` makes
stronger claims the happy-path tests can't exhaust, each mapping to a fuzzable oracle:

- crash-atomic clone / teardown / push / insert (WAL, `inplace_gen`, `set_batched`);
- refcount correctness (`strong`/`weak`, `try_clone`, `downgrade`/`upgrade`, `try_move`);
- the corruption-vs-leak baseline (an error/crash may leak, must never corrupt);
- the RTTI interpreter over **untrusted on-disk bytes**;
- feature *interactions* (Foreign × RTTI × niche × generics) — the hypercube.

The allocator fuzzer covers none of these.

## The throughput problem (read first)

`bstack::durable_sync` no-ops only under `bstack`'s own `cfg(all(test, debug_assertions))`;
its comment is explicit that "release builds and any dependent crate always issue the real
sync." A `bstack_raii` fuzzer is a dependent crate, so every committing op pays a real
`F_FULLFSYNC`/`fsync` — the same skip is what "takes the allocator fault fuzz from minutes
to seconds." We can't inherit it, so naive fuzzing is fsync-bound and 1–2 orders of
magnitude slower. Levers:

1. **RAM-back the files.** `/dev/shm` (Linux tmpfs); macOS needs a RAM disk
   (`diskutil erasevolume HFS+ ram `hdiutil attach -nomount ram://4096000``). One env knob on
   `TempStack` — immediate, zero coupling.
2. **Vendor `bstack` locally for the fuzz build.** `[patch.crates-io] bstack = { path =
   "vendor/bstack" }` (or a maintained fork branch) exposing the internal sync skip as an
   opt-in; fuzz-only, never shipped. Preferable to blocking on upstream: full local control,
   no dependence on a release cadence, and the durability change stays out of the published
   crate until vetted.

## Oracles (the reusable core)

Input generation is the easy half. Build these once, share across harnesses.

- **O1 — Leak-freedom.** Exists as `assert_teardown_reclaims` (`tests/hypercube/common.rs`):
  build → drop → baseline → build identical → drop → assert stack length returned (the
  constant WAL block cancels). Post-condition after every sequence. Catches the leak family
  (F2, offset-vector leaks, list-pop orphan).
- **O2 — Overlap / double-free.** Run the harness on
  `DebugCheckingAllocator<FirstFitBStackAllocator>` — tracks the exact `Range<u64>` of every
  live/freed region and panics on overlap, *partial* overlap, or double-free. Over FirstFit
  (exact-size, tightly packed at arbitrary offsets), any torn/forged/off-by-N offset is
  almost never a valid live-range start, so it trips. Needs a one-line harness `unsafe impl
  BStackRaiiAllocator` forwarding the null niche and (for H3) `wal_anchor`/`wal_file_id`/
  `atomic_bulk`/`alloc_many`/`free_many` to the inner FirstFit.

  **O2 is blind to freeing a *valid but wrong* object** — a forged offset equal to another
  live object's real start passes every allocator check. That class is O3/O4/O5's job (how
  the repros actually caught the wrong-file free: the victim's value was clobbered, not an
  overlap error). O2 and O3/O4 are complementary.
- **O3 — Refcount model.** Shadow every live block's `strong`/`weak`. After each op assert
  the on-disk counter matches, and that data frees exactly at `strong==0`, control at
  `weak==0`. Catches over/under-count from clone/teardown, niche-misread skips, mid-walk
  commits.
- **O4 — Value round-trip.** Model every POD field / vector written; after a read-preserving
  op assert the on-disk read matches (bstack's `Payload::write/verify`, lifted to typed
  fields). Catches torn writes, detached-descriptor desync, wrong-niche reads.
- **O5 — RTTI differential.** *(teardown half landed: `tests/o5_differential.rs`.)*
  Interpret the same live structure two ways — compiled typed accessors vs
  `rtti::RttiRegistry::{read_value, teardown, clone_value, move_out}` — and assert agreement
  (field values, freed-range set, clone shape). The sharpest instrument for the RTTI ×
  Foreign × niche cluster.

  The landed cut compares **freed-range sets**, not file length: build the same shape twice
  in two fresh files with identical allocation sequences, tear one down through the
  generated `bstack_drop` and the other through `RttiRegistry::teardown`, and require the
  two sets to be equal. This is fragmentation-insensitive (the defect that makes O1 invalid
  on `GhostTree`) and ignores the persistent WAL block, which the static path allocates and
  the RTTI path does not — only build-phase ranges are compared.

  It needs a **reporting** allocator, not a checking one: `DebugCheckingAllocator` panics on
  the first double free rather than enumerating divergences. `tests/o5/recorder.rs` is that
  counterpart — a `BStackRaiiAllocator` forwarding to `FirstFit` while logging every
  alloc/dealloc range, which also gives a leak set (`allocated - freed`) that works on
  backends where exact-file-length does not.

  **Equality is necessary, not sufficient**: two implementations can agree on being wrong.
  The harness therefore also asserts that between them they reclaim *everything* built,
  except the `#[bstack_ref]` targets the structure deliberately does not own. That second
  assertion is what caught a case where a 40-byte `CONTROL_SIZE` block was left behind
  by both teardowns, which on isolation turned out to be `rc.bstack_drop(&alloc)` deref'ing
  to the bare handle and freeing without decrementing. (The mistake was in the harness's own
  setup code, written while believing it released a strong reference — which is the argument
  for the severity of that entry.)

  The **clone half** is landed too: `TryCloneIn` vs `RttiRegistry::clone_value` must
  allocate the same fresh blocks and produce the same contents, with the copy read back
  *after* the original is torn down and its space scribbled over — so an aliasing clone
  cannot pass by reading stale-but-intact bytes.

  Two pitfalls cost real time here and are worth knowing before extending this:

  1. **WAL noise.** The static paths are WAL-backed; the RTTI ones bypass the WAL entirely
. So the persistent block shows up as a spurious differential — and
     not only on creation: `ClonePlan` **grows** it mid-transaction (free the old, allocate
     a larger one), so a size-based filter misses it. The harness forces the block into
     existence in a `warmup()` before the build, identifies it by reading the anchor slot
     rather than by size, and subtracts anything freed within the same window so the
     grown-from block drops out too.
  2. **The allocation log is a sequence, not a set.** Computing "what did this phase
     allocate" as `allocated_after − built` is wrong: a later allocation can reuse the exact
     slot of an earlier freed one, and set difference silently cancels the duplicate pair.
     That produced a convincing false positive — the static clone looked like it had failed
     to copy a `Vec` element, when it had merely allocated the copy into the `#[embed]`
     shell's recycled slot. `Recorder::mark`/`allocated_since` take an ordered suffix instead.

  Also note the comparison key differs between the two halves: teardown compares exact
  `(offset, len)` sets (both runs build identically and free the same blocks), while clone
  compares the **multiset of block sizes** — the two implementations legitimately order
  their allocations differently, so offsets diverge while the set of blocks does not. An
  aliasing clone still trips it: it allocates one block fewer.

  The **read half** is landed as well: `RttiRegistry::read_value`'s `Value` tree is
  reduced to the same struct the typed accessors produce, and the two must be equal. It
  is by far the cheapest of the three (~9 s vs ~60 s for clone, since it builds one
  structure and frees it) and it is the one that sees a field recorded with the wrong
  *kind*, not merely the wrong extent.

  **Every half carries a positive control**, because a green differential proves nothing
  until it has been shown able to go red. All three use the same shape — an
  un-annotated `Option<Vec<u8>>`, recorded as `Pod { width: 24 }` against a 16-byte
  `VecDesc`:

  ```
  o5_read_oracle_has_teeth
    typed accessor -> Vec [104, 101, 108, 108, 111]
    interpreter    -> Pod([64,0,…, 21,0,…, 171,0,…])   # 171 == 0xAB == the next field

  o5_teardown_oracle_has_teeth
    static       built {(64, 21), (424, 40)}  freed {(64, 21), (424, 40)}
    clone_value  built {(64, 21), (424, 40)}  freed {(424, 40)}
    -> the interpreter misses the vector data block
  ```

  These assert the divergence is *present*, so they will fail loudly if that divergence is ever fixed —
  which is the intended signal to retire them, not a regression.

  Still open on O5: the `move_out` half, and `Foreign` shapes (the cluster this oracle was
  ranked sharpest for — it needs a registry and a second file per run).
- **O6 — Crash consistency.** After a fault-injected op: reopen, `wal::finish`, re-run
  O1/O2/O3. Invariant: the recovered state is a leak-at-worst, never-corrupt version of the
  pre- or post-op state. Targets the atomicity family (WAL grow ordering, F2/F3).

## Harnesses (priority order)

### H1 — RTTI byte fuzzers (the true allocator-fuzz analog; `cargo fuzz`)

The RTTI decode/interpret input genuinely *is* an untrusted byte string, so it drops into
libFuzzer with `arbitrary`:

- `rtti_decode`: `&[u8]` → `decode_type` / `Shape::decode`. Oracle: never panic/abort/OOB;
  `Ok` or clean `Err`; `encode(decode(x)) == x` for accepted inputs. Covers the
  untrusted-length / narrow-field-truncation / unbounded-recursion / class-value-length
  cluster.
- `rtti_interpret`: seed a fixed registry (fixture schemas), run `read_value`/`teardown`/
  `clone_value` with `data` as a corrupt data file at a fuzzed root offset. Oracle: bounded
  time+memory, no OOB, no abort — the property (forged length → ~PB alloc → SIGABRT)
  violated.

Coverage-guided, deterministic, cheapest path to the RTTI feature. Seed the corpus with
valid encoded records dumped from the fixtures.

### H2 — Model-based op-sequence fuzzer (the hypercube, randomized)

Generalize the fixed hypercube to random sequences, mirroring `alloc_fuzz_tests` (an
`Arbitrary`/RNG `Op` stream + a `live` shadow + per-op verify + reopen), reusing
`tests/hypercube/{common,fixtures}.rs` verbatim for `TempStack`, the allocators, and the
type universe (`Leaf`, `Shared`, `BlockSink`, `EmbedSink`, `RcSink`, `RcWeakSink`,
`EnumSink`, `GenSink<T>`, `ConstArrSink<N>`, `RefWeakSink`, `ForeignSelfSink`, …). Run each
sequence across `DebugChecking<FirstFit>` (O2; non-bulk single-pass clone), `GhostTree`
(bulk two-pass clone), and `CheckedSlab` (placement diversity) — the WAL/clone paths diverge
by allocator.

```rust
#[derive(arbitrary::Arbitrary, Debug)]
enum Op {
    NewLeaf(u64), NewShared(u64),          // rc,weak
    CloneOwned(HandleId),                  // try_clone_in  -> O1/O2/O3
    TryClone(HandleId),                    // count bump    -> O3
    Downgrade(HandleId), Upgrade(HandleId),
    Move(HandleId),                        // bstack_move! (owned / rc try_unwrap)
    Drop(HandleId),
    ReplaceField { parent: HandleId, field: FieldSel, new: HandleId },
    SetWeak { parent: HandleId, idx: u8, target: HandleId },
    MapInsert(HandleId, u64, HandleId), MapRemove(HandleId, u64),
    DequePush(HandleId, End, HandleId), DequePop(HandleId, End),
    Reopen,                                // -> O6 boundary (finish + re-verify)
    // adversarial-but-safe (fuzz-flag gated; see below)
    ForeignAtCrossFile { src_file: FileId, into: HandleId },
    ForgedDropFromRange(HandleId),
    RttiSwapFabricated { parent: HandleId, offset: u64 },
}
```

The shadow `Model` holds, per handle: type, recursive owned children, `strong`/`weak`
counts, POD field values. Each `Op` applies to both the store and the model; O1–O5 check
agreement per step; a final teardown checks O1. proptest shrinking / libFuzzer minimization
yields a minimal reproducer — the payoff over the fixed hypercube. `Arbitrary`-derived so it
runs under both `cargo fuzz` (coverage) and a `proptest`/RNG loop (CI).

### H3 — Fault-injection layer (crash-consistency; bstack's `FaultPolicy`)

Wrap H2 with the `fault-injection` feature (debug build; `next_fault(op, seq)` is consulted
once per instrumented op, post-validation). A `FailAtSeq { at }` policy fails the `at`-th
committing op of a chosen compound (clone / teardown / push / insert / replace); sweep or
fuzz `at`, then **reopen + `wal::finish` + O6**. The only way to exercise the WAL/rollback
and mid-walk-failure patterns (F2, F3, WAL grow). `cargo fuzz`'s default profile
keeps `debug_assertions` and `overflow-checks` on, so fault-injection and the wrapping-offset
check are both live — keep them.

### H4 — Concurrency (harder)

A single `bstack` op (`inplace_gen`/`set_batched`/`cas`/`process`/`get`/`set`) holds the
file write lock for its whole duration: atomic, linearized, no intra-op interleaving, no
torn op — so modeling `inplace_gen` internals under loom is pointless. The hazards are
raii-layer sequences spanning ≥2 bstack ops **without a lock across them**, where another
thread linearizes *between* steps:

- two-phase refcount release — `fetch_sub(strong)`, then separately free-data + `fetch_sub
  (weak)`; and `upgrade`'s `increment_if_nonzero` then read `ctrl.x` (correctness rests on
  per-op atomicity and `0` being terminal, not a held lock);
- a collection `insert`'s load-factor read → `grow` → `probe_commit` window;
- `BStackVec` field growth: alloc → copy → commit-descriptor → free-old;
- the hash *set*'s deliberately two-block (bloom + table) ordering;
- the per-file WAL `Mutex` / registry `RwLock` — incl. the F4 self-deadlock (re-entering that
  `Mutex`, unrelated to `inplace_gen`).

Model each bstack op as one atomic loom step (a loom-guarded shadow of the words it touches)
and let loom permute cross-thread step order — the only nondeterminism the file lock leaves.
Complement with a thread-racing stress test (N threads `try_clone`/`drop`/`upgrade`/`insert`
on shared handles) checking O1/O3 at quiescence.

### H5 — Macro / generic fuzzing

Three targets of increasing cost; only the last needs `rustc`.

- **H5a — Proc-macro token fuzzing (in-process, fast; the macro analog of the allocator
  fuzz).** The macro is `TokenStream → TokenStream`; its input *is* tokens.
  `Arbitrary`-generate a `syn::ItemStruct`/`ItemEnum` + attrs from a **grammar of the
  supported field shapes and modes** and call `block::expand`/`enum_::expand_enum`/
  `class::expand_*` directly — no `rustc`. Oracles: never panics (a proc macro must `Err`,
  not panic — the derive has `unwrap`/`expect`/indexing a hostile shape can hit); `Ok` ⇒ the
  output parses as a `syn::File`; determinism (byte-identical output — iteration-order
  nondeterminism in the emitted `EightCC`/offsets would break the on-disk ABI); and a
  malformed input returns the *expected* `[BSTACKxxxx]` code (differential against
  `ERRORS.md`). Prerequisite: a `cargo fuzz` target can't depend on a `proc-macro` crate —
  either lift the expansion logic (already `proc_macro2`-typed) into a plain
  `bstack_raii_derive_core` lib, or run H5a as a `proptest` inside the derive
  crate's own `#[cfg(test)]` (shrinking, no coverage feedback).
- **H5b — Tag/identity fuzzing (pure function, fast; a standing property).** The tag
  pipeline (`fnv1a64` + `auto_prefix` + `build_tag` + `EightCC::mix`) is pure and already
  modeled byte-for-byte by the `generic_const_collision` repro. Fuzz random `(crate, type,
  [arg tags], [const values], prefix)` for **distinct instantiation ⇒ distinct tag** (the
  live residual collision is a counterexample), with a differential against real
  `<Foo<…>>::eightcc()` on a compiled sample to keep the model honest. The offline scan is
  just one seed strategy.
- **H5c — Generate-compile-run (slow; generated *runtime* behavior).** Emit a crate that
  builds / clones / tears down / casts / moves a generated schema, compile, run O1–O6.
  `rustc` monomorphizes each instantiation, sidestepping "can't instantiate at runtime."
  Positive: a legal schema must compile (catches over-rejection, e.g. the real `Vec<T>`-
  generic `E0392`) and pass the oracles, reaching interactions the fixed hypercube misses.
  Negative: an illegal schema must fail with the right `[BSTACKxxxx]` code (the `trybuild`/
  `ERRORS.md` contract at scale). `rustc`-bound, so biased generation on a schedule.

## Coverage map

| Pattern (FAILPATTERNS)                                      | Harness                               |
|-------------------------------------------------------------|---------------------------------------|
| Untrusted length / niche position / decode                  | **H1** + O5                           |
| Silent metadata erasure across boundary                     | **H2/O5**                             |
| Non-atomic commit, mid-walk effects, ordering               | **H3** + O6                           |
| Refcount over/under-count                                   | **H2/O3**                             |
| Owned resource consumed before last fallible step (F2)      | **H3/O1**                             |
| Ambient state not restored on unwind (F3)                   | **H3** (unwinding fault) + O1 next op |
| Overlap / double-free / torn-offset free                    | **H2/H3 on O2**                       |
| Wrong-file / wrong-target free of a *valid* object (F1)     | **H2/H3 on O3+O4**, not O2            |
| Macro panics / unparseable output / error-code drift        | **H5a**                               |
| Tag/identity collision, incl. residual                      | **H5b**                               |
| Generated over/under-rejection + runtime behavior           | **H5c**                               |

**Not reachable by fuzzing — separate tactics:**

- **Deliberate-but-safe misuse.** F1 (`Foreign::at` cross-file), safe RTTI mutators,
  `from_range` on a live handle require *intentional* footgun sequences a well-behaved
  fuzzer never emits. Encode them as the flag-gated adversarial ops in H2's `Op`, with O2 as
  the oracle — else fuzzing is silent on the crate's most interesting soundness holes.

Genuinely *hard* (not impossible) in H5: H5c is `rustc`-bound (biased generation, scheduled
runs), and H5a/H5c both need a hand-written **grammar of legal/illegal schemas** encoding
the crate's composition rules. That authoring effort — not a fundamental limit — is why the
dimension is last.

## Setup

```
fuzz/                       # cargo-fuzz crate
  Cargo.toml                # bstack_raii = { path = "..", features=["fault-injection"] }
  fuzz_targets/{rtti_decode,rtti_interpret,ops,ops_fault}.rs
tests/
  model.rs                  # shared Model + O1..O6 (also used by proptest)
  fuzz_ops_proptest.rs      # H2/H3 as a CI proptest (no libFuzzer)
  corpus/                   # valid encoded RTTI records + op scripts
```

- **Profiles.** `cargo fuzz` keeps `debug-assertions`/`overflow-checks` on — required
  (fault-injection needs `cfg(debug_assertions)`; overflow-checks makes the wrapping-offset
  pattern a visible panic). The proptest mirror runs on debug `cargo test`.
- **Backing store.** A `BSTACK_RAII_FUZZ_DIR` env knob on `TempStack::new` → tmpfs/RAM
  (lever 1); switch to ephemeral open once lever 3 lands.
- **Repro.** cargo-fuzz emits a crashing input (keep in `corpus/`); the proptest mirror
  prints its seed; `FailAtSeq` is a fixed integer.
- **CI.** The proptest mirror of H2/H3, plus H5a/H5b, per PR; libFuzzer targets (H1, long
  H2/H3) on a scheduled job with a persisted corpus.

## Rollout

1. **Done.** Lever 1 (RAM-backed `TempStack`, via `BSTACK_RAII_FUZZ_DIR`) and Lever 2
   (vendored `bstack` snapshot at `vendor/bstack`, `debug-no-sync` patched into the fuzz
   build only — see `vendor/bstack/PIN.md`).
2. **Done.** H1 (`rtti_decode`, `rtti_interpret`) — direct allocator-fuzz analog, best ROI.
   30-minute runs at ~85-93k execs/s (no-sync active) with 0 crashes across 320M execs.
3. **Partial.** Shared `Model` + O1/O2 landed as the proptest CI mirror
   (`tests/model_fuzz.rs`) over just the `Leaf`/`Shared` element pair, swept across all
   three backends (`DebugChecking<FirstFit>` with O2, `GhostTree`, `CheckedSlab` — the
   latter two via the new `unsafe impl BStackRaiiAllocator for DebugCheckingAllocator<
   FirstFitBStackAllocator>` in `src/wal.rs` plus `TempStack::checked_slab_allocator`),
   plus a behavioral stand-in for O3 (`Weak::upgrade` vs. a shadow strong count — this
   test crate can't read the real on-disk counter through the public API). The
   leak-freedom check (O1) had to become a *bounded-across-4-passes* check rather than
   exact-equality-after-one-warmup: `GhostTree`/`CheckedSlab` only shrink the file on a
   tail free, so an out-of-LIFO-order drain can legitimately oscillate between a couple
   of file lengths without leaking (confirmed by hand on `CheckedSlab`) — see
   `run_sequence`'s doc comment in `tests/model_fuzz.rs`.

   **`op_sequence_ghost_tree` is `#[ignore]`d — but it is an O1 *oracle* limitation, not a
   `bstack_raii` bug.** The earlier note here called it a control-block leak caused by
   `strong_release_ctrl` not routing through `free_many`; that diagnosis was wrong on both
   counts and has been retracted. What actually happens: the `(rc, weak)` constructor
   allocates data+control as one `alloc_many` pair, which on `GhostTree` (the only backend
   overriding to the atomic bulk path) becomes `alloc_bulk`; teardown then frees the two
   individually and in ascending address order. `GhostTree` does not coalesce or truncate
   in that case, so the file grows every cycle (+~32 B). **Nothing is leaked** — the
   regions stay tracked in `GhostTree`'s AVL and later individual `alloc`s can still be
   served from them. Confirmed upstream: this is known `GhostTree` fragmentation
   behaviour, violating no invariant or documented promise, and reproducible with pure
   `bstack` calls and no `bstack_raii` involved.

   The consequence for this harness is that **O1 as written (file length returns to
   exactly its pre-sequence value) is not a valid leak oracle on `GhostTree`** — the same
   reason it had to be relaxed to bounded-across-4-passes for `CheckedSlab`, only more so,
   since `GhostTree`'s fragmentation growth is unbounded rather than oscillating. Fixing
   the test means giving `GhostTree` a leak oracle that is fragmentation-insensitive (e.g.
   allocator-reported live bytes, or `DebugCheckingAllocator`-style region accounting),
   not changing the crate.

   Still open: O4, the rest of the `Op` enum (collections, `ReplaceField`, adversarial
   ops), and a `cargo fuzz` (`Arbitrary`) mirror for coverage-guided runs.
5. **Partial.** O5's **teardown** and **clone** halves landed as
   `tests/o5_differential.rs` (proptest, 32 shapes over a recursive `Node` fixture spanning
   owned / owned-`Option` / owned-`Vec` / strong / weak / ref / POD-`Vec` / `#[embed]` /
   nested), plus the reporting allocator at `tests/o5/recorder.rs`. Found a divergence on its first
   randomized shape.

   Six tests: three randomized differentials (read / teardown / clone), one fixed-shape
   smoke, and two positive controls. Runs ~77 s for 32 cases on a real filesystem, almost
   all of it the clone half, which builds *four* whole structures per case (two
   implementations × build + clone). Point `BSTACK_RAII_FUZZ_DIR`
   at a RAM disk before raising the case count — `debug-no-sync` removes the fsync but not
   the writes.

   One harness requirement worth stating: **backing-file names must come from a monotonic
   counter, not a timestamp.** proptest runs the tests in a binary in parallel and each
   case opens several files, so nanosecond stamps collide and two threads silently share a
   stack. That surfaced as a failure which disappeared when the binary was run alone —
   exactly the shape of bug that gets mistaken for a real finding. `TempStack` already does
   this; `tests/o5_differential.rs::tmp` now matches it.

   `tests/o5_differential.proptest-regressions` holds the shapes that failed while the
   *oracle* was still wrong (WAL noise, set-vs-sequence). They are kept deliberately: they
   are the shapes that stress the bookkeeping, and re-running them guards the oracle rather
   than the crate.
4. **Partial.** H3 (fault + O6) landed as `tests/fault_fuzz.rs` (gated behind
   `--features fault-injection`, which the plain `cargo test` run correctly compiles
   past as zero tests): a `FailAtSeq`-style `FaultPolicy` sweeps every fault point
   through a `try_clone`d + `downgrade`d `Shared`'s teardown on `DebugChecking<
   FirstFit>`, then "reopens" (a fresh allocator handle to the same file),
   `bstack_raii::finish`es, and proves the file is still fully usable by running one
   more clean lifecycle. Confirmed live (not vacuous) by instrumenting the sweep:
   faults visibly change the post-teardown file length (leaking in most positions,
   reaching the clean baseline in a few) rather than a no-op.

   The oracle here is deliberately **not** leak-freedom: `strong_release_ctrl`'s
   two-phase strong-then-weak release isn't WAL-wrapped, so a fault between its
   phases can legitimately leak the control block — `bstack_raii::finish` has
   nothing to recover there. What's actually asserted is the crate's own stated
   baseline (see O2 above): never corrupts, `DebugCheckingAllocator` would panic
   in-line otherwise. GhostTree deliberately excluded here — item 3's already-found
   bug is a corruption-adjacent leak with *no* fault involved, and mixing it in would
   confound this test's signal (fault-caused outcomes specifically).

   Still open: O5 into H2, sweeping H3 across more op shapes (not just one fixed
   `Shared` lifecycle) and more allocators once item 3's `GhostTree` bug is fixed.
5. H5b (a day, no refactor, standing property) and H5a via the `proptest`-in-derive
   path; lift into `bstack_raii_derive_core` for coverage-guided H5a.
6. Lever 3 (bstack ephemeral open) into the upstream pipeline.
7. H5c behind the schema grammar (scheduled) and H4 (concurrency), as follow-ons.
