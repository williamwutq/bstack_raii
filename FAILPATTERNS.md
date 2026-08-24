# Failure patterns

Recurring, *generalized* shapes of bug that this crate has proven prone to, with all
specifics removed. Use this as a review checklist and a place to look first when reviewing
a new feature or a feature interaction. These are patterns, not incidents: no file, line,
or concrete mechanism is named on purpose.

## On-disk / untrusted data

- **Trusting an untrusted length or count.** A length/count read from disk drives an
  allocation, a loop bound, or a walk without being validated against the containing
  block's own size — leading to out-of-bounds allocation, out-of-bounds reads into
  neighboring live blocks, or process abort. Especially in the RTTI interpreter, which
  walks fully untrusted bytes. A traversal budget that only charges for work *done* does
  not catch an oversized expansion planned up front.
- **Silent truncation of a length into a narrow field.** The encode-side mirror: writing
  a length/count/arity into a fixed-width on-disk field without checking it fits, so a
  large value is silently truncated and the record becomes mis-parsed or unreadable later.
- **Sentinel / null-niche position assumed fixed.** The location of a "none"/null sentinel
  is assumed to sit at a fixed word (e.g. the leading `u64`) when it actually depends on
  the inner type's on-disk layout (an offset-word niche vs an in-place POD niche). A
  present value is then misread as absent (skipping teardown/clone → leaks, over-counts,
  and double-frees) or a fixed-offset read runs past the block.
- **Weak identity instead of a structural fingerprint.** Cross-version or cross-instance
  reconciliation validates a cheap identity (a name, or a short hash tag) rather than a
  digest of the actual layout/shape, so an incompatible layout change is silently accepted
  and old offsets are applied to new-layout data.

## Type identity / metadata across boundaries

- **Silent erasure of critical metadata across a feature boundary.** Information that is
  meaningful to one feature is dropped when a value crosses into a representation that has
  no slot for it, then a round-trip through that representation erases it permanently.
  Narrowly: pointer metadata (an RTTI type ordinal on a `Foreign`) wiped whenever the
  pointer is rebuilt through a cast/repr path that constructs a bare pointer.
- **Insufficient or lossily-combined identity entropy.** A type-identity discriminant
  (hash/tag) that has too few discriminating bits, or that combines its parts with a lossy
  or commutative fold, collides for genuinely distinct types (permuted parameters, or
  distinct const/size parameters that change the on-disk width). When that discriminant is
  the *sole* gate for a safe cast/dispatch, a wrong-type or wrong-size reinterpretation
  passes and reads/frees run past the real block.

## Safety-obligation leaks

- **An `unsafe` obligation re-exposed as safe at a higher layer.** A capability is
  `unsafe` at the low level precisely because an offset/pointer can't be validated; a
  higher-level surface (RTTI, registry, a convenience constructor) re-offers the same
  capability as a *safe* API without performing the missing validation — so safe code can
  install a bogus offset that a later, entirely safe teardown frees.
- **A raw, unchecked constructor next to a safe destructive method.** An unvalidated
  handle/range constructor paired with a safe free/teardown lets safe code fabricate a
  handle over a live or aliased region and free it (use-after-free / double-free).
- **Context-relative references escaping their context.** A reference that resolves against
  the "current" scope/file rather than a fixed identity can be stored or carried into a
  *different* scope than it was minted for, then resolves against the wrong target — a
  wrong-scope read or free.

## Atomicity, ordering, and side-effect timing

- **Non-atomic commit of one logical value.** A single logical write is committed in
  several sub-steps (e.g. byte-by-byte, or metadata split from payload), so a crash or
  error between sub-steps leaves a torn/misaligned state; if the torn value is an
  offset/pointer, a later safe teardown frees a garbage range.
- **Irreversible side effects committed mid-operation.** Destructive effects (refcount
  decrements, cross-file frees) are applied *as a walk proceeds*, before the whole
  operation is known to succeed. A mid-operation failure leaves the structure "partly torn
  but reachable," and a retry re-applies them — premature free, double-free, over-decrement.
  The safe alternative is to collect intended effects during a read-only pass and commit
  them only after full validation.
- **Reclaiming before the replacement is durably linked.** Freeing or repointing a
  resource before its replacement is committed (violating allocate-before-free /
  link-before-free), so a crash in the window leaves a reference to freed space.
- **A precondition guarded only by a debug assertion.** A critical invariant is checked
  with a debug-only assert that is compiled out in release, so a violated precondition
  becomes a silent out-of-bounds write / UB in release instead of a hard error.

## Ownership / resource lifecycle

- **Consuming an owned resource before the last fallible step.** A fallible operation
  defuses or consumes an owned handle up front, then does more fallible work; on error the
  resource is neither stored, returned, nor freed — an orphan (leak) — instead of being
  handed back. Inconsistent with the crate's "hand the value back on failure" contracts
  elsewhere.
- **Ambient state not restored on unwind.** Thread-local / ambient context installed for
  the duration of an operation is torn down only on the normal return path, so a panic
  leaves it installed and corrupts the next operation on the same thread.

## Concurrency / re-entrancy

- **Re-entrant acquisition of a non-reentrant lock keyed by identity.** A per-resource lock
  is re-acquired when an operation re-enters the *same* resource through a different handle
  or adapter (a self-reference or a cross-file cycle that routes back to the origin file),
  causing self-deadlock. The same identity-keyed scheme can also fail the other way — two
  distinct handles onto one resource hashing to different locks and not serializing.
- **A bounded traversal that loses its bound at a boundary.** A walk that is
  non-recursive/budgeted/cycle-guarded *within* a scope becomes natively recursive at a
  feature/file boundary and restarts its budget and visited-set each hop — so cross-boundary
  cycles or depth are never caught (stack overflow / unbounded work).

## Consistency of parallel implementations

- **One code path in a family missing a check its siblings enforce.** Among a set of
  parallel mutators/accessors (per ownership kind, per container shape, per variant), one
  variant omits a bounds check, a validation, or a write-back binding that all the others
  have — an out-of-bounds write, an unvalidated offset, or a mutation that silently
  fails to persist.
- **A bidirectional map left inconsistent on replace/remove.** A forward and reverse index
  are not updated together: a reverse entry outlives its forward entry (or a replace leaves
  the old entry behind), so lookups resolve to a stale or wrong target.

## Arithmetic

- **Division / modulo / offset math on a value that can be zero or wrap.** A size or stride
  that can be zero (a zero-sized element) drives a division/modulo, or an index/offset
  computation wraps, turning an otherwise-safe call into a panic or a wrapped
  out-of-range access.
