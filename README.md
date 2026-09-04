# bstack_raii

Typed, RAII-style ownership for persistent objects — `Rc`/`Weak` semantics that
survive a process restart or crash, backed by [`bstack`] files.

`std::rc::Rc` and `Weak` live and die with the process. `bstack_raii` gives you
the same model, but the object graph *and its reference counts* live on disk,
crash-safely. You define blocks as ordinary structs and enums; the
[`#[bstack_block]`](#structs) / [`#[bstack_enum]`](#enums-bstack_enum) macros
generate the on-disk layout, typed accessors, constructors, recursive teardown,
and refcounting.

On top of that core it grows a full object layer:

- **Five ownership kinds** — [`#[bstack_owned]`, `#[embed]`, `#[bstack_strong]`,
  `#[bstack_weak]`, `#[bstack_ref]`](#field-ownership) — each with matching
  deep-clone ([`TryCloneIn`](#cloning-tryclonein--tryclone)), recursive teardown,
  and [`bstack_move!`](#moving-out-bstack_move) semantics.
- **Rich field shapes** — growable [`Vec` / `String`](#vectors-and-strings),
  fixed-size [`[T; N]` arrays](#fixed-size-arrays-t-n), nullable
  [`Option`](#nullable-fields-option), POD tuples, and
  [generic blocks](#generic-blocks), composed to depth.
- **[Cross-file pointers](#cross-file-pointers-foreignt) (`Foreign<T>`)** — a wide
  pointer naming a target file *and* an offset in it, so an object graph can span
  many `bstack` files, resolved through a process-wide file registry.
- **Crash-safe compound ops** — clone and teardown commit as one atomic batch and,
  on a WAL-anchoring allocator, reclaim crash-orphaned allocations on the next open.
- **[Standard-library collections](#standard-library-collections)** — persistent
  `HashMap`, `BTreeMap`, `VecDeque`, `LinkedList`, `String`, and more.
- **A self-describing on-disk schema (RTTI)** — an optional type-registry stack
  (see [BYTECODE.md](BYTECODE.md)) letting a general program interpret bstack_raii
  structures on disk with no compiled-in Rust types.

It layers on the mainline `bstack` allocator (`BStackRange` / `BStackSlice` /
`BStackOwnedSlice`), which supplies the atomicity, crash-safety, and
single-ownership primitives underneath.

> **Status:** feature-complete and tested, but the on-disk ABI is not yet
> stable (pre-1.0).

## Contents

- [Quick start](#quick-start)
- [Concepts](#concepts)
- [How it works on disk](#how-it-works-on-disk)
- [Handles & lifetimes](#handles--lifetimes)
  - [Type naming](#type-naming)
- [Generated types](#generated-types)
- [Blocks](#blocks)
  - [Field ownership](#field-ownership)
  - [Structs](#structs)
  - [Reference-counted blocks](#reference-counted-blocks)
  - [Vectors and strings](#vectors-and-strings)
  - [Fixed-size arrays: `[T; N]`](#fixed-size-arrays-t-n)
  - [Nullable fields: `Option`](#nullable-fields-option)
  - [Enums: `#[bstack_enum]`](#enums-bstack_enum)
  - [Field types](#field-types)
  - [Generic blocks](#generic-blocks)
- [Mutating fields: `#[bstack_mut]`](#mutating-fields-bstack_mut)
- [Moving out: `bstack_move!`](#moving-out-bstack_move)
- [Cloning: `TryCloneIn` / `TryClone`](#cloning-tryclonein--tryclone)
- [Casting: `bstack_cast!`](#casting-bstack_cast)
- [Cross-file pointers: `Foreign<T>`](#cross-file-pointers-foreignt)
- [Type tags (`EightCC`)](#type-tags-eightcc)
- [Standard library collections](#standard-library-collections)
- [Examples](#examples)
- [Limitations](#limitations)
- [Error codes](#error-codes)

## Quick start

Add both crates (`bstack` supplies the allocator; `bstack_raii` the object
layer):

```toml
[dependencies]
bstack_raii = "0.1"
bstack = "0.4"
```

```rust
use std::io;
use bstack::FirstFitBStackAllocator;
use bstack_raii::{BStack, BStackAllocator, BStackDrop, TryClone, bstack_block};

// A shared, reference-counted, weak-observable block.
#[bstack_block(rc, weak)]
struct Config {
    version: u64,
    flags: u64,
}

// A block that owns a *strong* reference to a shared Config.
#[bstack_block]
struct Session {
    id: u64,
    #[bstack_strong]
    config: Config,
}

fn main() -> io::Result<()> {
    let alloc = FirstFitBStackAllocator::new(BStack::open("app.bstack")?)?;
    let stack = alloc.stack();

    let config = Config::new(&alloc, 3, 0b1010)?;      // BStackRc<Config>, strong = 1
    let session = Session::new(&alloc, 0, config.try_clone()?)?; // strong = 2

    // Read fields through generated accessors.
    let cfg = session.handle().get_config(stack)?;      // -> a Config handle
    println!("v{} flags {:#b}", cfg.get_version(stack)?, cfg.get_flags(stack)?);

    drop(config);                  // strong = 1 — the session still owns it (Rc: auto-decrement)
    session.bstack_drop(&alloc)?;  // strong = 0 — Config freed automatically by its refcount
    Ok(())
}
```

Two things are load-bearing here, both covered below:

- **Owned vs. shared teardown.** A uniquely-owned `Session` handle frees
  **nothing on `Drop`** — free it explicitly with `bstack_drop`, so a persistent
  root is never silently deleted. A shared `Config` (`BStackRc`) *does*
  auto-manage its refcount on `Drop`. See [Handles & lifetimes](#handles--lifetimes).
- Every block also gets a stable 8-byte on-disk type tag — see
  [Type tags](#type-tags-eightcc).

A fuller walk-through is in [`examples/sessions.rs`](examples/sessions.rs):
`cargo run --example sessions`.

## Concepts

A **block** is a fixed-size record on disk. You write it as an ordinary `struct`
(or [`enum`](#enums-bstack_enum)) and annotate it; the macro generates a parallel
`#[repr(C, packed)]` on-disk layout and its machinery.

Both macros take the same three **modes**:

| Mode         | Meaning                                                  |
|--------------|----------------------------------------------------------|
| *(none)*     | Plain, exclusively owned (like `Box`).                   |
| `(rc)`       | Reference-counted, inline count (like `Rc`, no `Weak`).  |
| `(rc, weak)` | Refcounted **and** weak-observable (like `Rc` + `Weak`). |

Every non-POD field carries an [ownership annotation](#field-ownership) deciding
how it is torn down. `Pod` fields are stored inline and copied by value.

> **The allocator bound: [`BStackRaiiAllocator`].** Every operation is generic over
> [`BStackRaiiAllocator`], an `unsafe` trait over a freeing `bstack` allocator that
> asserts the **null niche**: offset 0 is never handed out, so a `0` offset reads as
> "none" everywhere. It also exposes an *optional* WAL anchor that lets teardown and
> clone reclaim crash-orphaned allocations on the next open.
>
> **Every bstack-provided allocator implements it** — `FirstFitBStackAllocator`,
> `SlabBStackAllocator`, `GhostTreeBstackAllocator`, … — and a custom one opts in
> with `unsafe impl BStackRaiiAllocator for MyAlloc {}`. **Not `LinearBStackAllocator`**:
> its `dealloc` is a no-op and it can hand out offset 0. For growable fields use a
> **realloc-safe** allocator; `FirstFitBStackAllocator` is one.
>
> **The WAL anchor in practice.** Call [`wal::finish`]`(&allocator)` once after `open`
> to complete any transaction a prior crash left in flight; `try_clone_in` and
> `bstack_drop` also call it opportunistically. [`STD_WAL_ANCHOR`] is the offset
> every bstack-provided allocator reserves.

## How it works on disk

Every block begins with a 16-byte `BlockHeader { size: u64, tag: EightCC }` (the
[tag](#type-tags-eightcc) is the downcast discriminant). References are bare `u64`
offsets; a target's length is recovered from `size_of::<T::OnDisk>()`, so blocks
are **fixed-size**.

The three modes differ only in what is injected after the header:

- **`#[bstack_block]`** — nothing; the payload follows the header directly.
- **`(rc)`** — an inline `refcount: u64`. The block is freed when it hits zero.
- **`(rc, weak)`** — a `ctrl` back-pointer to a separate **control block**
  (`XOnDiskRef`) holding `strong` / `weak` counters and a forward pointer to the
  data. The data block is reclaimed when `strong` hits zero; the control block
  persists until `weak` also hits zero, so a `Weak` can outlive the data — like
  `Arc` / `Weak`.

**Growable fields** (`Vec` / `String`) don't fit a fixed-size block, so they live
out of line: the field holds a 16-byte descriptor `{ data_off, data_size }`
*inline*, pointing at a separate data block that reallocates as it grows (see
[Vectors and strings](#vectors-and-strings)).

Refcount updates are single-lock read-modify-writes on `bstack` (crash-atomic).
Everything is durable and speaks [`std::io::Result`].

## Handles & lifetimes

The typed handle `X` is a bare `Copy` `(offset, len)` with no allocator; you read
fields through it. Ownership wrappers layer on top:

| Handle               | Allocator | Ownership                        | Teardown                                                             |
|----------------------|-----------|----------------------------------|----------------------------------------------------------------------|
| `X` (the block type) | no        | none (borrowed view / bare ref)  | `x.bstack_drop(alloc)?` — explicit only                              |
| `BStackOwned<X>`     | no        | exclusive (ownership marker)     | `owned.bstack_drop(alloc)?` — **nothing on `Drop`**                  |
| `AutoDrop<T>`        | yes       | RAII guard over any `BStackDrop` | runs `bstack_drop` on Rust `Drop`                                    |
| `BStackRc<X>`        | yes       | shared strong                    | `try_clone` / `downgrade`; **auto-decrements on `Drop`**, frees at 0 |
| `BStackWeak<X>`      | yes       | none (keeps control block alive) | `try_clone` / `upgrade`; auto-decrements weak on `Drop`              |
| `BStackRef<X>`       | no        | none (raw offset)                | none                                                                 |

**Owned is manual; shared is automatic.** A `BStackOwned<X>` carries no allocator
and frees **nothing** when it drops. Free it explicitly, or wrap it in an
`AutoDrop` guard for RAII:

```rust
let owned: BStackOwned<Node> = Node::new(&alloc, /* … */)?;
let value = owned.handle().get_tag(stack)?;   // read a field (or `owned.get_tag(stack)?` via Deref)

owned.bstack_drop(&alloc)?;               // free it now, explicitly …
// … or: let _guard = owned.auto(&alloc); // RAII — freed when `_guard` drops
```

Shared handles (`BStackRc` / `BStackWeak`) manage their counts on `Drop`, like
`std::rc`. Duplicating one bumps an on-disk counter, so cloning is the [`TryClone`]
trait, not `Clone`. `BStackRc` derefs to `X`; a `BStackWeak` doesn't — `upgrade` it
to a `BStackRc` first.

### Type naming

The **`BStack`** prefix marks the primary tier: the handles (`BStackOwned`,
`BStackRc`, `BStackWeak`, `BStackRef`), the capability traits (`BStackBlock`,
`BStackDrop`, `BStackShared`, …), and the [collections](#standard-library-collections)
(`BStackHashMap`, `BStackString`, …). Everything else drops the prefix:

- **Low-level ref / drop-core tokens** — `OwnedRef`, `StrongRef`, `WeakRef`,
  `VecRef`, `AnyRef` (`BStackRef` keeps the prefix — it's the typed pointer, not a
  token).
- **Module-namespaced subsystems** — RTTI (`RttiType`, `Shape`, `Value`, …) and
  cross-file pointers (`Foreign`, `ForeignOwned`, …).
- **On-disk primitives** — `EightCC`, `Offset`, `WidePtr`, `FileId`.

## Generated types

Each macro generates a small, fixed set of types (for a block named `X` / `E`):

| Source                               | Types generated                                                                                                                                                |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `#[bstack_block] struct X`           | `X` — the [handle](#handles--lifetimes); `XOnDisk` — the `#[repr(C, packed)]` on-disk payload                                                                  |
| `#[bstack_block(rc, weak)] struct X` | the above, plus `XOnDiskRef` — the [control block](#how-it-works-on-disk) (`strong`/`weak` counters)                                                           |
| `#[bstack_enum] enum E`              | `E`, `EOnDisk` (plus `EOnDiskRef` for `(rc, weak)`), and two companion enums — see [Enums](#enums-bstack_enum): `EData` (owned form) and `EView` (read result) |

Alongside the types come the trait impls (`BStackBlock`, `BStackDrop`,
`BStackCast`, `BStackMove`, and for rc modes `BStackShared` / `BStackWeakable`)
and inherent methods (`new`, field accessors, `set_<field>`, enum `read`, …). You
name only the handle (`X` / `E`) and `EData` / `EView` for enums.

## Blocks

A block is a `struct` or `enum` annotated with `#[bstack_block]` /
`#[bstack_enum]`. This section covers how fields and variants are declared.

### Field ownership

Every non-POD field carries exactly one annotation, which decides its teardown
and what [`bstack_move!`](#moving-out-bstack_move) yields:

| Annotation         | Child kind required    | On teardown                         | `bstack_move!` yields       |
|--------------------|------------------------|-------------------------------------|-----------------------------|
| `#[bstack_owned]`  | any block              | recursively frees the child         | `BStackOwned<T>`            |
| `#[embed]`         | any block              | frees the child's children in place | `BStackOwned<T>` (re-homed) |
| `#[bstack_strong]` | `(rc)` or `(rc, weak)` | decrements refcount; frees at zero  | `BStackRc<T>`               |
| `#[bstack_weak]`   | `(rc, weak)`           | decrements weak count only          | `Option<BStackWeak<T>>`     |
| `#[bstack_ref]`    | any block              | nothing                             | `BStackRef<T>`              |
| *(none)* — POD     | `Pod` type             | nothing (inline)                    | the value                   |

Rules are enforced at compile time. A `#[bstack_weak]` field whose target isn't
`(rc, weak)` or an unannotated non-`Pod` field is a compile error.

#### `#[embed]` — inline a child block

`#[bstack_owned]` stores a `u64` **offset** to a separately-allocated child.
`#[embed]` instead stores the child's *whole on-disk form* **inline** in the
parent, so it needs no separate allocation:

```text
#[bstack_owned]:  [ parent header ][ .. u64 offset .. ] ─▶ [ child header ][ child fields ]
#[embed]:         [ parent header ][ child header ][ child fields ][ .. ]
```

```rust
#[bstack_block]
struct Holder {
    #[embed] child: Child,   // Child's OnDisk lives here, inline
    tag: u32,
}
enum Wrapper {
    #[embed] One(Child),     // also works as an enum variant
    None,
}
```

It's still **exclusive ownership**: `new` takes a `BStackOwned<Child>`, folds its
bytes in, and frees the child's redundant shell (its own children stay live). The
accessor hands back a borrowed `Child` handle; teardown frees the embedded child's
children in place; `bstack_move!` re-homes the child to a fresh `BStackOwned<Child>`.
You can embed any block, but not a tuple, `Vec`, or `Option`.

### Structs

```rust
#[bstack_block]
struct Node {
    #[bstack_owned]  payload: Payload,    // exclusively owns the child
    #[bstack_strong] shared:  SharedThing,// a shared, refcounted reference
    #[bstack_weak]   parent:  Node,       // a non-owning back-pointer (may dangle)
    #[bstack_ref]    sibling: Node,       // a raw reference, no ownership
    tag: u32,                             // POD, stored inline
}
```

Besides the [generated types](#generated-types), the macro emits inherent
methods:

- a `new(...)` **constructor**, consuming the child handles (`#[bstack_owned]` →
  `BStackOwned<T>`, `#[bstack_strong]` → `BStackRc<T>`, `#[bstack_ref]` →
  `BStackRef<T>`, POD by value; `#[bstack_weak]` fields are **not** parameters);
- **accessors** — `node.get_field(stack)` reads each field;
- **mutators** — writing a field is opt-in per
  [`#[bstack_mut]`](#mutating-fields-bstack_mut) (`set_<field>` / `replace_<field>`),
  plus a `set_<field>` for wiring each `#[bstack_weak]`
  [back-pointer](#reference-counted-blocks);
- recursive teardown, [casting](#casting-bstack_cast), and
  [moving](#moving-out-bstack_move).

A **tuple struct** works if every field is `Pod`: positional fields get synthetic
names, so `struct Rgb(u8, u8, u8)` is built `Rgb::new(&alloc, 10, 20, 30)`, read via
`rgb.get_field0(stack)?` / …, and `bstack_move!` returns the fields in order. A
**unit struct** (`#[bstack_block] struct Marker;`) is a non-empty **header-only** block.

### Reference-counted blocks

Declare `#[bstack_block(rc)]` / `#[bstack_block(rc, weak)]`. `new` returns a
`BStackRc<X>`; clone it with `try_clone`, and for `(rc, weak)` get a non-owning
observer with `downgrade` / `upgrade`:

```rust
let a = Config::new(&alloc, 1, 0)?;   // BStackRc<Config>, strong = 1
let b = a.try_clone()?;               // strong = 2
let w = a.downgrade()?;               // BStackWeak<Config>
drop(a); drop(b);                     // strong = 0 — the data is freed
assert!(w.upgrade()?.is_none());      // the object is gone
```

**Weak fields** starts null and are wired afterward with
`set_<field>(&alloc, target.downgrade()?)`; the accessor `get_<field>(&alloc)`
*upgrades* to `Option<BStackRc<T>>`. The field stores the target's *control-block*
offset, so dropping the strong owner first and the holder second is sound.

### Vectors and strings

A `Vec<T>` (POD `T`) or `String` field stores a growable sequence, backed by the
inline descriptor from [How it works on disk](#how-it-works-on-disk).

```rust
#[bstack_block]
struct Record {
    name: String,
    tags: Vec<u32>,      // POD vectors are un-annotated (any Pod element type)
    id: u64,
}

let rec = Record::new(&alloc, "hello", &[1u32, 2, 3], 42)?;  // &str / &[T] / value
let mut tags = rec.handle().get_tags(&alloc)?;    // a BStackVec<u32> handle
tags.push(4)?;                                    // grows; rewrites the inline descriptor
assert_eq!(rec.handle().get_tags(&alloc)?.to_vec()?, vec![1, 2, 3, 4]);
```

The accessor returns a [`BStackVec<T>`] (`len` / `to_vec` / `push`); freeing the
block frees the data. A *detached* vector (from `BStackVec::from_slice` or
`bstack_move!`) persists only once written into a field
([moved-out-is-unrooted](#moving-out-bstack_move)).

When the elements are `#[bstack_block]` values, the **annotation** states the
elements' ownership; an **un-annotated** `Vec<T>` is always POD (`T: Pod`):

| Field                                | Element handle       | Accessor                 | On the struct's teardown                              |
|--------------------------------------|----------------------|--------------------------|-------------------------------------------------------|
| `Vec<T>` / `String` *(un-annotated)* | POD value (`T: Pod`) | `BStackVec<T>`           | frees the data block                                  |
| `#[bstack_owned] Vec<Thing>`         | `BStackOwned<Thing>` | `BStackBlockVec<Thing>`  | recursively frees every child, then the offset array  |
| `#[bstack_strong] Vec<Thing>`        | `BStackRc<Thing>`    | `BStackStrongVec<Thing>` | releases each strong ref (frees at 0), then the array |
| `#[bstack_weak] Vec<Thing>`          | `BStackWeak<Thing>`  | `BStackWeakVec<Thing>`   | releases each weak ref, then the array                |
| `#[bstack_ref] Vec<Thing>`           | `BStackRef<Thing>`   | `BStackRefVec<Thing>`    | frees the offset array only                           |

The constructor takes a `Vec` of the matching element handle; the accessor
returns the vector handle (`len` / `to_vec` / `get`; `BStackWeakVec` has
`upgrade(i)`; each has a `push_*`).

A vector's element may itself be a [fixed-size array](#fixed-size-arrays-t-n):
`Vec<[T; N]>`, POD or any annotated kind. But an element must be a single leaf:
`Vec<Vec<T>>` and `Vec<(A, B)>` are rejected.

To **share** a vector, wrap it in its own `#[bstack_block]` and share *that* with
`#[bstack_strong]` / `#[bstack_ref]`.

### Fixed-size arrays: `[T; N]`

A fixed-size array `[T; N]` is stored **inline**. The annotation states the
elements' ownership; an un-annotated array of `Pod` is itself `Pod`.

```rust
#[bstack_block]
struct Board {
    cells: [u16; 9],                    // POD array (un-annotated) — inline bytes
    #[bstack_owned] tiles: [Leaf; 3],   // 3 owned children (freed on teardown)
    #[bstack_ref]   marks: [Leaf; 2],   // 2 borrowed refs (free nothing)
    #[embed]        kids:  [Child; 2],  // 2 children embedded verbatim, inline
}

let b = Board::new(&alloc, [0; 9], [a, b, c], [r0, r1], [k0, k1])?;
let tiles: [Leaf; 3] = b.handle().get_tiles(stack)?;   // an array of block views
```

A reference array stores `[u64; N]` inline. The constructor takes an array of the
matching handle, the accessor hands back `[T; N]` block views, and teardown
frees/releases each element per the annotation. `#[bstack_weak]` is wired per index
(`set_field(&alloc, i, weak)`) and upgrades each slot to `[Option<BStackRc<T>>; N]`.
`#[embed] [Child; N]` stores the N children's on-disk forms back-to-back.

Arrays compose freely:

- **Per-element `Option`** — `[Option<T>; N]` makes each slot nullable; a
  whole-array `Option<[T; N]>` is rejected (put the `Option` on the element).
- **Nesting to any depth** — `[[T; N]; M]`, `[[[T; N]; M]; K]`, … work for every
  kind (POD / owned / strong / weak / ref / embed).

### Nullable fields: `Option`

Wrap a reference or vector field in `Option` to make it nullable. On disk it's
unchanged — a `0` offset is `None`; an *empty* present vector still has a non-zero
data block, distinct from `None`:

```rust
#[bstack_block]
struct Node {
    #[bstack_owned]  left: Option<Child>,     // may be absent
    #[bstack_strong] shared: Option<Thing>,
    labels: Option<Vec<u32>>,                 // nullable POD vector
}
```

The accessor returns `io::Result<Option<_>>`, the constructor takes an `Option`
(`Option<BStackOwned<Child>>` / `Option<&[T]>` / …), and `bstack_move!` yields
`Option<_>`. (`#[bstack_weak]` fields are already nullable.)

An `Option` on an **un-annotated POD** field is different: `Option<A>` is stored
*inline* whenever `A: bytemuck::PodInOption` (so `Option<A>: Pod`) and handed
back by value. A plain `Option<u32>` is *not* `Pod` and is rejected.

`Option` *is* a Rust `enum`, but this is a macro niche optimization and **not** a
[`#[bstack_enum]`](#enums-bstack_enum) (no discriminant byte, no `EData` / `EView`,
no extra block). Any other sum type is a `#[bstack_enum]`.

### Enums: `#[bstack_enum]`

A `#[bstack_enum]` lowers a Rust `enum` to a **tagged-union block**: a
discriminant plus a payload sized to the largest variant. A variant is either a
**POD aggregate** or an **annotated single-field tuple** whose annotation works
like a struct field's annotation:

```rust
#[bstack_enum]
enum Node {
    Empty,                            // unit (POD aggregate, 0 fields)
    Num(u32),                         // POD, inline
    Rect { w: u32, h: u32 },          // POD struct variant, packed inline
    #[bstack_ref]    Link(Leaf),      // borrowed reference (frees nothing)
    #[bstack_owned]  Child(Leaf),     // owned child (freed on teardown)
    #[bstack_strong] Shared(Thing),   // a strong ref (Thing is (rc)/(rc, weak))
    #[bstack_weak]   Watch(Thing),    // a weak ref (Thing is (rc, weak))
}

let node = Node::new(&alloc, NodeData::Child(leaf))?;   // construct a variant
match node.handle().read(&alloc)? {                     // read / match it
    NodeView::Child(c) => assert_eq!(c.get_val(stack)?, 7),
    _ => {}
}
node.bstack_drop(&alloc)?;                              // frees the owned child too
```

The two [companion enums](#generated-types) are directional duals:

- **`NodeData`** — the in-memory *owned* form (POD by value; `#[bstack_owned]` →
  `BStackOwned<T>`, `#[bstack_strong]` → `BStackRc<T>`, `#[bstack_weak]` →
  `BStackWeak<T>`, `#[bstack_ref]` → `BStackRef<T>`). The **same** type is passed
  to `new` and returned by [`bstack_move!`](#moving-out-bstack_move).
- **`NodeView`** — the read result: POD by value, owned/ref children as borrowed
  handles, a weak variant *upgraded* to `Option<BStackRc<T>>`. `read` takes the
  allocator.

A single-field variant carries the same shapes a struct field does:

```rust
#[bstack_enum]
enum Cell {
    Empty,
    Tags(Vec<u32>),                          // POD vector variant
    Text(String),                            // POD string variant
    #[bstack_owned] Kids(Vec<Leaf>),         // owned vector (freed on teardown)
    #[bstack_ref]   Row([Leaf; 3]),          // inline reference array
    #[bstack_owned] Grid(Vec<[Leaf; 2]>),    // vector of reference-arrays
}
```

An array variant `V([T; N])` mirrors a scalar `V(T)` per element; a vector variant
`V(Vec<…>)` stores a descriptor. The same nesting and directed-error rules apply as
for struct fields.

Like a struct, an enum has [modes](#concepts): `#[bstack_enum(rc)]` / `(rc, weak)`
make the enum itself refcounted / weak-observable (`new` returns `BStackRc<E>`), so
it can be a `#[bstack_strong]` / `#[bstack_weak]` field. [`bstack_move!`](#moving-out-bstack_move)
and [`bstack_cast!`](#casting-bstack_cast) work as on structs. Moving frees the
shell and hands the active variant out through `NodeData`.

An enum is a block, so it is **always referenced** like a struct. A POD aggregate
variant's fields must all be `Pod` and it takes no annotation. Duplicate
discriminant values are a clear compile error.

#### Discriminant width

The discriminant defaults to the **smallest integer** that fits every variant's
value, honors explicit `= value` discriminants, and chooses a **signed** type
if any value is negative.

Pin it with `repr(..)` — `#[bstack_enum(repr(u16))]` (any of
`u8|u16|u32|u64|i8|i16|i32|i64`; `usize`/`isize` are rejected). `repr(aligned)` is
`repr(u64)`, letting a variant's on-disk `u64` ref gets aligned writes.

Enums take the same tag controls as structs: `tag = "…"`, `ctrl_tag = "…"` (for
`(rc, weak)`), and `allow(overlong_tag)` — e.g.
`#[bstack_enum(repr(u64), rc, weak, tag = "NODE")]`.

#### Default variant

Mark one **unit** variant `#[default]` to generate `impl Default for <Enum>Data`
returning it. `Default::default()` takes no allocator, so only a **unit** variant
qualifies (`[BSTACK0206]` otherwise), and at most one may be marked (`[BSTACK0207]`).
Then `Node::new(&alloc, NodeData::default())?` builds that variant.

### Field types

`Vec<T>`, `String`, `Option<…>`, and `[T; N]` in a field are **recognized
spellings**, not the `std` types — the macro lowers each to a bstack_raii on-disk
form. They **compose**, with two directed-error limits: at most one `Option` layer
(`Option<Option<T>>` → use a `#[bstack_enum]`), and a `Vec` element must be a single
leaf (`Vec<Vec<T>>` → a named `#[bstack_block]`).

A **POD tuple** field — `a: (A, B, …)`, every element `Pod` — also works: stored
through a generated packed wrapper and handed back as a tuple. `bstack_move!` keeps
each tuple as **one** element (a `(u8, u8)` comes back whole, not flattened). A
tuple is *not* a valid `Vec`/array element — `Vec<(A, B)>` is rejected in favor of
a named `#[bstack_block]`.

A field written `&T` is coerced to owned `T` (and `&str` to `String`) with a
compile warning; silence it with `#[bstack_block(allow(coerced_ref))]`.

### Generic blocks

A `#[bstack_block]` / `#[bstack_enum]` may be **generic** over type parameters
(and, for arrays, `const` parameters). Each concrete instantiation is its own block
type and [type tag](#type-tags-eightcc).

```rust
#[bstack_block]
struct Node<T> {
    #[bstack_owned] child: T,          // owns a child of any block type
    #[bstack_ref]   refs:  [T; 3],     // an array of references to T
    weight: u32,
}

#[bstack_block]
struct Buf<const N: usize> { data: [u16; N], len: u32 }   // const-length POD array

let n = Node::<Leaf>::new(&alloc, leaf, [r0, r1, r2], 5)?;
let b = Buf::<8>::new(&alloc, [0; 8], 0)?;
```

A type parameter works in **every** field shape it makes sense in:

- **Reference kinds** — `#[bstack_owned/strong/weak/ref]` (scalar, `Vec<T>`,
  `[T; N]`, `Vec<[T; N]>`, …): the on-disk form is a bare `u64` offset, so the
  layout is independent of `T`. The parameter is auto-bounded `BStackBlock` (plus
  `BStackShared` / `BStackWeakable` for a strong / weak use).
- **Inline kinds** — a POD field (`item: T`, `T: Pod`) or `#[embed] item: T`
  (`BStackBlock`): `T` is stored *inline*, so `XOnDisk` becomes generic over it. A
  parameter can't be used **both** as POD and as a reference.
- **`const N`** in an array length — `[T; N]` (single dimension). A nested
  `[[T; N]; M]` with a const dimension is rejected; make one dimension concrete.

Each instantiation folds its arguments into the tag, so
[`bstack_cast!`](#casting-bstack_cast) can't confuse `Node<A>` with `Node<B>`. A
generic **enum** is supported in the layout-preserving case only: a type parameter
may appear in a reference variant (`#[bstack_owned] Branch(T)`), not in a
POD/`#[embed]` variant.

Currently unsupported (a clear compile error): lifetime parameters, a generic
block in `rc` / `rc, weak` mode, and const parameters in a generic *enum*.

## Mutating fields: `#[bstack_mut]`

Every scalar field gets a reader (`get_<field>`). *Writing* one is opt-in: mark it
`#[bstack_mut]` and the macro adds the mutator for its kind. Immutability is the
default, and each generated write is a single crash-atomic `set`.

```rust
#[bstack_block]
struct Counter {
    #[bstack_mut] hits: u64,   // writable
    created_at: u64,           // read-only (no setter generated)
}

let c = Counter::new(&alloc, 0, now)?;
c.handle().set_hits(stack, 42)?;             // atomic overwrite
assert_eq!(c.handle().get_hits(stack)?, 42);
```

The mutator depends on the field's ownership:

| Field kind         | Mutator                              | Semantics                                                                     |
|--------------------|--------------------------------------|-------------------------------------------------------------------------------|
| POD                | `set_<field>(stack, value)`          | overwrite the inline bytes                                                    |
| `#[bstack_ref]`    | `set_<field>(stack, ref)`            | repoint the offset (nullable → `Option<BStackRef>`, `None` writes `0`)        |
| `#[bstack_owned]`  | `replace_<field>(stack, new)`        | install `new`, **return the old** `BStackOwned<T>` (neither leaked nor freed) |
| `#[bstack_strong]` | `replace_<field>(&alloc, new)`       | install `new`, return the old `BStackRc<T>` (dropping it decrements)          |
| `#[bstack_ref]`    | *(also)* `replace_<field>(stack, r)` | ref is the only kind with **both** `set_` and `replace_`                      |

`replace_` is a persistent `mem::replace`: it hands the old value back to reuse or free.

Because it **consumes** the new value, `replace_` returns `Result<Old,
ReplaceError<New>>`: an I/O failure hands the consumed value back in
`ReplaceError.value`. The *old* value is never at risk. The lone `value: None`
case is a strong field whose old handle fails to reconstruct *after* commit;
the old block is then reclaimable only via crash-recovery.

`#[bstack_weak]` fields already have their own
[`set_<field>`](#reference-counted-blocks) wiring, making `#[bstack_mut]` a no-op.
`#[bstack_mut]` on an `#[embed]` field is a compile error.

### Containers

`#[bstack_mut]` also works on POD tuples and block-reference arrays; a `Vec` is
already mutable and needs no annotation:

| Field shape                          | Mutator(s)                                                                 |
|--------------------------------------|----------------------------------------------------------------------------|
| POD tuple `(A, B, …)`                | `set_<field>(stack, tuple)` — one atomic overwrite (owns no children)       |
| `#[bstack_owned/strong] [T; N]`      | `replace_<field>_at(&alloc, i, new) -> old` and `replace_<field>(&alloc, arr) -> old_arr` |
| `#[bstack_ref] [T; N]`               | *(also)* `set_<field>_at` / `set_<field>` (a ref owns nothing)              |
| `Vec<T>` (any kind)                  | *none needed* — mutate in place via the `get_<field>()` handle (`push_*`, …) |
| `#[bstack_owned/strong/weak] Foreign<T>` | `replace_<field>(&alloc, new) -> old` — moves the old cross-file target out as `ForeignOwned`/`ForeignRc`/`ForeignWeak` |
| `#[bstack_ref] Foreign<T>`           | *(also)* `set_<field>` / `replace_<field>` trafficking in plain `Foreign`   |

Arrays are fixed-size. The element mutators use a **row-major flat** `index`
(for `[[T; M]; N]`, slot `i*M + j` is `grid[i][j]`). Both `replace_` forms are one
crash-atomic `set` and uphold the same `ReplaceError` contract. A `Vec` persists its
descriptor back to the field on every mutation, so mutating the accessor handle *is*
mutating the field; `#[bstack_mut]` on a `Vec` is an accepted no-op.

A **scalar `Foreign<T>` / `Option<Foreign<T>>`** field is mutable, and its swap is
*purely local* and atomic: `replace_` hands the old target back as its RAII dual;
it needs no registry access and works even if the target file is detached. Owning
kinds get **only** `replace_`; a foreign `ref` also gets `set_`. `Foreign` inside
a container or tuple has no mutator yet.

### Enums

An enum's payload has no stable "field" to set, so `#[bstack_mut]` on the **enum
itself** generates a *whole-value* mutator that overwrites the discriminant +
payload together:

```rust
#[bstack_enum]
#[bstack_mut]
enum State { Idle, Active(u32), #[bstack_owned] Holding(Child) }

// owns nothing → `set` (wholesale overwrite):
// e.handle().set(&alloc, StateData::Active(7))?;
// owns children → `replace` (moves the old value out, ReplaceError contract):
let old = e.handle().replace(&alloc, StateData::Idle)?; // old: StateData::Holding(BStackOwned<Child>)
```

`set` is generated when no variant owns anything (pure POD / `ref` / foreign-`ref`);
`replace` when some variant owns children (owned / strong / weak / foreign),
returning the old value as `EData`. `#[bstack_mut]` on a **variant** is an error (it
goes on the enum), as is `#[bstack_mut]` on a shared (`rc` / `rc, weak`) enum or one
with an `#[embed]` variant.

There is also a raw escape hatch on **every** scalar field:
`unsafe fn raw_<field>_slice(stack) -> BStackSlice`, which is a view over the field's
inline storage (`.read()` / `.write()`). Writing bypasses the typed invariants and is `unsafe`.

## Moving out: `bstack_move!`

`bstack_move!` destructures a handle, transferring each field/variant out and
freeing only the parent *shell*.

On a **`BStackOwned<X>`** it is infallible. Because a bare owned handle carries
no allocator, pass one — `bstack_move!(owned, &alloc)`:

```rust
let pair: BStackOwned<Pair> = /* … */;
let (left, shared, right) = bstack_move!(pair, &alloc)?;
//   ^BStackOwned<Leaf>  ^BStackRc<Thing>  ^u32
```

On a **`BStackRc<X>`** it is a `try_unwrap`: success only when this is the **sole
strong owner**, else it hands the handle back. A weak observer doesn't block it. An
allocator-carrying handle (a `BStackRc`, or `owned.auto(&alloc)`) takes the
single-argument form:

```rust
match bstack_move!(rc)? {
    Ok((left, shared, right)) => { /* we were the only owner */ }
    Err(rc)                    => { /* someone else still holds it */ }
}
```

An [enum](#enums-bstack_enum) moves out through its `EData` companion instead of
a tuple.

> **Moved-out values are unrooted.** A handle from `bstack_move!` — like one from
> `X::new` — is detached from any persistent structure. Its block still lives on
> disk, but is reachable *only* through your in-memory handle: drop it without
> re-attaching or freeing it and it becomes unreachable garbage.

## Cloning: `TryCloneIn` / `TryClone`

Duplicating a handle means one of two things depending on whether the block is
uniquely owned or shared.

### Deep-clone an owned block: `TryCloneIn`

A plain `#[bstack_block]` / `#[bstack_enum]` implements `TryCloneIn`, a **deep**,
fallible clone into a fresh, independent `BStackOwned<Self>`:

```rust
use bstack_raii::TryCloneIn;

let copy: BStackOwned<Node> = node.try_clone_in(&alloc)?;
```

Each field is duplicated according to its ownership, which is the mirror of teardown:

| Field                 | On clone                                                                                                               |
|-----------------------|------------------------------------------------------------------------------------------------------------------------|
| POD / `#[bstack_ref]` | byte-copied (a ref clone **aliases** the same target)                                                                  |
| `#[bstack_owned]`     | the child is recursively deep-cloned into a fresh block                                                                |
| `#[embed]`            | the inline child is folded — its own children deep-cloned in place                                                     |
| `#[bstack_strong]`    | the shared child stays shared; its strong count is bumped                                                              |
| `#[bstack_weak]`      | stays weak to the same target; its weak count is bumped                                                                |
| `Vec<Thing>`          | per element, by the vector's annotation (POD data copied; owned elements deep-cloned; strong/weak bumped; ref aliased) |

So owned subtrees are copied into independent storage while shared children are
*re-referenced*: freeing the clone never disturbs the original's owned data.

> **Atomicity & crash-safety.** A clone allocates the whole new subtree up front,
> then commits every payload write *and* refcount bump as one crash-atomic batch
> (`BStack::inplace_gen`): a mid-clone allocation failure rolls back with nothing
> written, and a crash never leaves a torn copy. With a WAL anchor the fresh
> allocations are logged as made, so a crash *mid-clone* is reclaimed on the next
> open, completed by [`wal::finish`]. On a bulk-capable allocator
> (`BStackBulkAllocator`) the whole subtree is allocated in a single atomic `alloc_bulk`.

### Duplicate a shared handle: `TryClone`

A shared block is **not** deep-cloned. `BStackRc` / `BStackWeak` implement
`TryClone`, whose `try_clone` bumps the on-disk refcount and hands back another
handle to the *same* block, like `Rc::clone`:

```rust
use bstack_raii::TryClone;

let rc2 = rc.try_clone()?;      // another strong owner of the same block
let weak2 = weak.try_clone()?;  // another weak observer of the same block
```

An `(rc)` / `(rc, weak)` block therefore has no `try_clone_in`. A weak reference has
no coherent deep copy at all.

## Casting: `bstack_cast!`

Convert between typed handles and the untyped `bstack` primitives. Upcasts are
infallible; downcasts check the block's [tag](#type-tags-eightcc). Name the target
with `as`:

```rust
use bstack_raii::{BStackCastAs, BStackCastInto};   // the cast methods

let owned: BStackOwned<Node> = /* … */;
let slice = bstack_cast!(owned.auto(&alloc) as BStackOwnedSlice);   // owned upcast

match bstack_cast!(slice as BStackOwned<Node, _>) {                 // owned downcast
    Ok(node) => { /* tag matched */ }
    Err(e)   => { let slice = e.into_slice(); /* not a Node — slice handed back */ }
}

let view = node.handle().as_slice(stack);                           // borrowed upcast
let maybe: Option<Node> = bstack_cast!(view as Node)?;              // borrowed downcast
```

The equivalent methods (`into_slice`, `cast_into::<T>`, `cast_as::<T>`,
`as_slice`) can also be called directly. Casting works the same for enums.

## Cross-file pointers: `Foreign<T>`

Every reference covered so far points *within one file*. A `Foreign<T>` crosses
the boundary with a **wide pointer** naming both a target **file** and an offset inside
it, so an object graph can span many `bstack` files, each an independent crash-safe
unit.

### The file registry

Paths are awkward to store on disk, so a process-wide **registry** maps each file's
path ↔ a small, stable numeric [`FileId`]. A `Foreign` stores `(FileId, offset)`,
resolved to a live file through the registry. It is opt-in.

```rust
use bstack_raii::registry;

registry::init("registry.bstack")?;                  // once, at startup
let store_id = registry::attach("store.bstack", store_alloc)?;  // hand a file to the registry
```

`init` brings up the registry, which a small `bstack` file, allowing ids to survive a restart.
`attach` registers a file's path and installs its allocator as the **live host** a
`Foreign` into that file resolves through; it takes a
[`SyncBStackRaiiAllocator`](src/registry.rs) ([`BStackRaiiAllocator`] + `Send + Sync`),
which every bstack allocator qualifies. `FileId::SELF` (id `0`) is the current file, resolved
locally.

### Declaring a foreign field

A `Foreign<T>` field **must** carry an [ownership annotation](#field-ownership),
meaning the same thing as in-file. The target `T` must be a `#[bstack_block]` — an
un-annotated / POD / `#[embed]` `Foreign` is a compile error:

```rust
#[bstack_block]
struct Card {
    title: String,
    #[bstack_owned] body: Foreign<Document>,   // owns a Document in another file
}

// `Foreign::new` / `Foreign::at` are `unsafe` (a raw offset can't prove it names a
// valid `T`); the safe ways are `bstack_cast!(slice as Foreign<T>)` / `from_local`,
// or reading a field.
let ptr = unsafe { Foreign::<Document>::new(store_id, doc_off) };
let card = Card::new(&catalog, "report", ptr)?;

// Resolve across the boundary: `with` runs a closure against the target and its
// file's stack — `Ok(None)` if null, `Err` if that file isn't live.
let size = card.handle().get_body(catalog.stack())?
    .with(&catalog, |doc, fs| doc.get_size(fs).unwrap())?  // io::Result<Option<u64>>
    .expect("owned Foreign is never null");
```

`Foreign<'a, T>` is a 16-byte, zero-cost enum: an **explicit** pointer (a `FileId`
+ offset) or a [`SELF`] pointer. An accessor returns it with `'a` **bound to the
`&'a BStack` it read through**, so a `SELF` pointer can never be stored into or
outlive its file; an explicit pointer ignores the borrow and can be [`detach`]ed
to a `'static` `Foreign` (a `SELF` one cannot).

The annotation decides what teardown and clone do **in the target's own file**:

| Annotation         | Cross-file teardown                       | Cross-file clone                                  | `bstack_move!` yields    |
|--------------------|-------------------------------------------|---------------------------------------------------|--------------------------|
| `#[bstack_owned]`  | frees the target in its file              | deep-clones it into a fresh block in that file    | `ForeignOwned<T>`        |
| `#[bstack_strong]` | decrements its refcount there (free at 0) | bumps its refcount there (stays shared)           | `ForeignRc<T>`           |
| `#[bstack_weak]`   | decrements its weak count there           | bumps its weak count there                        | `ForeignWeak<T>`         |
| `#[bstack_ref]`    | nothing                                   | byte-copies the pointer (aliases the same target) | `Foreign<T>`             |

So tearing down a `Card` reclaims its `Document` in the store file, and cloning a
`Card` gives the copy its own independent `Document` there. (`#[bstack_owned]` needs
a deep-cloneable target, so `#[bstack_owned] Foreign<SharedBlock>` where the target
is itself `(rc)` is a compile error; use `#[bstack_strong]`.)

Moving an owning foreign field out with `bstack_move!` hands back its **RAII dual** —
`ForeignOwned` / `ForeignRc` / `ForeignWeak`, the cross-file analogues of
`BStackOwned` / `BStackRc` / `BStackWeak`. Each is non-`Copy` and carries
`.bstack_drop(&home)` (frees the target in its own file) and `.into_foreign()`
(→ a plain `Foreign` to re-store). A `#[bstack_ref]` field moves out as a plain
`Foreign`. Each also **resolves to its in-file handle** with `.into_local(..)` —
`ForeignOwned → BStackOwned<T>`, `ForeignRc::into_local(&target) → BStackRc<T>`,
`ForeignWeak::into_local(&target) → BStackWeak<T>`. Like `BStackOwned`, these don't
free on `Drop`. *(Currently scalar foreign fields only; an owning `Foreign` inside a
`Vec`/array/tuple/enum moves out as a bare `Foreign`.)*

> **Nullable & atomicity.** `Option<Foreign<T>>` is nullable on the offset-0 niche.
> Cross-file operations are *best-effort atomic*: the far side is committed before
> the home side, so a mid-op failure errs toward an over-provision (reclaimable) and
> never an under-count. If the target file is detached, teardown leaks (never
> corrupts) and a clone returns an error rather than aliasing an owner.

### Containers and shapes

A `Foreign<T>` composes everywhere an in-file reference does:

```rust
#[bstack_owned] parts: Vec<Foreign<Document>>,          // a growable list of pointers
#[bstack_owned] shards: [Foreign<Document>; 4],          // an inline fixed array
#[bstack_ref]   pair: (u32, Foreign<Document>),          // a foreign element in a tuple
```

Nesting and generics compose too. `Vec<Option<Foreign<T>>>`, nested arrays, and
generic targets all work, in struct fields and `#[bstack_enum]` variants alike.

A `Foreign` must target a plain block, not a pointer or container, so `Foreign<Vec<T>>`,
`Foreign<Foreign<T>>`, `Foreign<[T; N]>`, and `Foreign<(A, B)>` are rejected (bridge
through a named `#[bstack_block]`). A collection *of* pointers (`Vec<Foreign<T>>`) is
fine allowed.

[`bstack_cast!`](#casting-bstack_cast) bridges a `Foreign` and a local handle:
`slice as Foreign<T>` tags a slice with its file identity; `foreign as BStackRef<T>`
recovers a same-file reference for a local target. Both return `Option` (no I/O).

A full two-file walk-through is in [`examples/crossfile.rs`](examples/crossfile.rs):
`cargo run --example crossfile`.

## Type tags (`EightCC`)

Each block's header carries an 8-byte tag — the discriminant a
[downcast](#casting-bstack_cast) checks. It's computed at compile time: a short
**readable prefix** followed by a **hash tail**.

1. **Prefix** — derived from the type name: for a multi-word camel-case name, the
   uppercased word initials (`OrderLine` → `OL`); for a single word, its de-voweled
   uppercase (`Session` → `SSSN`, clamped). 2–4 bytes.
2. **Hash** — a 64-bit **FNV-1a** hash of `crate_name ++ "\0" ++ type_name`, with
   `module_path!()` folded into the tail (so same-named types in different modules
   stay distinct), little-endian. Every byte then has its **high bit set** (`| 0x80`),
   pushing it out of the printable range.
3. **Overlay** — the prefix bytes overwrite the low bytes of the hash; the high
   bytes are the hash tail.

A hex dump thus reads as a prefix followed by clearly-not-a-name bytes (every hash
byte ≥ `0x80`), e.g. `O L 8B C2 A9 F0 BD 91`. Being pure and deterministic, the tag
is stable across builds and versions. Override the prefix for a fixed tag (0–8 bytes;
fewer than 8 leaves room for the hash, exactly 8 is fully manual):

```rust
#[bstack_block(rc, tag = "ORDLINE")]        // explicit data tag
struct OrderLine { /* … */ }
```

`ctrl_tag = "…"` overrides the control-block tag (default: the data tag, distinct).
An override longer than 8 bytes is truncated with a compile warning;
`#[bstack_block(allow(overlong_tag))]` silences it.

This also works for `#[bstack_enum]` — e.g. `#[bstack_enum(rc, tag = "ENMTAG")] enum Mode { Unit, Val(u32) }`.

## Standard library collections

Built entirely on the primitives above — no privileged access. Every collection is
a plain [`BStackBlock`] (`BStackDrop` + `TryCloneIn`), so it can be used bare (a
top-level `BStackOwned<...>`) or composed as a `#[bstack_owned]` field, nested in
another collection, or held in a `#[bstack_enum]` variant.

| Type                              | Rust analogue                    | What it holds |
|------------------------------------|-----------------------------------|----------------|
| [`BStackCow<T>`]                  | `std::borrow::Cow`                | either a borrowed reference or an owned block, deep-copying on first write. |
| [`BStackBox<T>`]                  | `std::boxed::Box`                 | a single owned `Pod` value in its own block — the macro-free way to own a bare scalar/POD struct. |
| [`BStackLinkedList<T>`]           | `std::collections::LinkedList`    | an owned doubly-linked list of block values. Prefer `BStackDeque` / `BStackBlockVec` unless you need O(1) end/splice ops. |
| [`BStackDeque<T>`]                | `std::collections::VecDeque`      | an owned double-ended queue: a contiguous ring, O(1) amortized push/pop at both ends. |
| [`BStackHashMap<K, V>`]           | `std::collections::HashMap`       | an owned open-addressing map from a `Pod` key to a block value. |
| [`BStackBTreeMap<K, V>`]          | `std::collections::BTreeMap`      | an owned **ordered** map (copy-on-write B-tree) with sorted iteration. Keys are `Pod + Ord`. |
| [`BStackString`]                  | `std::string::String`             | a standalone owned, growable UTF-8 string block — the first-class way to own text (a deque element, a map value). |
| [`BStackCountingBloomFilter<K>`]  | (Bloom filter)                    | a probabilistic set: no false negatives, supports removal — a cheap fast-reject front for exact lookups. |
| [`BStackHashSet<K>`]              | `std::collections::HashSet`       | an owned open-addressing set of `Pod` keys, with an embedded Bloom-filter fast-reject front. |
| [`BStackBTreeSet<K>`]             | `std::collections::BTreeSet`      | an owned **ordered** set (copy-on-write B-tree), with an embedded Bloom-filter front. Keys are `Pod + Ord`. |
| [`BStackBinaryHeap<K, V>`]        | `std::collections::BinaryHeap`    | an owned priority queue (array-backed binary **min**-heap): `pop` returns the smallest-key entry. Keys are `Pod + Ord`. |

Each is constructed with `new` (or `with_capacity`), torn down with `bstack_drop`,
and deep-cloned with `try_clone_in`:

```rust
use bstack_raii::{BStackDrop, BStackHashMap, BStackString};

let map = BStackHashMap::<u32, BStackString>::new(&alloc)?;
map.insert(&alloc, 1, BStackString::new(&alloc, "one")?)?;
map.insert(&alloc, 2, BStackString::new(&alloc, "two")?)?;

let v = map.get(alloc.stack(), &1)?.unwrap();  // -> a BStackString handle
assert_eq!(v.to_string(alloc.stack())?, "one");

map.bstack_drop(&alloc)?;  // frees the map AND every owned BStackString value
```

Composing one into a block field: clone and teardown recurse through it
automatically:

```rust
#[bstack_block]
struct Session {
    id: u64,
    #[bstack_owned]
    log: BStackDeque<BStackString>,
}
```

Each collection's iterator (`HashMapIter`, `DequeIter`, `ListIter`, `BTreeMapIter`,
…) borrows the allocator's [`BStack`] and yields owned element handles.

## Examples

Runnable end-to-end programs live in [`examples/`](examples/):

| Example                                 | Run                             | Shows                                                                                                          |
|-----------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------|
| [`sessions.rs`](examples/sessions.rs)   | `cargo run --example sessions`  | shared `(rc, weak)` ownership, refcount-driven cleanup, durability across a reopen                             |
| [`expr.rs`](examples/expr.rs)           | `cargo run --example expr`      | a recursive `#[bstack_enum]` tree — evaluation, deep clone (`TryCloneIn`), `bstack_move!`                      |
| [`crossfile.rs`](examples/crossfile.rs) | `cargo run --example crossfile` | [`Foreign<T>`](#cross-file-pointers-foreignt) across two files — resolution, cross-file ownership, reclamation |

## Limitations

- **Fixed-size block payloads.** [Arrays](#fixed-size-arrays-t-n) `[T; N]` (nested
  to any depth) are stored *inline*; a *variable-length* sequence lives out-of-line
  via an inline descriptor (`Vec<T>` / `String`, `#[bstack_owned/strong/weak/ref]
  Vec<Thing>`, `Vec<[Thing; N]>`, and their `Option<…>` forms).
- **Requires a [`BStackRaiiAllocator`]** — a freeing allocator that reserves offset
  0; not `LinearBStackAllocator` (see [Concepts](#concepts)).
- **[Generic blocks](#generic-blocks)** work over type parameters (every field kind)
  and `const` array lengths; not over lifetime parameters, `rc` / `rc, weak` mode,
  or const parameters in a generic enum.
- **`Vec` / `Option` nesting** is capped at a single leaf / one `Option` layer (see
  [Field types](#field-types)); deeper nesting or a tuple element needs a named
  `#[bstack_block]` / `#[bstack_enum]`.
- **Enums** support unit / POD / all four annotated variant kinds (as scalars,
  arrays, and vectors) in all three modes, plus `bstack_move!` / `bstack_cast!`;
  struct and multi-field tuple variants aren't supported, nor an `#[embed]` variant.
- **[Cross-file pointers](#cross-file-pointers-foreignt)** (`Foreign<T>`) must target
  a plain block, never a pointer or container ("no double pointer"); their operations
  are *best-effort atomic* (a failure over-provisions rather than under-counts), and
  resolution requires the target file `attach`ed to the registry.
- **[Standard library collections](#standard-library-collections)** can't be shared
  (`#[bstack_strong]` / `#[bstack_weak]`) — they aren't `(rc)` blocks; `bstack_move!`
  only works on `BStackBox`.
- The on-disk **ABI is not yet stable**.

## Error codes

Every compile-time error from the `#[bstack_block]`, `#[bstack_enum]`,
`bstack_move!`, and `bstack_cast!` macros carries a stable `[BSTACKxxxx]` code, e.g.:

```text
error: [BSTACK0301] `Foreign` is a pointer and cannot be `#[embed]`ed
```

The message states the fix inline; **[ERRORS.md](ERRORS.md)** is the full reference,
with one entry per code (grouped by domain — attributes `00xx`, field shapes `01xx`,
enums `02xx`, `Foreign` `03xx`, generics `04xx`, `#[embed]` `05xx`, `#[bstack_mut]`
`06xx`, cast/move macros `07xx`).

## License

MIT (same as `bstack`).

[`bstack`]: https://github.com/williamwutq/bstack
[`std::io::Result`]: https://doc.rust-lang.org/std/io/type.Result.html
[`TryClone`]: src/clone.rs
[`BStackVec<T>`]: src/vec.rs
[`FileId`]: src/registry.rs
[`BStackRaiiAllocator`]: src/lib.rs
[`BStackBlock`]: src/block.rs
[`wal::finish`]: src/wal.rs
[`STD_WAL_ANCHOR`]: src/wal.rs
[`BStack`]: https://docs.rs/bstack
[`BStackCow<T>`]: src/stdlib/cow.rs
[`BStackBox<T>`]: src/stdlib/boxed.rs
[`BStackLinkedList<T>`]: src/stdlib/list.rs
[`BStackDeque<T>`]: src/stdlib/deque.rs
[`BStackHashMap<K, V>`]: src/stdlib/map.rs
[`BStackBTreeMap<K, V>`]: src/stdlib/tree.rs
[`BStackString`]: src/stdlib/string.rs
[`BStackCountingBloomFilter<K>`]: src/stdlib/bloom.rs
[`BStackHashSet<K>`]: src/stdlib/hashset.rs
[`BStackBTreeSet<K>`]: src/stdlib/btreeset.rs
[`BStackBinaryHeap<K, V>`]: src/stdlib/heap.rs
