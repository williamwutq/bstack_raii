//! End-to-end test of the `#[bstack_class]` emitter: the macro must produce the
//! ordinary block machinery **and** register an RTTI descriptor that `rtti::sync`
//! persists, with the right shapes / offsets / sizes.
//!
//! This is its own integration-test binary, so its `linkme` slice collects only the
//! types declared here — isolated from the crate's own unit tests.
#![allow(dead_code)] // some fixtures are inspected only via RTTI, never instantiated

use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::registry;
use bstack_raii::rtti::{self, AnyRef, ForeignPtr, Moved, OwnershipKind, RttiBody, Shape, Value};
use bstack_raii::{
    BStackBlock, BStackBlockVec, BStackCast, BStackDrop, BStackOwned, Foreign, TryClone, WidePtr,
    bstack_class, rtti_path,
};

#[bstack_class]
struct Point {
    x: u32,
    y: u32,
}

#[bstack_class]
struct Line {
    #[bstack_owned]
    a: Point,
    #[bstack_owned]
    b: Point,
    labels: Vec<u8>,
    coords: [u64; 4],
    #[bstack_owned]
    next: Option<Point>,
}

#[bstack_class]
enum Kind2 {
    Empty,
    Pair(u32, u16),
    #[bstack_owned]
    Owns(Point),
}

#[bstack_class]
struct Wrap {
    #[bstack_owned]
    inner: Point,
    n: u32,
}

#[bstack_class]
struct VecArr {
    labels: Vec<u8>,
    coords: [u32; 3],
    #[bstack_owned]
    maybe: Option<Point>,
}

#[bstack_class]
struct Embedder {
    #[embed]
    e: Point,
    k: u32,
}

#[bstack_class(rc)]
struct RCell {
    v: u32,
}

#[bstack_class(rc, weak)]
struct WCell {
    v: u32,
}

#[bstack_class]
struct RcHolder {
    #[bstack_strong]
    s: RCell,
}

#[bstack_class]
struct WHolder {
    #[bstack_strong]
    s: WCell,
}

#[bstack_class]
struct WeakHolder {
    tag: u32,
    #[bstack_weak]
    w: WCell,
}

#[bstack_class]
struct WArr {
    #[bstack_weak]
    ws: [WCell; 2],
}

#[bstack_class]
struct Config {
    /// A constant class variable.
    #[bstack_static(7u32)]
    version: u32,
    /// A mutable class variable.
    #[bstack_mut]
    #[bstack_static(0u64)]
    counter: u64,
    /// An ordinary per-instance field.
    id: u32,
}

#[bstack_class]
struct FModel {
    #[bstack_owned]
    owned_f: Foreign<Point>,
    #[bstack_ref]
    ref_f: Foreign<Point>,
    n: u32,
}

#[bstack_class]
struct FShared {
    #[bstack_strong]
    s: Foreign<RCell>,
    #[bstack_weak]
    w: Foreign<WCell>,
}

#[bstack_class]
struct FStrong {
    #[bstack_strong]
    s: Foreign<RCell>,
    n: u32,
}

// Cross-file `Foreign` inside containers — the RTTI interpreter walks each element.
#[bstack_class]
struct FVec {
    #[bstack_owned]
    links: Vec<Foreign<Point>>,
    n: u32,
}

#[bstack_class]
struct FArr {
    #[bstack_owned]
    links: [Foreign<Point>; 2],
    n: u32,
}

#[bstack_class]
struct FTup {
    #[bstack_owned]
    pair: (Foreign<Point>, u32),
    n: u32,
}

// A pure-POD tuple field (no foreign) — describable by RTTI as a `Shape::Tuple`.
#[bstack_class]
struct PTup {
    pair: (u16, u8),
    n: u32,
}

// Complex enum-variant payloads (in-file): a POD vector, an owned vector, an owned
// fixed array.
#[bstack_class]
enum CEnum {
    Empty,
    Tags(Vec<u32>),
    #[bstack_owned]
    Kids(Vec<Point>),
    #[bstack_owned]
    Row([Point; 2]),
}

// Enum variants holding foreign containers.
#[bstack_class]
enum FEnum {
    Empty,
    #[bstack_owned]
    Many(Vec<Foreign<Point>>),
    #[bstack_owned]
    Duo([Foreign<Point>; 2]),
}

// An array of embedded children; a nested owned reference array.
#[bstack_class]
struct EmbArr {
    #[embed]
    kids: [Point; 2],
    n: u32,
}

#[bstack_class]
struct NestArr {
    #[bstack_owned]
    grid: [[Point; 2]; 2],
    n: u32,
}

// A nullable cross-file foreign — its Option niche is the offset word (byte 8).
#[bstack_class]
struct FOpt {
    #[bstack_owned]
    maybe: Option<Foreign<Point>>,
    n: u32,
}

// An owned vector of child blocks — its teardown reads a `u64` child offset per
// element, so a forged length would read neighbor blocks as offsets and free them.
#[bstack_class]
struct OVec {
    #[bstack_owned]
    kids: Vec<Point>,
    n: u32,
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "bstack_raii_rtti_class_{tag}_{}.stack",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn bstack_class_emits_block_machinery() {
    // The block half must still work exactly like `#[bstack_block]`.
    let path = temp_path("block");
    let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
    let p: BStackOwned<Point> = Point::new(&alloc, 7, 9).unwrap();
    assert_eq!(p.handle().get_x(alloc.stack()).unwrap(), 7);
    assert_eq!(p.handle().get_y(alloc.stack()).unwrap(), 9);
    p.bstack_drop(&alloc).unwrap();
    std::fs::remove_file(&path).ok();
}

#[test]
fn bstack_class_syncs_rtti_schema() {
    let path = temp_path("schema");
    let mut reg = rtti::sync(&path).unwrap();

    // Point: two POD fields, packed at offsets 0 and 4.
    let po = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();
    let pt = reg.load_type(po).unwrap();
    assert_eq!(pt.name, "Point");
    assert!(!pt.rc && !pt.weak);
    // Offsets and size are block-relative: `XOnDisk` leads with the 16-byte
    // `BlockHeader`, so the payload begins at 16 and the full block is 16 + 8.
    assert_eq!(pt.ondisk_size, 24);
    let RttiBody::Struct(pf) = &pt.body else {
        panic!("Point should be a struct body");
    };
    assert_eq!(pf.len(), 2);
    assert_eq!(pf[0].name, "x");
    assert_eq!(pf[0].offset, 16);
    assert_eq!(pf[0].shape, Shape::Pod { width: 4 });
    assert_eq!(pf[1].name, "y");
    assert_eq!(pf[1].offset, 20);
    assert_eq!(pf[1].shape, Shape::Pod { width: 4 });

    // Line: owned children, a vec, an array, and an optional pod.
    let lo = reg.ordinal_of(<Line as BStackCast>::eightcc()).unwrap();
    let lt = reg.load_type(lo).unwrap();
    let RttiBody::Struct(lf) = &lt.body else {
        panic!("Line should be a struct body");
    };
    assert_eq!(lf.len(), 5);
    let point_tag = <Point as BStackCast>::eightcc();
    assert_eq!(lf[0].name, "a");
    assert_eq!(lf[0].shape, Shape::Owned(point_tag));
    assert_eq!(lf[1].shape, Shape::Owned(point_tag));
    assert_eq!(lf[2].shape, Shape::Vec(Box::new(Shape::Pod { width: 1 })));
    assert_eq!(
        lf[3].shape,
        Shape::Array {
            n: 4,
            inner: Box::new(Shape::Pod { width: 8 }),
        }
    );
    assert_eq!(
        lf[4].shape,
        Shape::Option(Box::new(Shape::Owned(point_tag)))
    );

    // Idempotent: re-syncing the same binary's types appends nothing more.
    assert_eq!(reg.sync_compiled().unwrap(), 0);

    drop(reg);
    std::fs::remove_file(&path).ok();
}

#[test]
fn bstack_class_enum_syncs_rtti_schema() {
    let path = temp_path("enum");
    let reg = rtti::sync(&path).unwrap();

    let ord = reg.ordinal_of(<Kind2 as BStackCast>::eightcc()).unwrap();
    let ty = reg.load_type(ord).unwrap();
    assert_eq!(ty.name, "Kind2");
    let RttiBody::Enum(e) = &ty.body else {
        panic!("Kind2 should be an enum body");
    };
    // Discriminants 0/1/2 fit a u8.
    assert_eq!(e.disc_width, 1);
    assert!(e.payload_off > e.disc_off);
    assert_eq!(e.variants.len(), 3);

    // Unit variant.
    assert_eq!(e.variants[0].name, "Empty");
    assert_eq!(e.variants[0].disc_value, 0);
    assert!(e.variants[0].fields.is_empty());

    // POD-aggregate variant: `u32` at 0, `u16` at 4 (packed, payload-relative).
    assert_eq!(e.variants[1].name, "Pair");
    assert_eq!(e.variants[1].disc_value, 1);
    assert_eq!(e.variants[1].fields.len(), 2);
    assert_eq!(e.variants[1].fields[0].offset, 0);
    assert_eq!(e.variants[1].fields[0].shape, Shape::Pod { width: 4 });
    assert_eq!(e.variants[1].fields[1].offset, 4);
    assert_eq!(e.variants[1].fields[1].shape, Shape::Pod { width: 2 });

    // Owned-child variant: one `u64` offset at 0, shaped as Owned(Point).
    assert_eq!(e.variants[2].name, "Owns");
    assert_eq!(e.variants[2].disc_value, 2);
    assert_eq!(e.variants[2].fields.len(), 1);
    assert_eq!(e.variants[2].fields[0].offset, 0);
    assert_eq!(
        e.variants[2].fields[0].shape,
        Shape::Owned(<Point as BStackCast>::eightcc())
    );

    drop(reg);
    std::fs::remove_file(&path).ok();
}

#[test]
fn bstack_class_static_class_variables() {
    let path = temp_path("static");
    let reg = rtti::sync(&path).unwrap();

    let ord = reg.ordinal_of(<Config as BStackCast>::eightcc()).unwrap();
    let ty = reg.load_type(ord).unwrap();
    // Class variables are NOT per-instance: only `id` occupies `XOnDisk`.
    assert_eq!(ty.ondisk_size, 20); // 16-byte header + u32
    let RttiBody::Struct(f) = &ty.body else {
        panic!("Config should be a struct body");
    };
    assert_eq!(f.len(), 3);

    // Const class variable: value bytes are the encoded `7u32`.
    assert_eq!(f[0].name, "version");
    assert_eq!(
        f[0].shape,
        Shape::Class {
            mutable: false,
            inner: Box::new(Shape::Pod { width: 4 }),
            value: vec![7u8, 0, 0, 0].into(),
        }
    );

    // Mutable class variable: initial value `0u64`.
    assert_eq!(f[1].name, "counter");
    assert_eq!(
        f[1].shape,
        Shape::Class {
            mutable: true,
            inner: Box::new(Shape::Pod { width: 8 }),
            value: vec![0u8; 8].into(),
        }
    );

    // The one real instance field sits right after the header.
    assert_eq!(f[2].name, "id");
    assert_eq!(f[2].offset, 16);
    assert_eq!(f[2].shape, Shape::Pod { width: 4 });

    drop(reg);
    std::fs::remove_file(&path).ok();
}

/// A `Value::Pod` from raw bytes.
fn pod(bytes: &[u8]) -> Value {
    Value::Pod(bytes.into())
}

/// A fresh data-file allocator (separate from the schema file).
fn data_alloc(tag: &str) -> (FirstFitBStackAllocator, std::path::PathBuf) {
    let path = temp_path(tag);
    let alloc = FirstFitBStackAllocator::new(BStack::open(&path).unwrap()).unwrap();
    (alloc, path)
}

#[test]
fn interpret_reads_pod_struct() {
    let schema = temp_path("read1_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("read1_data");

    let p = Point::new(&alloc, 7, 9).unwrap();
    let off = BStackBlock::range(p.handle()).start();
    let ord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();

    let Value::Block { tag, fields } = reg.read_value(alloc.stack(), ord, off).unwrap() else {
        panic!("expected a block");
    };
    assert_eq!(tag, <Point as BStackCast>::eightcc());
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].0, "x");
    assert_eq!(fields[0].1, pod(&7u32.to_le_bytes()));
    assert_eq!(fields[1].0, "y");
    assert_eq!(fields[1].1, pod(&9u32.to_le_bytes()));

    p.bstack_drop(&alloc).unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_read_value_rejects_offset_overflow() {
    // Found by fuzzing (`fuzz/fuzz_targets/rtti_interpret.rs`): a caller-controlled
    // root offset near `u64::MAX`, combined with a nonzero field offset, must fail
    // cleanly via a checked add rather than panic with "attempt to add with overflow".
    let schema = temp_path("ovf_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("ovf_data");
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();

    let err = reg
        .read_value(alloc.stack(), ord, u64::MAX - 2)
        .unwrap_err();
    assert!(err.to_string().contains("BSTACK081A"), "got: {err}");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_reads_nested_and_via_pointer() {
    let schema = temp_path("read2_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("read2_data");

    let inner = Point::new(&alloc, 1, 2).unwrap();
    let w = Wrap::new(&alloc, inner, 5).unwrap();
    let off = BStackBlock::range(w.handle()).start();
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();

    // Direct read: the owned child is *followed* into a nested block.
    let v = reg.read_value(alloc.stack(), ord, off).unwrap();
    let Value::Block { fields, .. } = &v else {
        panic!("expected a block");
    };
    assert_eq!(fields[0].0, "inner");
    let Value::Block {
        fields: inner_fields,
        ..
    } = &fields[0].1
    else {
        panic!("owned child should be a followed block");
    };
    assert_eq!(inner_fields[0].1, pod(&1u32.to_le_bytes()));
    assert_eq!(inner_fields[1].1, pod(&2u32.to_le_bytes()));
    assert_eq!(fields[1].1, pod(&5u32.to_le_bytes()));

    // The same read through a typed pointer (ordinal recovered from the pointer).
    let ptr = rtti::typed_ptr(0, off, ord);
    assert_eq!(&reg.read_ptr(alloc.stack(), ptr).unwrap(), &v);

    w.bstack_drop(&alloc).unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_reads_enum_variant() {
    let schema = temp_path("read3_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("read3_data");

    let k = Kind2::new(&alloc, Kind2Data::Pair(3, 4)).unwrap();
    let off = BStackBlock::range(k.handle()).start();
    let ord = reg.ordinal_of(<Kind2 as BStackCast>::eightcc()).unwrap();

    let Value::Enum {
        variant, fields, ..
    } = reg.read_value(alloc.stack(), ord, off).unwrap()
    else {
        panic!("expected an enum");
    };
    assert_eq!(variant, "Pair");
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].1, pod(&3u32.to_le_bytes()));
    assert_eq!(fields[1].1, pod(&4u16.to_le_bytes()));

    k.bstack_drop(&alloc).unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_reads_vec_array_and_option() {
    let schema = temp_path("read4_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("read4_data");

    let v = VecArr::new(
        &alloc,
        &[10, 20],
        [1, 2, 3],
        Some(Point::new(&alloc, 8, 9).unwrap()),
    )
    .unwrap();
    let off = BStackBlock::range(v.handle()).start();
    let ord = reg.ordinal_of(<VecArr as BStackCast>::eightcc()).unwrap();

    let Value::Block { fields, .. } = reg.read_value(alloc.stack(), ord, off).unwrap() else {
        panic!("expected a block");
    };

    // Vec<u8> → a vector of 1-byte pod leaves.
    assert_eq!(fields[0].0, "labels");
    assert_eq!(fields[0].1, Value::Vec(vec![pod(&[10]), pod(&[20])].into()));

    // [u32; 3] → a fixed array of 4-byte pod leaves.
    assert_eq!(fields[1].0, "coords");
    assert_eq!(
        fields[1].1,
        Value::Array(
            vec![
                pod(&1u32.to_le_bytes()),
                pod(&2u32.to_le_bytes()),
                pod(&3u32.to_le_bytes()),
            ]
            .into()
        )
    );

    // Option<owned Point> present → Some(followed block).
    assert_eq!(fields[2].0, "maybe");
    let Value::Some(boxed) = &fields[2].1 else {
        panic!("expected Some");
    };
    let Value::Block {
        fields: pt_fields, ..
    } = boxed.as_ref()
    else {
        panic!("expected a followed Point block");
    };
    assert_eq!(pt_fields[0].1, pod(&8u32.to_le_bytes()));
    assert_eq!(pt_fields[1].1, pod(&9u32.to_le_bytes()));

    v.bstack_drop(&alloc).unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_teardown_reclaims_owned_tree() {
    let schema = temp_path("td1_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("td1_data");
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();

    // Build a Wrap owning a Point, return its (detached) root offset.
    let build = || {
        let inner = Point::new(&alloc, 1, 2).unwrap();
        let w = Wrap::new(&alloc, inner, 5).unwrap();
        BStackBlock::range(w.handle()).start()
    };

    // Leak oracle: warm once (allocator high-water settles), snapshot, then do an
    // identical build + RTTI teardown and assert the stack returned to baseline —
    // so teardown reclaimed the root *and* the owned child, with nothing leaked.
    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    let base = alloc.stack().len().unwrap();
    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    assert_eq!(alloc.stack().len().unwrap(), base, "RTTI teardown leaked");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_teardown_strong_rc() {
    let schema = temp_path("tds1_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("tds1_data");
    let ord = reg.ordinal_of(<RcHolder as BStackCast>::eightcc()).unwrap();

    // A holder that is the sole strong owner of an `rc` cell. Teardown must
    // decrement the inline refcount to zero and free the cell (+ the root).
    let build = || {
        let cell = RCell::new(&alloc, 5).unwrap();
        let h = RcHolder::new(&alloc, cell).unwrap();
        BStackBlock::range(h.handle()).start()
    };

    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    let base = alloc.stack().len().unwrap();
    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    assert_eq!(
        alloc.stack().len().unwrap(),
        base,
        "strong(rc) teardown leaked"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_teardown_strong_rc_weak() {
    let schema = temp_path("tds2_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("tds2_data");
    let ord = reg.ordinal_of(<WHolder as BStackCast>::eightcc()).unwrap();

    // Sole strong owner of an `(rc, weak)` cell: teardown decrements `ctrl.strong`
    // to zero (frees the data block), then the phantom weak to zero (frees control).
    let build = || {
        let cell = WCell::new(&alloc, 5).unwrap();
        let h = WHolder::new(&alloc, cell).unwrap();
        BStackBlock::range(h.handle()).start()
    };

    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    let base = alloc.stack().len().unwrap();
    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    assert_eq!(
        alloc.stack().len().unwrap(),
        base,
        "strong(rc,weak) teardown leaked"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_teardown_weak_field() {
    let schema = temp_path("tds3_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("tds3_data");
    let ord = reg
        .ordinal_of(<WeakHolder as BStackCast>::eightcc())
        .unwrap();

    // A holder with a *weak* reference to an externally-owned cell. Tearing the
    // holder down must release only the weak (never the still-strong-alive cell);
    // dropping the cell's strong owner afterward reclaims the cell + control. A full
    // cycle must return to baseline — no leak, no premature free.
    let cycle = || {
        let cell = WCell::new(&alloc, 5).unwrap(); // BStackRc: strong=1, weak=1 (phantom)
        let h = WeakHolder::new(&alloc, 7).unwrap();
        h.handle().set_w(&alloc, cell.downgrade().unwrap()).unwrap(); // weak 1 -> 2
        let off = BStackBlock::range(h.handle()).start();
        // Frees the holder root and decrements the cell's weak (2 -> 1); the cell
        // and its control survive because a strong owner is still live.
        unsafe { reg.teardown(&alloc, ord, off) }.unwrap();
        // `cell` (BStackRc) drops here: strong 1 -> 0 frees the data, weak 1 -> 0
        // frees the control.
    };

    cycle();
    let base = alloc.stack().len().unwrap();
    cycle();
    assert_eq!(
        alloc.stack().len().unwrap(),
        base,
        "weak-field teardown leaked"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_teardown_reclaims_vec_array_and_option() {
    let schema = temp_path("td2_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("td2_data");
    let ord = reg.ordinal_of(<VecArr as BStackCast>::eightcc()).unwrap();

    // A VecArr owns a data-block-backed Vec and an Option<owned Point>.
    let build = || {
        let v = VecArr::new(
            &alloc,
            &[10, 20, 30],
            [1, 2, 3],
            Some(Point::new(&alloc, 8, 9).unwrap()),
        )
        .unwrap();
        BStackBlock::range(v.handle()).start()
    };

    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    let base = alloc.stack().len().unwrap();
    unsafe { reg.teardown(&alloc, ord, build()) }.unwrap();
    assert_eq!(alloc.stack().len().unwrap(), base, "RTTI teardown leaked");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

fn read_u64(data: &BStack, off: u64) -> u64 {
    let mut b = [0u8; 8];
    data.get_into(off, &mut b).unwrap();
    u64::from_le_bytes(b)
}

fn field_offset(reg: &rtti::RttiRegistry, ord: u32, name: &str) -> u64 {
    match reg.load_type(ord).unwrap().body {
        RttiBody::Struct(f) => f.iter().find(|x| x.name == name).unwrap().offset as u64,
        _ => panic!("not a struct"),
    }
}

#[test]
fn interpret_set_pod_field() {
    let schema = temp_path("set_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("set_data");
    let ord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();

    let p = Point::new(&alloc, 7, 9).unwrap();
    let off = BStackBlock::range(p.handle()).start();

    // Overwrite `x`; `y` is untouched.
    unsafe { reg.set(alloc.stack(), ord, off, &["x"], &42u32.to_le_bytes()) }.unwrap();
    assert_eq!(p.handle().get_x(alloc.stack()).unwrap(), 42);
    assert_eq!(p.handle().get_y(alloc.stack()).unwrap(), 9);

    // Wrong width and unknown field are rejected.
    assert!(unsafe { reg.set(alloc.stack(), ord, off, &["x"], &[1, 2]) }.is_err());
    assert!(unsafe { reg.set(alloc.stack(), ord, off, &["z"], &0u32.to_le_bytes()) }.is_err());

    p.bstack_drop(&alloc).unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_clone_equal_and_independent() {
    let schema = temp_path("cl1_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("cl1_data");
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();

    let inner = Point::new(&alloc, 1, 2).unwrap();
    let w = Wrap::new(&alloc, inner, 5).unwrap();
    let src = BStackBlock::range(w.handle()).start();

    // Clone → a distinct, detached root whose Value tree equals the source's.
    let dst = unsafe { reg.clone_value(&alloc, ord, src) }.unwrap();
    assert_ne!(dst, src);
    let src_val = reg.read_value(alloc.stack(), ord, src).unwrap();
    assert_eq!(reg.read_value(alloc.stack(), ord, dst).unwrap(), src_val);

    // The owned child was deep-copied: mutating the source leaves the clone intact.
    unsafe { reg.set(alloc.stack(), ord, src, &["n"], &99u32.to_le_bytes()) }.unwrap();
    assert_eq!(reg.read_value(alloc.stack(), ord, dst).unwrap(), src_val);
    assert_ne!(reg.read_value(alloc.stack(), ord, src).unwrap(), src_val);

    // Both trees tear down independently (no shared blocks, no double free).
    unsafe { reg.teardown(&alloc, ord, src) }.unwrap();
    unsafe { reg.teardown(&alloc, ord, dst) }.unwrap();

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_clone_then_teardown_reclaims() {
    let schema = temp_path("cl2_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("cl2_data");
    let ord = reg.ordinal_of(<VecArr as BStackCast>::eightcc()).unwrap();

    // A full build → clone → teardown-both cycle must return to baseline: the clone
    // reproduces every owned block (vec data, option child) and both are reclaimed.
    let cycle = || {
        let v = VecArr::new(
            &alloc,
            &[10, 20, 30],
            [1, 2, 3],
            Some(Point::new(&alloc, 8, 9).unwrap()),
        )
        .unwrap();
        let src = BStackBlock::range(v.handle()).start();
        let dst = unsafe { reg.clone_value(&alloc, ord, src) }.unwrap();
        unsafe { reg.teardown(&alloc, ord, src) }.unwrap();
        unsafe { reg.teardown(&alloc, ord, dst) }.unwrap();
    };

    cycle();
    let base = alloc.stack().len().unwrap();
    cycle();
    assert_eq!(alloc.stack().len().unwrap(), base, "clone/teardown leaked");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_clone_shares_strong() {
    let schema = temp_path("cl3_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("cl3_data");
    let ord = reg.ordinal_of(<RcHolder as BStackCast>::eightcc()).unwrap();
    let soff = field_offset(&reg, ord, "s");

    // The leak oracle also proves sharing: a *copy* of the rc cell would need two
    // frees, but a shared clone bumps the refcount to 2 and both holders tearing
    // down reclaim the single cell exactly once.
    let cycle = || {
        let cell = RCell::new(&alloc, 5).unwrap();
        let h = RcHolder::new(&alloc, cell).unwrap();
        let src = BStackBlock::range(h.handle()).start();
        let dst = unsafe { reg.clone_value(&alloc, ord, src) }.unwrap();
        // The strong field points at the *same* cell in both holders (shared).
        assert_eq!(
            read_u64(alloc.stack(), src + soff),
            read_u64(alloc.stack(), dst + soff),
            "strong clone must share the target, not copy it"
        );
        unsafe { reg.teardown(&alloc, ord, src) }.unwrap(); // refcount 2 -> 1 (cell alive)
        unsafe { reg.teardown(&alloc, ord, dst) }.unwrap(); // refcount 1 -> 0 (cell freed)
    };

    cycle();
    let base = alloc.stack().len().unwrap();
    cycle();
    assert_eq!(alloc.stack().len().unwrap(), base, "strong clone leaked");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn any_ref_downcast_and_generic_fallback() {
    let schema = temp_path("any_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("any_data");
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();

    let p = Point::new(&alloc, 7, 9).unwrap();
    let off = BStackBlock::range(p.handle()).start();

    // A typed pointer resolves to an AnyRef with a registry-authoritative tag.
    let any = reg.any_ref(rtti::typed_ptr(0, off, pord)).unwrap();
    assert_eq!(any.tag(), <Point as BStackCast>::eightcc());
    assert_eq!(any.offset(), off);

    // Downcast to the matching compiled-in type → a live typed handle.
    assert!(any.is::<Point>());
    let pt = any.downcast::<Point>().expect("downcast to Point");
    assert_eq!(pt.get_x(alloc.stack()).unwrap(), 7);
    assert_eq!(pt.get_y(alloc.stack()).unwrap(), 9);

    // A non-matching type → None; fall back to generic interpretation.
    assert!(!any.is::<Wrap>());
    assert!(any.downcast::<Wrap>().is_none());
    let Value::Block { tag, .. } = reg.read_any(alloc.stack(), &any).unwrap() else {
        panic!("expected a block");
    };
    assert_eq!(tag, <Point as BStackCast>::eightcc());

    // The no-registry path recovers the same tag from the block header.
    assert_eq!(AnyRef::from_block(alloc.stack(), off).unwrap(), any);

    // An untyped pointer resolves to no AnyRef (never masquerades as a type).
    assert!(reg.any_ref(WidePtr::from_raw(0, 0, off)).is_none());

    p.bstack_drop(&alloc).unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_path_get_and_set_nested() {
    let schema = temp_path("path_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("path_data");
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();

    let w = Wrap::new(&alloc, Point::new(&alloc, 1, 2).unwrap(), 5).unwrap();
    let off = BStackBlock::range(w.handle()).start();

    // Navigate through the owned child with the path macro: `inner.x`, and read
    // `inner` as a whole block.
    assert_eq!(
        reg.get(alloc.stack(), ord, off, rtti_path!(inner.x))
            .unwrap(),
        pod(&1u32.to_le_bytes())
    );
    let Value::Block { tag, .. } = reg.get(alloc.stack(), ord, off, rtti_path!(inner)).unwrap()
    else {
        panic!("inner should be a block");
    };
    assert_eq!(tag, <Point as BStackCast>::eightcc());

    // Set a nested POD field, then a top-level one.
    unsafe {
        reg.set(
            alloc.stack(),
            ord,
            off,
            rtti_path!(inner.x),
            &42u32.to_le_bytes(),
        )
    }
    .unwrap();
    assert_eq!(
        reg.get(alloc.stack(), ord, off, rtti_path!(inner.x))
            .unwrap(),
        pod(&42u32.to_le_bytes())
    );
    unsafe { reg.set(alloc.stack(), ord, off, rtti_path!(n), &7u32.to_le_bytes()) }.unwrap();
    assert_eq!(
        reg.get(alloc.stack(), ord, off, rtti_path!(n)).unwrap(),
        pod(&7u32.to_le_bytes())
    );

    // Cannot descend through a POD leaf, and cannot `set` an owning reference.
    assert!(reg.get(alloc.stack(), ord, off, rtti_path!(n.x)).is_err());
    assert!(unsafe { reg.set(alloc.stack(), ord, off, rtti_path!(inner), &[0u8; 8]) }.is_err());

    unsafe { reg.teardown(&alloc, ord, off) }.unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_swap_reference() {
    let schema = temp_path("swap_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("swap_data");
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();
    let point_tag = <Point as BStackCast>::eightcc();

    let w = Wrap::new(&alloc, Point::new(&alloc, 1, 2).unwrap(), 5).unwrap();
    let off = BStackBlock::range(w.handle()).start();

    // A fresh Point to install; swapping hands back the old one.
    let np = Point::new(&alloc, 8, 9).unwrap();
    let np_off = BStackBlock::range(np.handle()).start();
    let old = reg
        .swap(alloc.stack(), ord, off, &["inner"], unsafe {
            AnyRef::new(point_tag, np_off)
        })
        .unwrap()
        .expect("field was non-null");
    assert_eq!(old.tag(), point_tag);

    // `inner` now reads as the installed Point.
    assert_eq!(
        reg.get(alloc.stack(), ord, off, &["inner", "x"]).unwrap(),
        pod(&8u32.to_le_bytes())
    );

    // Rejections: wrong eightcc, and a POD target.
    let wrap_tag = <Wrap as BStackCast>::eightcc();
    assert!(
        reg.swap(alloc.stack(), ord, off, &["inner"], unsafe {
            AnyRef::new(wrap_tag, off)
        })
        .is_err()
    );
    assert!(
        reg.swap(alloc.stack(), ord, off, &["n"], unsafe {
            AnyRef::new(point_tag, np_off)
        })
        .is_err()
    );

    // The old target and the wrap (now owning the new target) both reclaim cleanly.
    unsafe { reg.teardown(&alloc, pord, old.offset()) }.unwrap();
    unsafe { reg.teardown(&alloc, ord, off) }.unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn rtti_path_macro_expands() {
    let two: &[&str] = rtti_path!(inner.x);
    assert_eq!(two, &["inner", "x"]);
    let one: &[&str] = rtti_path!(n);
    assert_eq!(one, &["n"]);
    let deep: &[&str] = rtti_path!(a.b.c.d);
    assert_eq!(deep, &["a", "b", "c", "d"]);
}

#[test]
fn interpret_move_out_owned() {
    let schema = temp_path("mv1_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("mv1_data");
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();

    let w = Wrap::new(&alloc, Point::new(&alloc, 1, 2).unwrap(), 5).unwrap();
    let off = BStackBlock::range(w.handle()).start();

    // Disassemble: the Wrap shell is freed; `n` comes out by value, `inner` as an
    // AnyRef the caller now owns (the Point block survives).
    let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();
    assert_eq!(moved.len(), 2);
    let Moved::Pod(n) = &moved["n"] else {
        panic!("n should be POD");
    };
    assert_eq!(&n[..], &5u32.to_le_bytes());
    let Moved::Ref(Some(inner)) = &moved["inner"] else {
        panic!("inner should be a moved reference");
    };
    assert_eq!(inner.tag(), <Point as BStackCast>::eightcc());
    // The moved child is a live, downcastable Point.
    let pt = inner.downcast::<Point>().expect("downcast");
    assert_eq!(pt.get_x(alloc.stack()).unwrap(), 1);
    assert_eq!(pt.get_y(alloc.stack()).unwrap(), 2);
    // The caller owns it: tear it down.
    unsafe { reg.teardown(&alloc, pord, inner.offset()) }.unwrap();

    // Leak oracle: a full build → move_out (frees shell) → teardown-child cycle
    // returns to baseline (shell reclaimed, child reclaimed, nothing leaked).
    let cycle = || {
        let w = Wrap::new(&alloc, Point::new(&alloc, 1, 2).unwrap(), 5).unwrap();
        let off = BStackBlock::range(w.handle()).start();
        let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();
        let Moved::Ref(Some(inner)) = &moved["inner"] else {
            panic!()
        };
        unsafe { reg.teardown(&alloc, pord, inner.offset()) }.unwrap();
    };
    cycle();
    let base = alloc.stack().len().unwrap();
    cycle();
    assert_eq!(alloc.stack().len().unwrap(), base, "move_out leaked");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_move_out_embed_materializes() {
    let schema = temp_path("mv2_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("mv2_data");
    let ord = reg.ordinal_of(<Embedder as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();

    let em = Embedder::new(&alloc, Point::new(&alloc, 3, 4).unwrap(), 9).unwrap();
    let off = BStackBlock::range(em.handle()).start();
    let embed_inline = off + field_offset(&reg, ord, "e");

    let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();
    let Moved::Ref(Some(e)) = &moved["e"] else {
        panic!("embed should materialize into a reference");
    };
    // Materialized into a *fresh* block (not the original inline location).
    assert_ne!(e.offset(), embed_inline);
    let pt = e.downcast::<Point>().expect("downcast");
    assert_eq!(pt.get_x(alloc.stack()).unwrap(), 3);
    assert_eq!(pt.get_y(alloc.stack()).unwrap(), 4);
    unsafe { reg.teardown(&alloc, pord, e.offset()) }.unwrap();

    // Leak oracle over the whole cycle (materialized copy + shell both reclaimed).
    let cycle = || {
        let em = Embedder::new(&alloc, Point::new(&alloc, 3, 4).unwrap(), 9).unwrap();
        let off = BStackBlock::range(em.handle()).start();
        let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();
        let Moved::Ref(Some(e)) = &moved["e"] else {
            panic!()
        };
        unsafe { reg.teardown(&alloc, pord, e.offset()) }.unwrap();
    };
    cycle();
    let base = alloc.stack().len().unwrap();
    cycle();
    assert_eq!(alloc.stack().len().unwrap(), base, "embed move_out leaked");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_move_out_vec_array_option() {
    let schema = temp_path("mv3_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("mv3_data");
    let ord = reg.ordinal_of(<VecArr as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();

    let v = VecArr::new(
        &alloc,
        &[10, 20, 30],
        [1, 2, 3],
        Some(Point::new(&alloc, 8, 9).unwrap()),
    )
    .unwrap();
    let off = BStackBlock::range(v.handle()).start();

    let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();

    // The whole POD vector transfers as one VecRef (its data block untouched).
    let Moved::Vec(Some(vr)) = &moved["labels"] else {
        panic!("labels should be a whole vec");
    };
    assert_eq!(vr.elem, Shape::Pod { width: 1 });
    assert_eq!(read_u64(alloc.stack(), vr.data_off), 3); // len @0
    let mut bytes = [0u8; 3];
    alloc
        .stack()
        .get_into(vr.data_off + 16, &mut bytes)
        .unwrap();
    assert_eq!(bytes, [10, 20, 30]);

    // The inline POD array comes out by value.
    let Moved::Pod(coords) = &moved["coords"] else {
        panic!("coords should be POD");
    };
    assert_eq!(coords.len(), 12); // [u32; 3]

    // The optional owned child comes out as a reference.
    let Moved::Ref(Some(maybe)) = &moved["maybe"] else {
        panic!("maybe should be a moved reference");
    };
    assert_eq!(
        maybe
            .downcast::<Point>()
            .unwrap()
            .get_x(alloc.stack())
            .unwrap(),
        8
    );

    // Reclaim the transferred parts (the shell is already gone).
    unsafe { reg.teardown(&alloc, pord, maybe.offset()) }.unwrap();

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn class_variable_live_read_write() {
    let path = temp_path("classvar");
    let ctag = <Config as BStackCast>::eightcc();

    {
        let reg = rtti::sync(&path).unwrap();
        // Registration values: mutable `counter` = 0u64, const `version` = 7u32.
        assert_eq!(
            reg.class_value(ctag, "counter").unwrap(),
            0u64.to_le_bytes()
        );
        assert_eq!(
            reg.class_value(ctag, "version").unwrap(),
            7u32.to_le_bytes()
        );

        // Live in-place write to the mutable class variable.
        reg.set_class_value(ctag, "counter", &42u64.to_le_bytes())
            .unwrap();
        assert_eq!(
            reg.class_value(ctag, "counter").unwrap(),
            42u64.to_le_bytes()
        );

        // Rejections: a const var, a wrong-width value, an unknown name.
        assert!(
            reg.set_class_value(ctag, "version", &1u32.to_le_bytes())
                .is_err()
        );
        assert!(reg.set_class_value(ctag, "counter", &[1, 2, 3]).is_err());
        assert!(reg.class_value(ctag, "nope").is_err());
    }

    // The write persisted to the schema file: reopen and it is still 42.
    {
        let reg = rtti::sync(&path).unwrap();
        assert_eq!(
            reg.class_value(ctag, "counter").unwrap(),
            42u64.to_le_bytes()
        );
        assert_eq!(
            reg.class_value(ctag, "version").unwrap(),
            7u32.to_le_bytes()
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn class_variable_via_path() {
    let schema = temp_path("cvpath_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("cvpath_data");
    let ctag = <Config as BStackCast>::eightcc();
    let ord = reg.ordinal_of(ctag).unwrap();
    let data = alloc.stack(); // unused for a top-level class-var path, but the API takes it

    // get a class variable by path → its live value.
    let Value::Class(b) = reg.get(data, ord, 0, rtti_path!(counter)).unwrap() else {
        panic!("counter should read as a class value");
    };
    assert_eq!(&b[..], &0u64.to_le_bytes());

    // set a mutable class variable by path; it routes to the schema-side write.
    unsafe { reg.set(data, ord, 0, rtti_path!(counter), &99u64.to_le_bytes()) }.unwrap();
    let Value::Class(b) = reg.get(data, ord, 0, rtti_path!(counter)).unwrap() else {
        panic!()
    };
    assert_eq!(&b[..], &99u64.to_le_bytes());
    // the low-level accessor agrees (same slot).
    assert_eq!(
        reg.class_value(ctag, "counter").unwrap(),
        99u64.to_le_bytes()
    );

    // A const class variable rejects `set`; `swap` rejects any class variable.
    assert!(unsafe { reg.set(data, ord, 0, rtti_path!(version), &1u32.to_le_bytes()) }.is_err());
    assert!(
        reg.swap(data, ord, 0, rtti_path!(counter), unsafe {
            AnyRef::new(ctag, 0)
        })
        .is_err()
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn bstack_class_foreign_shapes() {
    let path = temp_path("fschema");
    let reg = rtti::sync(&path).unwrap();
    let point_tag = <Point as BStackCast>::eightcc();

    // Scalar Foreign fields carry the target tag + the ownership kind.
    let fm = reg
        .load_type(reg.ordinal_of(<FModel as BStackCast>::eightcc()).unwrap())
        .unwrap();
    let RttiBody::Struct(f) = &fm.body else {
        panic!("FModel struct");
    };
    assert_eq!(f[0].name, "owned_f");
    assert_eq!(
        f[0].shape,
        Shape::Foreign {
            tag: point_tag,
            kind: OwnershipKind::Owned
        }
    );
    assert_eq!(
        f[1].shape,
        Shape::Foreign {
            tag: point_tag,
            kind: OwnershipKind::Ref
        }
    );

    let fs = reg
        .load_type(reg.ordinal_of(<FShared as BStackCast>::eightcc()).unwrap())
        .unwrap();
    let RttiBody::Struct(g) = &fs.body else {
        panic!("FShared struct");
    };
    assert_eq!(
        g[0].shape,
        Shape::Foreign {
            tag: <RCell as BStackCast>::eightcc(),
            kind: OwnershipKind::Strong
        }
    );
    assert_eq!(
        g[1].shape,
        Shape::Foreign {
            tag: <WCell as BStackCast>::eightcc(),
            kind: OwnershipKind::Weak
        }
    );

    drop(reg);
    std::fs::remove_file(&path).ok();
}

#[test]
fn interpret_reads_foreign_pointer() {
    let schema = temp_path("fread_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("fread_data");
    let point_tag = <Point as BStackCast>::eightcc();
    let ord = reg.ordinal_of(<FModel as BStackCast>::eightcc()).unwrap();

    // Two local Points; the FModel points at them with SELF (file_id 0) foreigns.
    let p1 = Point::new(&alloc, 1, 2).unwrap();
    let p2 = Point::new(&alloc, 3, 4).unwrap();
    let p1_off = BStackBlock::range(p1.handle()).start();
    let p2_off = BStackBlock::range(p2.handle()).start();
    let fm = FModel::new(
        &alloc,
        unsafe { Foreign::at(p1.handle()) },
        unsafe { Foreign::at(p2.handle()) },
        5,
    )
    .unwrap();
    let off = BStackBlock::range(fm.handle()).start();

    // Read records the pointer (tag + kind + file_id + offset), not followed.
    let Value::Block { fields, .. } = reg.read_value(alloc.stack(), ord, off).unwrap() else {
        panic!("expected a block");
    };
    assert_eq!(
        fields[0].1,
        Value::Foreign {
            tag: point_tag,
            kind: OwnershipKind::Owned,
            file_id: 0,
            offset: p1_off,
        }
    );
    assert_eq!(
        fields[1].1,
        Value::Foreign {
            tag: point_tag,
            kind: OwnershipKind::Ref,
            file_id: 0,
            offset: p2_off,
        }
    );
    assert_eq!(fields[2].1, pod(&5u32.to_le_bytes()));

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_teardown_foreign_owned_across_files() {
    // Cross-file RTTI teardown: an FModel in the home file owns a Point in a foreign
    // file (via `#[bstack_owned] Foreign<Point>`). Tearing the FModel down through the
    // interpreter must free that Point in *its own* file, resolved through the global
    // registry — exactly as the generated `bstack_drop` does.
    let schema = temp_path("ftd_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FModel as BStackCast>::eightcc()).unwrap();

    let (home, hpath) = data_alloc("ftd_home");
    let (foreign, fpath) = data_alloc("ftd_foreign");

    // Allocate the owned target in the foreign file.
    // Warm the foreign file's WAL block once (a fixed one-time cost) so the length
    // oracle below is not confounded by the persistent block the now-WAL-backed
    // foreign teardown leaves behind.
    Point::new(&foreign, 0, 0)
        .unwrap()
        .bstack_drop(&foreign)
        .unwrap();
    let base = foreign.stack().len().unwrap();
    let leaf = Point::new(&foreign, 88, 99).unwrap();
    let leaf_off = BStackBlock::range(leaf.handle()).start();
    // The leaf reuses the slot the WAL warmup above freed, so the file need not
    // grow past `base`; it must not shrink.
    assert!(foreign.stack().len().unwrap() >= base);

    // Attach the foreign file to the process-wide registry (tolerant of a prior init).
    let reg_file = temp_path("ftd_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();
    assert!(!fid.is_self());

    // Build the home holder: a cross-file owned foreign + a harmless self ref foreign.
    let hp = Point::new(&home, 1, 2).unwrap();
    let fm = FModel::new(
        &home,
        unsafe { Foreign::<Point>::new(fid, leaf_off) },
        unsafe { Foreign::at(hp.handle()) },
        7,
    )
    .unwrap();
    let off = BStackBlock::range(fm.handle()).start();

    // Interpret the teardown; the foreign target is reclaimed in the foreign file.
    unsafe { reg.teardown(&home, ord, off) }.unwrap();
    let after = registry::with_host(fid, |host| host.stack().len().unwrap()).unwrap();
    assert!(
        after <= base,
        "foreign owned target not reclaimed by RTTI teardown: {after} > {base}"
    );

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_teardown_foreign_strong_across_files() {
    // Cross-file RC teardown: a `#[bstack_strong] Foreign<RCell>` decrements the
    // target's refcount *in its own file* and frees it at zero. The home holder is the
    // sole owner (count 1), so the interpreted teardown drives 1 -> 0.
    let schema = temp_path("fst_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FStrong as BStackCast>::eightcc()).unwrap();

    let (home, hpath) = data_alloc("fst_home");
    let (foreign, fpath) = data_alloc("fst_foreign");

    // An rc target in the foreign file at refcount 1, with no live handle to auto-drop
    // it — the strong foreign is its (sole, uncounted-here) owner.
    // Warm the foreign file's WAL block once (a fixed one-time cost) so the length
    // oracle below is not confounded by the persistent block the now-WAL-backed
    // foreign teardown leaves behind.
    Point::new(&foreign, 0, 0)
        .unwrap()
        .bstack_drop(&foreign)
        .unwrap();
    let base = foreign.stack().len().unwrap();
    let cell = RCell::new(&foreign, 88).unwrap();
    let cell_off = cell.handle().range().start();
    std::mem::forget(cell);
    // The leaf reuses the slot the WAL warmup above freed, so the file need not
    // grow past `base`; it must not shrink.
    assert!(foreign.stack().len().unwrap() >= base);

    let reg_file = temp_path("fst_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let h = FStrong::new(&home, unsafe { Foreign::<RCell>::new(fid, cell_off) }, 1).unwrap();
    let off = BStackBlock::range(h.handle()).start();

    unsafe { reg.teardown(&home, ord, off) }.unwrap();
    let after = registry::with_host(fid, |host| host.stack().len().unwrap()).unwrap();
    assert!(
        after <= base,
        "foreign strong target not freed at zero by RTTI teardown: {after} > {base}"
    );

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_clone_foreign_owned_across_files() {
    // Cross-file RTTI clone: cloning an FModel deep-copies its `#[bstack_owned]
    // Foreign<Point>` target into a FRESH block in the foreign file and repoints the
    // clone's pointer there; the `#[bstack_ref]` foreign is aliased (copied verbatim).
    let schema = temp_path("fcl_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ford = reg.ordinal_of(<FModel as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();
    let point_tag = <Point as BStackCast>::eightcc();

    let (home, hpath) = data_alloc("fcl_home");
    let (foreign, fpath) = data_alloc("fcl_foreign");

    let leaf = Point::new(&foreign, 88, 99).unwrap();
    let leaf_off = BStackBlock::range(leaf.handle()).start();

    let reg_file = temp_path("fcl_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let hp = Point::new(&home, 1, 2).unwrap();
    let fm = FModel::new(
        &home,
        unsafe { Foreign::<Point>::new(fid, leaf_off) },
        unsafe { Foreign::at(hp.handle()) },
        7,
    )
    .unwrap();
    let src = BStackBlock::range(fm.handle()).start();

    let dst = unsafe { reg.clone_value(&home, ford, src) }.unwrap();
    let Value::Block { fields, .. } = reg.read_value(home.stack(), ford, dst).unwrap() else {
        panic!("clone should be a block");
    };
    // owned foreign → deep-copied to a new offset in the same foreign file.
    let Value::Foreign {
        kind,
        file_id,
        offset: new_target,
        ..
    } = fields[0].1
    else {
        panic!("owned_f should be a foreign");
    };
    assert_eq!(kind, OwnershipKind::Owned);
    assert_ne!(file_id, 0);
    assert_ne!(
        new_target, leaf_off,
        "owned foreign must deep-copy to a new offset"
    );

    // The cloned target carries the same value, and lives in the foreign file.
    let cloned = registry::with_host(fid, |host| reg.read_value(host.stack(), pord, new_target))
        .unwrap()
        .unwrap();
    let Value::Block { tag, fields: pf } = cloned else {
        panic!("cloned target block");
    };
    assert_eq!(tag, point_tag);
    assert_eq!(pf[0].1, pod(&88u32.to_le_bytes()));
    assert_eq!(pf[1].1, pod(&99u32.to_le_bytes()));

    // ref foreign → aliased (kind Ref; not deep-copied).
    let Value::Foreign { kind: rk, .. } = fields[1].1 else {
        panic!("ref_f should be a foreign");
    };
    assert_eq!(rk, OwnershipKind::Ref);

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_move_out_foreign() {
    // move_out of a block with foreign fields hands each foreign pointer back whole
    // (tag + kind + file id + offset); the targets live in their own files and outlive
    // the freed shell.
    let schema = temp_path("fmv_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FModel as BStackCast>::eightcc()).unwrap();
    let point_tag = <Point as BStackCast>::eightcc();

    let (home, hpath) = data_alloc("fmv_home");
    let (foreign, fpath) = data_alloc("fmv_foreign");

    let leaf = Point::new(&foreign, 88, 99).unwrap();
    let leaf_off = BStackBlock::range(leaf.handle()).start();

    let reg_file = temp_path("fmv_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let hp = Point::new(&home, 1, 2).unwrap();
    let hp_off = BStackBlock::range(hp.handle()).start();
    let fm = FModel::new(
        &home,
        unsafe { Foreign::<Point>::new(fid, leaf_off) },
        unsafe { Foreign::at(hp.handle()) },
        7,
    )
    .unwrap();
    let off = BStackBlock::range(fm.handle()).start();

    let moved = unsafe { reg.move_out(&home, ord, off) }.unwrap();
    assert_eq!(
        moved["owned_f"],
        Moved::Foreign {
            tag: point_tag,
            kind: OwnershipKind::Owned,
            file_id: fid.get() as u64,
            offset: leaf_off,
        }
    );
    assert_eq!(
        moved["ref_f"],
        Moved::Foreign {
            tag: point_tag,
            kind: OwnershipKind::Ref,
            file_id: 0,
            offset: hp_off,
        }
    );
    let Moved::Pod(n) = &moved["n"] else {
        panic!("n should be POD");
    };
    assert_eq!(&n[..], &7u32.to_le_bytes());

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn bstack_class_foreign_container_shapes() {
    // The emitter lowers `Foreign` inside a `Vec` / array / tuple to the container
    // shape wrapping a `Foreign` leaf (all sharing the field's ownership kind).
    let path = temp_path("fcont_schema");
    let reg = rtti::sync(&path).unwrap();
    let pt = <Point as BStackCast>::eightcc();
    let f = Shape::Foreign {
        tag: pt,
        kind: OwnershipKind::Owned,
    };

    let fv = reg
        .load_type(reg.ordinal_of(<FVec as BStackCast>::eightcc()).unwrap())
        .unwrap();
    let RttiBody::Struct(g) = &fv.body else {
        panic!("FVec struct");
    };
    assert_eq!(g[0].shape, Shape::Vec(Box::new(f.clone())));

    let fa = reg
        .load_type(reg.ordinal_of(<FArr as BStackCast>::eightcc()).unwrap())
        .unwrap();
    let RttiBody::Struct(g) = &fa.body else {
        panic!("FArr struct");
    };
    assert_eq!(
        g[0].shape,
        Shape::Array {
            n: 2,
            inner: Box::new(f.clone()),
        }
    );

    let ft = reg
        .load_type(reg.ordinal_of(<FTup as BStackCast>::eightcc()).unwrap())
        .unwrap();
    let RttiBody::Struct(g) = &ft.body else {
        panic!("FTup struct");
    };
    assert_eq!(
        g[0].shape,
        Shape::Tuple(vec![f.clone(), Shape::Pod { width: 4 }].into())
    );

    drop(reg);
    std::fs::remove_file(&path).ok();
}

#[test]
fn bstack_class_pod_tuple_field() {
    // A pure-POD tuple field is describable too — a `Shape::Tuple` of opaque POD
    // members (each its whole width), read as a `Value::Tuple` and moved out whole.
    let path = temp_path("ptup_schema");
    let reg = rtti::sync(&path).unwrap();
    let ord = reg.ordinal_of(<PTup as BStackCast>::eightcc()).unwrap();

    let ty = reg.load_type(ord).unwrap();
    let RttiBody::Struct(g) = &ty.body else {
        panic!("PTup struct");
    };
    assert_eq!(
        g[0].shape,
        Shape::Tuple(vec![Shape::Pod { width: 2 }, Shape::Pod { width: 1 }].into())
    );

    let (home, hpath) = data_alloc("ptup_home");
    let h = PTup::new(&home, (0x1234u16, 0x56u8), 7).unwrap();
    let off = BStackBlock::range(h.handle()).start();

    let Value::Block { fields, .. } = reg.read_value(home.stack(), ord, off).unwrap() else {
        panic!("block");
    };
    let Value::Tuple(items) = &fields[0].1 else {
        panic!("pair tuple");
    };
    assert_eq!(items[0], pod(&0x1234u16.to_le_bytes()));
    assert_eq!(items[1], pod(&0x56u8.to_le_bytes()));

    // A POD tuple moves out as one opaque blob (cumulative packed bytes).
    let moved = unsafe { reg.move_out(&home, ord, off) }.unwrap();
    let Moved::Pod(bytes) = &moved["pair"] else {
        panic!("pod tuple should move whole");
    };
    assert_eq!(&bytes[..], &[0x34, 0x12, 0x56]);

    drop(reg);
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&hpath).ok();
}

#[test]
fn interpret_foreign_vec_teardown_across_files() {
    // `#[bstack_owned] Vec<Foreign<Point>>`: interpreted teardown frees every element
    // target in the foreign file (each a 16-byte `WidePtr` in the vec data block).
    let schema = temp_path("fvect_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FVec as BStackCast>::eightcc()).unwrap();

    let (home, hpath) = data_alloc("fvect_home");
    let (foreign, fpath) = data_alloc("fvect_foreign");

    // Baseline BEFORE the targets, then build three owned Points in the foreign file.
    // Warm the foreign file's WAL block once (a fixed one-time cost) so the length
    // oracle below is not confounded by the persistent block the now-WAL-backed
    // foreign teardown leaves behind.
    Point::new(&foreign, 0, 0)
        .unwrap()
        .bstack_drop(&foreign)
        .unwrap();
    let base = foreign.stack().len().unwrap();
    let mut offs = Vec::new();
    for i in 0..3u32 {
        let p = Point::new(&foreign, 10 + i, 20 + i).unwrap();
        offs.push(BStackBlock::range(p.handle()).start());
    }
    // The leaf reuses the slot the WAL warmup above freed, so the file need not
    // grow past `base`; it must not shrink.
    assert!(foreign.stack().len().unwrap() >= base);

    let reg_file = temp_path("fvect_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let links: Vec<Foreign<Point>> = offs
        .iter()
        .map(|&o| unsafe { Foreign::<Point>::new(fid, o) })
        .collect();
    let h = FVec::new(&home, links, 7).unwrap();
    let off = BStackBlock::range(h.handle()).start();

    unsafe { reg.teardown(&home, ord, off) }.unwrap();
    let after = registry::with_host(fid, |host| host.stack().len().unwrap()).unwrap();
    assert!(
        after <= base,
        "foreign vec element targets not reclaimed: {after} > {base}"
    );

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_foreign_vec_clone_across_files() {
    // Cloning `Vec<Foreign<Point>>` deep-copies EVERY element's target into a fresh
    // block in the foreign file and repoints the clone's `WidePtr` there.
    let schema = temp_path("fvecc_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FVec as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();

    let (home, hpath) = data_alloc("fvecc_home");
    let (foreign, fpath) = data_alloc("fvecc_foreign");

    let mut offs = Vec::new();
    for i in 0..3u32 {
        let p = Point::new(&foreign, 10 + i, 20 + i).unwrap();
        offs.push(BStackBlock::range(p.handle()).start());
    }

    let reg_file = temp_path("fvecc_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let links: Vec<Foreign<Point>> = offs
        .iter()
        .map(|&o| unsafe { Foreign::<Point>::new(fid, o) })
        .collect();
    let h = FVec::new(&home, links, 7).unwrap();
    let src = BStackBlock::range(h.handle()).start();

    let dst = unsafe { reg.clone_value(&home, ord, src) }.unwrap();

    // Pull each element's foreign offset out of a read of the vec.
    let elem_offs = |root: u64| -> Vec<u64> {
        let Value::Block { fields, .. } = reg.read_value(home.stack(), ord, root).unwrap() else {
            panic!("block");
        };
        let Value::Vec(items) = &fields[0].1 else {
            panic!("links vec");
        };
        items
            .iter()
            .map(|v| match v {
                Value::Foreign { offset, .. } => *offset,
                _ => panic!("foreign element"),
            })
            .collect()
    };
    let orig = elem_offs(src);
    let cloned = elem_offs(dst);
    assert_eq!(cloned.len(), 3);
    for (o, c) in orig.iter().zip(&cloned) {
        assert_ne!(
            *o, *c,
            "each foreign element must deep-copy to a new offset"
        );
    }
    // Each cloned target holds the same Point value in the foreign file.
    for (i, &c) in cloned.iter().enumerate() {
        let v = registry::with_host(fid, |host| reg.read_value(host.stack(), pord, c))
            .unwrap()
            .unwrap();
        let Value::Block { fields, .. } = v else {
            panic!("point block");
        };
        assert_eq!(fields[0].1, pod(&(10u32 + i as u32).to_le_bytes()));
    }

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_foreign_array_across_files() {
    // `#[bstack_owned] [Foreign<Point>; 2]`: read yields an array of foreign pointers;
    // move_out hands them back as a `Moved::ForeignList` (the shell is freed, targets
    // survive in the foreign file).
    let schema = temp_path("farr_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FArr as BStackCast>::eightcc()).unwrap();
    let pt = <Point as BStackCast>::eightcc();

    let (home, hpath) = data_alloc("farr_home");
    let (foreign, fpath) = data_alloc("farr_foreign");

    let p0 = Point::new(&foreign, 1, 2).unwrap();
    let o0 = BStackBlock::range(p0.handle()).start();
    let p1 = Point::new(&foreign, 3, 4).unwrap();
    let o1 = BStackBlock::range(p1.handle()).start();

    let reg_file = temp_path("farr_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let h = FArr::new(
        &home,
        [unsafe { Foreign::<Point>::new(fid, o0) }, unsafe {
            Foreign::<Point>::new(fid, o1)
        }],
        9,
    )
    .unwrap();
    let off = BStackBlock::range(h.handle()).start();

    // Read: a 2-element array of foreign pointers.
    let Value::Block { fields, .. } = reg.read_value(home.stack(), ord, off).unwrap() else {
        panic!("block");
    };
    let Value::Array(items) = &fields[0].1 else {
        panic!("links array");
    };
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0], Value::Foreign { offset, .. } if offset == o0));

    // move_out: the array becomes a ForeignList; targets outlive the freed shell.
    let moved = unsafe { reg.move_out(&home, ord, off) }.unwrap();
    assert_eq!(
        moved["links"],
        Moved::ForeignList(
            vec![
                ForeignPtr {
                    tag: pt,
                    kind: OwnershipKind::Owned,
                    file_id: fid.get() as u64,
                    offset: o0,
                },
                ForeignPtr {
                    tag: pt,
                    kind: OwnershipKind::Owned,
                    file_id: fid.get() as u64,
                    offset: o1,
                },
            ]
            .into()
        )
    );

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_foreign_tuple_across_files() {
    // `#[bstack_owned] (Foreign<Point>, u32)`: read yields a tuple whose members are a
    // foreign pointer and a POD; move_out hands each member back as its own `Moved`;
    // teardown frees the foreign target in its file.
    let schema = temp_path("ftup_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FTup as BStackCast>::eightcc()).unwrap();
    let pt = <Point as BStackCast>::eightcc();

    let (home, hpath) = data_alloc("ftup_home");
    let (foreign, fpath) = data_alloc("ftup_foreign");

    // Warm the foreign file's WAL block once (a fixed one-time cost) so the length
    // oracle below is not confounded by the persistent block the now-WAL-backed
    // foreign teardown leaves behind.
    Point::new(&foreign, 0, 0)
        .unwrap()
        .bstack_drop(&foreign)
        .unwrap();
    let base = foreign.stack().len().unwrap();
    let p = Point::new(&foreign, 5, 6).unwrap();
    let po = BStackBlock::range(p.handle()).start();

    let reg_file = temp_path("ftup_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    // Read + move_out on one holder.
    let h = FTup::new(&home, (unsafe { Foreign::<Point>::new(fid, po) }, 42u32), 9).unwrap();
    let off = BStackBlock::range(h.handle()).start();
    let Value::Block { fields, .. } = reg.read_value(home.stack(), ord, off).unwrap() else {
        panic!("block");
    };
    let Value::Tuple(items) = &fields[0].1 else {
        panic!("pair tuple");
    };
    assert!(matches!(items[0], Value::Foreign { offset, .. } if offset == po));
    assert_eq!(items[1], pod(&42u32.to_le_bytes()));

    let moved = unsafe { reg.move_out(&home, ord, off) }.unwrap();
    let Moved::Tuple(parts) = &moved["pair"] else {
        panic!("pair should move as a tuple, got {:?}", moved["pair"]);
    };
    assert_eq!(
        parts[0],
        Moved::Foreign {
            tag: pt,
            kind: OwnershipKind::Owned,
            file_id: fid.get() as u64,
            offset: po,
        }
    );
    assert_eq!(parts[1], Moved::Pod(Box::from(&42u32.to_le_bytes()[..])));

    // A second holder, torn down: its foreign target is reclaimed in the foreign file.
    let h2 = FTup::new(&home, (unsafe { Foreign::<Point>::new(fid, po) }, 1u32), 0).unwrap();
    let off2 = BStackBlock::range(h2.handle()).start();
    unsafe { reg.teardown(&home, ord, off2) }.unwrap();
    let after = registry::with_host(fid, |host| host.stack().len().unwrap()).unwrap();
    assert!(
        after <= base,
        "foreign tuple target not reclaimed by teardown: {after} > {base}"
    );

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn bstack_class_complex_enum_variant_shapes() {
    // Complex variant payloads lower to the right container shapes (in-file + foreign).
    let path = temp_path("cev_schema");
    let reg = rtti::sync(&path).unwrap();
    let pt = <Point as BStackCast>::eightcc();

    let ce = reg
        .load_type(reg.ordinal_of(<CEnum as BStackCast>::eightcc()).unwrap())
        .unwrap();
    let RttiBody::Enum(e) = &ce.body else {
        panic!("CEnum enum");
    };
    let cv = |n: &str| e.variants.iter().find(|v| v.name == n).unwrap();
    assert!(cv("Empty").fields.is_empty());
    assert_eq!(
        cv("Tags").fields[0].shape,
        Shape::Vec(Box::new(Shape::Pod { width: 4 }))
    );
    assert_eq!(
        cv("Kids").fields[0].shape,
        Shape::Vec(Box::new(Shape::Owned(pt)))
    );
    assert_eq!(
        cv("Row").fields[0].shape,
        Shape::Array {
            n: 2,
            inner: Box::new(Shape::Owned(pt)),
        }
    );

    let fe = reg
        .load_type(reg.ordinal_of(<FEnum as BStackCast>::eightcc()).unwrap())
        .unwrap();
    let RttiBody::Enum(e2) = &fe.body else {
        panic!("FEnum enum");
    };
    let fv = |n: &str| e2.variants.iter().find(|v| v.name == n).unwrap();
    let f = Shape::Foreign {
        tag: pt,
        kind: OwnershipKind::Owned,
    };
    assert_eq!(fv("Many").fields[0].shape, Shape::Vec(Box::new(f.clone())));
    assert_eq!(
        fv("Duo").fields[0].shape,
        Shape::Array {
            n: 2,
            inner: Box::new(f),
        }
    );

    drop(reg);
    std::fs::remove_file(&path).ok();
}

#[test]
fn interpret_enum_owned_vec_variant() {
    // An `#[bstack_owned] V(Vec<Point>)` variant: read yields the enum + its owned
    // vector of child blocks; teardown reclaims the whole variant (leak oracle).
    let schema = temp_path("eov_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<CEnum as BStackCast>::eightcc()).unwrap();
    let (alloc, dpath) = data_alloc("eov_data");

    let e = CEnum::new(
        &alloc,
        CEnumData::Kids(
            BStackBlockVec::from_handles(
                &alloc,
                vec![
                    Point::new(&alloc, 1, 2).unwrap(),
                    Point::new(&alloc, 3, 4).unwrap(),
                ],
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let off = BStackBlock::range(e.handle()).start();

    let Value::Enum {
        variant, fields, ..
    } = reg.read_value(alloc.stack(), ord, off).unwrap()
    else {
        panic!("enum");
    };
    assert_eq!(variant, "Kids");
    let Value::Vec(items) = &fields[0].1 else {
        panic!("owned vec payload");
    };
    assert_eq!(items.len(), 2);
    assert!(matches!(&items[0], Value::Block { .. }));
    unsafe { reg.teardown(&alloc, ord, off) }.unwrap();

    // Full build+teardown cycle returns to baseline.
    let cycle = || {
        let e = CEnum::new(
            &alloc,
            CEnumData::Kids(
                BStackBlockVec::from_handles(
                    &alloc,
                    vec![
                        Point::new(&alloc, 5, 6).unwrap(),
                        Point::new(&alloc, 7, 8).unwrap(),
                    ],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let o = BStackBlock::range(e.handle()).start();
        unsafe { reg.teardown(&alloc, ord, o) }.unwrap();
    };
    cycle();
    let base = alloc.stack().len().unwrap();
    cycle();
    assert_eq!(
        alloc.stack().len().unwrap(),
        base,
        "owned-vec enum variant teardown leaked"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_enum_foreign_vec_variant_across_files() {
    // An `#[bstack_owned] V(Vec<Foreign<Point>>)` variant: teardown frees every element
    // target in the foreign file.
    let schema = temp_path("efv_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FEnum as BStackCast>::eightcc()).unwrap();

    let (home, hpath) = data_alloc("efv_home");
    let (foreign, fpath) = data_alloc("efv_foreign");

    // Warm the foreign file's WAL block once (a fixed one-time cost) so the length
    // oracle below is not confounded by the persistent block the now-WAL-backed
    // foreign teardown leaves behind.
    Point::new(&foreign, 0, 0)
        .unwrap()
        .bstack_drop(&foreign)
        .unwrap();
    let base = foreign.stack().len().unwrap();
    let mut offs = Vec::new();
    for i in 0..3u32 {
        let p = Point::new(&foreign, 10 + i, 20 + i).unwrap();
        offs.push(BStackBlock::range(p.handle()).start());
    }

    let reg_file = temp_path("efv_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let links: Vec<Foreign<Point>> = offs
        .iter()
        .map(|&o| unsafe { Foreign::<Point>::new(fid, o) })
        .collect();
    let e = FEnum::new(&home, FEnumData::Many(links)).unwrap();
    let off = BStackBlock::range(e.handle()).start();

    unsafe { reg.teardown(&home, ord, off) }.unwrap();
    let after = registry::with_host(fid, |host| host.stack().len().unwrap()).unwrap();
    assert!(
        after <= base,
        "foreign-vec enum variant targets not reclaimed: {after} > {base}"
    );

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_move_out_embed_array() {
    // `#[embed] [Point; 2]`: move_out materializes each embedded child into a fresh
    // standalone block and hands them back as a `Moved::List`.
    let schema = temp_path("mea_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<EmbArr as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();
    let ptag = <Point as BStackCast>::eightcc();
    let (alloc, dpath) = data_alloc("mea_data");

    let e = EmbArr::new(
        &alloc,
        [
            Point::new(&alloc, 1, 2).unwrap(),
            Point::new(&alloc, 3, 4).unwrap(),
        ],
        9,
    )
    .unwrap();
    let off = BStackBlock::range(e.handle()).start();

    let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();
    let Moved::List(kids) = &moved["kids"] else {
        panic!("embed array should move as a List, got {:?}", moved["kids"]);
    };
    assert_eq!(kids.len(), 2);
    for (i, slot) in kids.iter().enumerate() {
        let a = slot.expect("materialized embed is never null");
        assert_eq!(a.tag(), ptag);
        let Value::Block { fields, .. } = reg.read_value(alloc.stack(), pord, a.offset()).unwrap()
        else {
            panic!("materialized point block");
        };
        assert_eq!(fields[0].1, pod(&((2 * i as u32) + 1).to_le_bytes()));
    }

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_move_out_nested_array() {
    // `#[bstack_owned] [[Point; 2]; 2]`: move_out hands back a `Moved::Array` of inner
    // `Moved::List`s (the child blocks survive the freed shell).
    let schema = temp_path("mna_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<NestArr as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();
    let (alloc, dpath) = data_alloc("mna_data");

    let e = NestArr::new(
        &alloc,
        [
            [
                Point::new(&alloc, 1, 0).unwrap(),
                Point::new(&alloc, 2, 0).unwrap(),
            ],
            [
                Point::new(&alloc, 3, 0).unwrap(),
                Point::new(&alloc, 4, 0).unwrap(),
            ],
        ],
        9,
    )
    .unwrap();
    let off = BStackBlock::range(e.handle()).start();

    let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();
    let Moved::Array(rows) = &moved["grid"] else {
        panic!(
            "nested array should move as an Array, got {:?}",
            moved["grid"]
        );
    };
    assert_eq!(rows.len(), 2);
    let mut expect = 1u32;
    for row in rows {
        let Moved::List(cells) = row else {
            panic!("inner row should move as a List");
        };
        assert_eq!(cells.len(), 2);
        for slot in cells {
            let a = slot.expect("owned element is non-null");
            let Value::Block { fields, .. } =
                reg.read_value(alloc.stack(), pord, a.offset()).unwrap()
            else {
                panic!("point block");
            };
            assert_eq!(fields[0].1, pod(&expect.to_le_bytes()));
            expect += 1;
        }
    }

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

// A `[#[bstack_weak] T; N]` array must come out of `move_out` as a `Moved::WeakList`
// (control-block offsets), NOT a `Moved::List` (data offsets) — else a caller could
// `swap` a control offset into a non-weak slot and type-confuse control bytes as `T`.
#[test]
fn interpret_move_out_weak_array_is_weaklist() {
    let schema = temp_path("mvwa_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("mvwa_data");
    let ord = reg.ordinal_of(<WArr as BStackCast>::eightcc()).unwrap();

    let c1 = WCell::new(&alloc, 1).unwrap();
    let c2 = WCell::new(&alloc, 2).unwrap();
    let wa = WArr::new(&alloc).unwrap();
    wa.handle()
        .set_ws(&alloc, 0, c1.downgrade().unwrap())
        .unwrap();
    wa.handle()
        .set_ws(&alloc, 1, c2.downgrade().unwrap())
        .unwrap();
    let off = wa.handle().range().start();

    let moved = unsafe { reg.move_out(&alloc, ord, off) }.unwrap();
    // `wa`'s handle owns nothing on drop (its shell is freed by move_out); no forget.
    let Moved::WeakList(list) = &moved["ws"] else {
        panic!(
            "weak array must come out as WeakList, got {:?}",
            moved["ws"]
        );
    };
    assert_eq!(list.len(), 2);
    assert!(list[0].is_some() && list[1].is_some());

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

// `move_out` on a `rc` / `(rc, weak)` root is a sole-owner try_unwrap: a shared root is
// refused ([BSTACK0819]) untouched; the sole owner is disassembled and (for rc,weak) its
// control block released.
#[test]
fn interpret_move_out_rc_try_unwrap() {
    let schema = temp_path("mvrc_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("mvrc_data");
    let rord = reg.ordinal_of(<RCell as BStackCast>::eightcc()).unwrap();

    // Shared: strong count 2 → refused, object untouched.
    let cell = RCell::new(&alloc, 5).unwrap();
    let cell2 = cell.try_clone().unwrap(); // strong = 2
    let off = cell.handle().range().start();
    let err = unsafe { reg.move_out(&alloc, rord, off) }.unwrap_err();
    assert!(err.to_string().contains("[BSTACK0819]"), "got: {err}");

    // Reduce to the sole owner, then move_out succeeds and hands back `v`.
    drop(cell2); // strong 2 -> 1
    let moved = unsafe { reg.move_out(&alloc, rord, off) }.unwrap();
    std::mem::forget(cell); // ownership consumed by move_out
    let Moved::Pod(v) = &moved["v"] else {
        panic!("v should be POD");
    };
    assert_eq!(u32::from_le_bytes(v[..4].try_into().unwrap()), 5);

    // `(rc, weak)` sole-owner move_out must free BOTH the data shell and the control
    // block — a leak would grow the stack across identical cycles.
    let word = reg.ordinal_of(<WCell as BStackCast>::eightcc()).unwrap();
    let cycle = |a: &_| {
        let c = WCell::new(a, 7).unwrap();
        let o = c.handle().range().start();
        let _ = unsafe { reg.move_out(a, word, o) }.unwrap();
        std::mem::forget(c);
    };
    cycle(&alloc); // warm
    let base = alloc.stack().len().unwrap();
    cycle(&alloc);
    assert_eq!(
        alloc.stack().len().unwrap(),
        base,
        "(rc,weak) move_out leaked the control block"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_swap_weak() {
    // `swap` on a `weak` path exchanges the control-block pointer (no refcount change),
    // returning the old control `AnyRef`.
    let schema = temp_path("swpw_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg
        .ordinal_of(<WeakHolder as BStackCast>::eightcc())
        .unwrap();
    let wtag = <WCell as BStackCast>::eightcc();
    let (alloc, dpath) = data_alloc("swpw_data");

    let cell1 = WCell::new(&alloc, 1).unwrap();
    let cell2 = WCell::new(&alloc, 2).unwrap();
    let h1 = WeakHolder::new(&alloc, 7).unwrap();
    h1.handle()
        .set_w(&alloc, cell1.downgrade().unwrap())
        .unwrap();
    let off1 = BStackBlock::range(h1.handle()).start();
    let h2 = WeakHolder::new(&alloc, 8).unwrap();
    h2.handle()
        .set_w(&alloc, cell2.downgrade().unwrap())
        .unwrap();
    let off2 = BStackBlock::range(h2.handle()).start();

    let ctrl = |off: u64| match reg.get(alloc.stack(), ord, off, rtti_path!(w)).unwrap() {
        Value::Ref { offset, .. } => offset,
        v => panic!("weak get: {v:?}"),
    };
    let c1 = ctrl(off1);
    let c2 = ctrl(off2);
    assert_ne!(c1, c2);

    let old = reg
        .swap(alloc.stack(), ord, off1, rtti_path!(w), unsafe {
            AnyRef::new(wtag, c2)
        })
        .unwrap();
    assert_eq!(old, Some(unsafe { AnyRef::new(wtag, c1) }));
    assert_eq!(ctrl(off1), c2, "weak now points at cell2's control");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_swap_foreign_across_files() {
    // `swap_foreign` on an owned `Foreign` path exchanges the 16-byte pointer, returning
    // the old cross-file target (which the caller now owns) — a purely local rewrite.
    let schema = temp_path("swpf_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FModel as BStackCast>::eightcc()).unwrap();
    let ptag = <Point as BStackCast>::eightcc();

    let (home, hpath) = data_alloc("swpf_home");
    let (foreign, fpath) = data_alloc("swpf_foreign");

    let p1 = Point::new(&foreign, 1, 2).unwrap();
    let o1 = BStackBlock::range(p1.handle()).start();
    let p2 = Point::new(&foreign, 3, 4).unwrap();
    let o2 = BStackBlock::range(p2.handle()).start();

    let reg_file = temp_path("swpf_reg");
    let _ = registry::init(&reg_file);
    let fid = registry::attach(&fpath, foreign).unwrap();

    let hp = Point::new(&home, 9, 9).unwrap();
    let fm = FModel::new(
        &home,
        unsafe { Foreign::<Point>::new(fid, o1) },
        unsafe { Foreign::at(hp.handle()) },
        7,
    )
    .unwrap();
    let off = BStackBlock::range(fm.handle()).start();

    let new = ForeignPtr {
        tag: ptag,
        kind: OwnershipKind::Owned,
        file_id: fid.get() as u64,
        offset: o2,
    };
    let old = reg
        .swap_foreign(home.stack(), ord, off, rtti_path!(owned_f), new)
        .unwrap();
    assert_eq!(
        old,
        Some(ForeignPtr {
            tag: ptag,
            kind: OwnershipKind::Owned,
            file_id: fid.get() as u64,
            offset: o1,
        })
    );

    // owned_f now points at o2.
    let Value::Foreign { offset, .. } = reg
        .get(home.stack(), ord, off, rtti_path!(owned_f))
        .unwrap()
    else {
        panic!("foreign");
    };
    assert_eq!(offset, o2);

    registry::detach(fid);
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&hpath).ok();
    std::fs::remove_file(&fpath).ok();
    std::fs::remove_file(&reg_file).ok();
}

#[test]
fn interpret_option_foreign_self_niche() {
    // Regression: an `Option<Foreign>` niche lives in the target-offset
    // word (byte 8), not the leading `file_id|type_index` word. A present SELF-file
    // foreign (file_id=0, type_index=0) has a zero leading word, so testing that word
    // would misread the live pointer as `None` → leaked / double-owned target.
    let schema = temp_path("fopt_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FOpt as BStackCast>::eightcc()).unwrap();
    let ptag = <Point as BStackCast>::eightcc();
    let (alloc, dpath) = data_alloc("fopt_data");

    // A present, SELF-file owned foreign target + its holder.
    let pt = Point::new(&alloc, 88, 99).unwrap();
    let pt_off = BStackBlock::range(pt.handle()).start();
    let h = FOpt::new(&alloc, Some(unsafe { Foreign::at(pt.handle()) }), 7).unwrap();
    let off = BStackBlock::range(h.handle()).start();

    // Read: the present `Some` is seen, not `Null`.
    let Value::Block { fields, .. } = reg.read_value(alloc.stack(), ord, off).unwrap() else {
        panic!("block");
    };
    let Value::Some(inner) = &fields[0].1 else {
        panic!(
            "present Option<Foreign> must read as Some, got {:?}",
            fields[0].1
        );
    };
    let Value::Foreign {
        file_id,
        offset,
        tag,
        ..
    } = inner.as_ref()
    else {
        panic!("foreign inner");
    };
    assert_eq!(*file_id, 0);
    assert_eq!(*offset, pt_off);
    assert_eq!(*tag, ptag);
    unsafe { reg.teardown(&alloc, ord, off) }.unwrap();

    // Teardown reclaims the present target — a full build+teardown cycle returns to
    // baseline (with the bug the target was skipped and leaked).
    let cycle = || {
        let pt = Point::new(&alloc, 1, 2).unwrap();
        let h = FOpt::new(&alloc, Some(unsafe { Foreign::at(pt.handle()) }), 0).unwrap();
        let o = BStackBlock::range(h.handle()).start();
        unsafe { reg.teardown(&alloc, ord, o) }.unwrap();
    };
    cycle();
    let base = alloc.stack().len().unwrap();
    cycle();
    assert_eq!(
        alloc.stack().len().unwrap(),
        base,
        "present Option<Foreign> target leaked on teardown"
    );

    // A `None` reads as `Null` and tears down cleanly.
    let hn = FOpt::new(&alloc, None, 5).unwrap();
    let offn = BStackBlock::range(hn.handle()).start();
    let Value::Block { fields, .. } = reg.read_value(alloc.stack(), ord, offn).unwrap() else {
        panic!("block");
    };
    assert_eq!(fields[0].1, Value::Null);
    unsafe { reg.teardown(&alloc, ord, offn) }.unwrap();

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_vec_len_bounded_to_block() {
    // Regression: a corrupt/oversized vec length word must yield a clean
    // [BSTACK0813] error — never an out-of-block read, petabyte allocation, or a
    // teardown that frees ranges over neighboring live blocks.
    let schema = temp_path("veclen_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<VecArr as BStackCast>::eightcc()).unwrap();
    let (alloc, dpath) = data_alloc("veclen_data");

    let h = VecArr::new(&alloc, &[10u8, 20, 30], [1, 2, 3], None).unwrap();
    let off = BStackBlock::range(h.handle()).start();

    // A clean read/teardown works before corruption (sanity).
    assert!(reg.read_value(alloc.stack(), ord, off).is_ok());

    // Locate the `labels` vec's data block via the schema's field offset, then forge
    // its length word to an absurd value.
    let ty = reg.load_type(ord).unwrap();
    let RttiBody::Struct(fields) = &ty.body else {
        panic!("struct");
    };
    let labels = fields.iter().find(|f| f.name == "labels").unwrap();
    let mut b = [0u8; 8];
    alloc
        .stack()
        .get_into(off + labels.offset as u64, &mut b)
        .unwrap(); // VecDesc.data_off
    let data_off = u64::from_le_bytes(b);
    assert_ne!(data_off, 0);
    alloc
        .stack()
        .set(data_off, (1u64 << 45).to_le_bytes())
        .unwrap(); // forge len word @0

    // Read and teardown both reject it cleanly (no abort, no neighbor free).
    let re = reg.read_value(alloc.stack(), ord, off).unwrap_err();
    assert!(re.to_string().contains("BSTACK0813"), "read: {re}");
    let te = unsafe { reg.teardown(&alloc, ord, off) }.unwrap_err();
    assert!(te.to_string().contains("BSTACK0813"), "teardown: {te}");

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn interpret_owned_vec_len_bounded() {
    // Regression (soundness): a forged length on an OWNED vector must not
    // let teardown read neighboring blocks as child offsets and free them.
    let schema = temp_path("ovec_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<OVec as BStackCast>::eightcc()).unwrap();
    let pord = reg.ordinal_of(<Point as BStackCast>::eightcc()).unwrap();
    let (alloc, dpath) = data_alloc("ovec_data");

    let h = OVec::new(
        &alloc,
        vec![
            Point::new(&alloc, 1, 2).unwrap(),
            Point::new(&alloc, 3, 4).unwrap(),
        ],
        7,
    )
    .unwrap();
    let off = BStackBlock::range(h.handle()).start();

    // A live victim block that must survive a corrupt teardown.
    let victim = Point::new(&alloc, 0xCAFE, 0xBABE).unwrap();
    let victim_off = BStackBlock::range(victim.handle()).start();

    // Forge the kids vec's length word.
    let ty = reg.load_type(ord).unwrap();
    let RttiBody::Struct(fields) = &ty.body else {
        panic!("struct");
    };
    let kids = fields.iter().find(|f| f.name == "kids").unwrap();
    let mut b = [0u8; 8];
    alloc
        .stack()
        .get_into(off + kids.offset as u64, &mut b)
        .unwrap();
    let data_off = u64::from_le_bytes(b);
    alloc
        .stack()
        .set(data_off, (1u64 << 40).to_le_bytes())
        .unwrap();

    // Teardown rejects the forged length *before* freeing anything; the victim (and,
    // incidentally, the whole structure) is untouched.
    let te = unsafe { reg.teardown(&alloc, ord, off) }.unwrap_err();
    assert!(te.to_string().contains("BSTACK0813"), "teardown: {te}");
    let Value::Block { fields, .. } = reg.read_value(alloc.stack(), pord, victim_off).unwrap()
    else {
        panic!("victim block");
    };
    assert_eq!(fields[0].1, pod(&0xCAFE_u32.to_le_bytes()));

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

#[test]
fn sync_rejects_layout_change() {
    // Regression: `sync` must reject a persisted descriptor whose layout
    // differs from the compiled type sharing its tag+name — not silently keep the stale
    // one and then misread every instance.
    let ptag = <Point as BStackCast>::eightcc();

    // The real compiled Point descriptor (from a throwaway sync).
    let real = {
        let t = temp_path("sync_ok_schema");
        let reg = rtti::sync(&t).unwrap();
        let d = reg.load_type(reg.ordinal_of(ptag).unwrap()).unwrap();
        drop(reg);
        std::fs::remove_file(&t).ok();
        d
    };
    let RttiBody::Struct(rf) = &real.body else {
        panic!("Point is a struct");
    };
    assert!(rf.len() >= 2, "Point needs >=2 fields for this test");

    // A fresh schema pre-seeded with a STALE Point descriptor: same tag + name, but one
    // field removed → a different layout.
    let schema = temp_path("sync_bad_schema");
    let mut reg = rtti::RttiRegistry::open(&schema).unwrap();
    let mut fake = real.clone();
    if let RttiBody::Struct(fields) = &mut fake.body {
        let mut v = fields.to_vec();
        v.pop();
        *fields = v.into();
    }
    fake.ondisk_size -= 4;
    assert_ne!(fake, real);
    reg.append(&fake).unwrap();

    // Syncing the real compiled schema over the stale descriptor is rejected.
    let e = reg.sync_compiled().unwrap_err();
    assert!(
        e.to_string().contains("BSTACK0814"),
        "expected 0814, got: {e}"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
}

#[test]
fn interpret_swap_rejects_fabricated_target() {
    // A safe RTTI mutator must not install a caller-fabricated offset into an
    // owning slot — it validates the target's on-disk header tag first.
    let schema = temp_path("swapval_schema");
    let reg = rtti::sync(&schema).unwrap();
    let (alloc, dpath) = data_alloc("swapval_data");
    let ord = reg.ordinal_of(<Wrap as BStackCast>::eightcc()).unwrap();
    let point_tag = <Point as BStackCast>::eightcc();

    let w = Wrap::new(&alloc, Point::new(&alloc, 1, 2).unwrap(), 5).unwrap();
    let off = BStackBlock::range(w.handle()).start();

    // (a) An out-of-bounds fabricated offset (correct tag) is rejected.
    let oob = unsafe { AnyRef::new(point_tag, 0xDEAD_BEEF) };
    let e = reg
        .swap(alloc.stack(), ord, off, &["inner"], oob)
        .unwrap_err();
    assert!(e.to_string().contains("BSTACK0815"), "oob: {e}");

    // (b) An in-bounds but WRONG-TYPE offset is rejected (point the `Point` slot at the
    // `Wrap` block itself — a real block, but not a `Point`).
    let wrong = unsafe { AnyRef::new(point_tag, off) };
    let e = reg
        .swap(alloc.stack(), ord, off, &["inner"], wrong)
        .unwrap_err();
    assert!(e.to_string().contains("BSTACK0815"), "wrong-type: {e}");

    // The field is untouched by the rejected swaps.
    assert_eq!(
        reg.get(alloc.stack(), ord, off, &["inner", "x"]).unwrap(),
        pod(&1u32.to_le_bytes())
    );

    // (c) A real Point still swaps in fine.
    let np = Point::new(&alloc, 8, 9).unwrap();
    let np_off = BStackBlock::range(np.handle()).start();
    let _old = reg
        .swap(alloc.stack(), ord, off, &["inner"], unsafe {
            AnyRef::new(point_tag, np_off)
        })
        .unwrap()
        .expect("field was non-null");
    assert_eq!(
        reg.get(alloc.stack(), ord, off, &["inner", "x"]).unwrap(),
        pod(&8u32.to_le_bytes())
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

// A self-referential owned foreign — used to build a teardown/clone cycle.
#[bstack_class]
struct FCycle {
    #[bstack_owned]
    me: Option<Foreign<FCycle>>,
    n: u32,
}

#[test]
fn interpret_foreign_cycle_recursion_bounded() {
    // A foreign cycle (here a SELF back-edge to the block itself) must error
    // cleanly (bounded native recursion), not stack-overflow / abort.
    let schema = temp_path("fcycle_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<FCycle as BStackCast>::eightcc()).unwrap();
    let ftag = <FCycle as BStackCast>::eightcc();
    let (alloc, dpath) = data_alloc("fcycle_data");

    let a = FCycle::new(&alloc, None, 7).unwrap();
    let a_off = BStackBlock::range(a.handle()).start();

    // Point `me` at the block itself → a self-owning foreign cycle. (`swap_foreign`
    // validates the target — the block really is an `FCycle` at `a_off`.)
    reg.swap_foreign(
        alloc.stack(),
        ord,
        a_off,
        rtti_path!(me),
        ForeignPtr {
            tag: ftag,
            kind: OwnershipKind::Owned,
            file_id: 0,
            offset: a_off,
        },
    )
    .unwrap();

    // Teardown recurses the cycle but returns a clean error instead of overflowing.
    let e = unsafe { reg.teardown(&alloc, ord, a_off) }.unwrap_err();
    assert!(e.to_string().contains("BSTACK0807"), "teardown: {e}");

    // Clone likewise (its owned-foreign deep-copy recurses the same edge). Since the
    // intention-first clone WAL, the clone holds the file's WAL lock for the whole
    // descent, so the SELF back-edge is rejected as a same-file re-entry (the clean
    // `WouldBlock` the static `ClonePlan` already gives) before the recursion budget is
    // reached — either way a clean error, never a stack overflow.
    let e = unsafe { reg.clone_value(&alloc, ord, a_off) }.unwrap_err();
    assert!(
        e.to_string().contains("BSTACK0807") || e.kind() == std::io::ErrorKind::WouldBlock,
        "clone: {e}"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

// A strong ref + an owned vec, for the deferred-release regression.
#[bstack_class]
struct Td12 {
    #[bstack_owned]
    v: Vec<Point>,
    #[bstack_strong]
    s: RCell,
}

#[test]
fn interpret_teardown_defers_shared_release_on_walk_error() {
    // A mid-walk error (here a corrupt vec length) must NOT have decremented
    // the shared `strong` refcount — releases are deferred to the commit phase, so a
    // retry re-does nothing (the bug decremented during the walk → retry double-frees).
    let schema = temp_path("td12_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<Td12 as BStackCast>::eightcc()).unwrap();
    let (alloc, dpath) = data_alloc("td12_data");

    // A shared rc cell; the field holds one strong ref, `cell` another → refcount 2.
    let cell = RCell::new(&alloc, 5).unwrap();
    let cell_off = cell.handle().range().start();
    let h = Td12::new(
        &alloc,
        vec![Point::new(&alloc, 1, 2).unwrap()],
        cell.try_clone().unwrap(),
    )
    .unwrap();
    let off = BStackBlock::range(h.handle()).start();

    // Read the cell's inline refcount (RC_REFCOUNT_OFFSET = 16).
    let refcount = |alloc: &FirstFitBStackAllocator| -> u64 {
        let mut b = [0u8; 8];
        alloc.stack().get_into(cell_off + 16, &mut b).unwrap();
        u64::from_le_bytes(b)
    };
    assert_eq!(refcount(&alloc), 2);

    // Corrupt the owned vec's length word → the teardown walk errors mid-way.
    let ty = reg.load_type(ord).unwrap();
    let RttiBody::Struct(fields) = &ty.body else {
        panic!("struct");
    };
    let vf = fields.iter().find(|f| f.name == "v").unwrap();
    let mut b = [0u8; 8];
    alloc
        .stack()
        .get_into(off + vf.offset as u64, &mut b)
        .unwrap();
    let data_off = u64::from_le_bytes(b);
    alloc
        .stack()
        .set(data_off, (1u64 << 45).to_le_bytes())
        .unwrap();

    // Teardown errors — and the strong refcount is untouched (release was deferred).
    let e = unsafe { reg.teardown(&alloc, ord, off) }.unwrap_err();
    assert!(e.to_string().contains("BSTACK0813"), "teardown: {e}");
    assert_eq!(
        refcount(&alloc),
        2,
        "shared strong refcount was decremented during a failed walk"
    );

    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}

// POD `Option` fields (bytemuck in-place niche), scalar + array.
#[bstack_class]
struct PodOpt {
    m: Option<std::num::NonZeroU32>,
    arr: [Option<std::num::NonZeroU32>; 2],
    n: u32,
}

#[test]
fn interpret_pod_option_niche() {
    // A POD `Option<T>` (T: PodInOption) uses bytemuck's IN-PLACE niche, not
    // the offset-0 `u64` niche `Shape::Option` denotes. It must lower to opaque POD
    // (scalar and per array element), else the interpreter reads a `u64` overlapping the
    // next field and misreads `None` as `Some`.
    use std::num::NonZeroU32;
    let schema = temp_path("podopt_schema");
    let reg = rtti::sync(&schema).unwrap();
    let ord = reg.ordinal_of(<PodOpt as BStackCast>::eightcc()).unwrap();

    // Schema: `m` is opaque POD, `arr` an Array of opaque POD — NOT `Shape::Option`.
    let ty = reg.load_type(ord).unwrap();
    let RttiBody::Struct(fields) = &ty.body else {
        panic!("struct");
    };
    let f = |n: &str| &fields.iter().find(|f| f.name == n).unwrap().shape;
    assert_eq!(*f("m"), Shape::Pod { width: 4 });
    assert_eq!(
        *f("arr"),
        Shape::Array {
            n: 2,
            inner: Box::new(Shape::Pod { width: 4 }),
        }
    );

    // A `None` scalar reads as the niche bytes (zero), and the neighbor `n` is intact.
    let (alloc, dpath) = data_alloc("podopt_data");
    let h = PodOpt::new(&alloc, None, [None, NonZeroU32::new(5)], 7).unwrap();
    let off = BStackBlock::range(h.handle()).start();
    let Value::Block { fields, .. } = reg.read_value(alloc.stack(), ord, off).unwrap() else {
        panic!("block");
    };
    assert_eq!(fields[0].1, pod(&0u32.to_le_bytes()), "None scalar");
    let Value::Array(items) = &fields[1].1 else {
        panic!("arr array");
    };
    assert_eq!(items[0], pod(&0u32.to_le_bytes()), "None element");
    assert_eq!(items[1], pod(&5u32.to_le_bytes()), "Some(5) element");
    assert_eq!(fields[2].1, pod(&7u32.to_le_bytes()), "neighbor n intact");

    h.bstack_drop(&alloc).unwrap();
    drop(reg);
    std::fs::remove_file(&schema).ok();
    std::fs::remove_file(&dpath).ok();
}
