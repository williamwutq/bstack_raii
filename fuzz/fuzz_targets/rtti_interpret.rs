//! H1b — RTTI read-interpreter fuzzer (FUZZ.md's `rtti_interpret`).
//!
//! Seeds a fixed registry against a handful of real, valid `#[bstack_class]`
//! fixtures (covering `owned`/`strong`/`weak`/`vec`/`array`/`option`/`enum`
//! shapes), built once per process, then repeatedly calls
//! [`RttiRegistry::read_value`] with a **fuzzer-chosen root offset** — i.e. a
//! schema that is real but very likely paired with the wrong bytes, the
//! interpreter's actual untrusted-input surface (`RttiRegistry::load_type`
//! itself is exercised by `rtti_decode`).
//!
//! Only the non-mutating reader is fuzzed here: `teardown`/`clone_value` at a
//! forged offset can free or bump a refcount on an unrelated *live* fixture
//! (the "valid but wrong object" case FUZZ.md calls out as O2-blind), which
//! would poison every later iteration sharing this process's one data file.
//! Fuzzing those safely needs the isolated per-sequence state H2 builds
//! (`DebugCheckingAllocator` + the `Model`); out of scope for this harness.
//!
//! Oracle: `read_value` never panics/aborts, and returns in bounded
//! time/memory (`Ok` or a clean `Err`) — the interpreter's own budget
//! (`src/rtti.rs`'s `run_read`) is the thing under test.

#![no_main]

use std::sync::{Mutex, OnceLock};

use arbitrary::Arbitrary;
use bstack::{BStack, BStackAllocator, FirstFitBStackAllocator};
use bstack_raii::rtti::{self, RttiOrdinal, RttiRegistry};
use bstack_raii::{BStackCast, bstack_class};
use libfuzzer_sys::fuzz_target;

#[bstack_class]
struct Point {
    x: u32,
    y: u32,
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
enum Kind2 {
    Empty,
    Pair(u32, u16),
    #[bstack_owned]
    Owns(Point),
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
struct WeakHolder {
    tag: u32,
    #[bstack_weak]
    w: WCell,
}

struct Fixture {
    reg: RttiRegistry,
    alloc: FirstFitBStackAllocator,
    /// Registered ordinals actually exercised by a live fixture below — reading
    /// under one of these is the realistic "right schema, wrong offset" case.
    ordinals: Vec<RttiOrdinal>,
    file_len: u64,
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    let dir = std::env::var_os("BSTACK_RAII_FUZZ_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.join(format!(
        "bstack_raii_fuzz_{tag}_{}_{nanos}.bstack",
        std::process::id()
    ))
}

fn ord_of<T: BStackCast>(reg: &RttiRegistry) -> RttiOrdinal {
    reg.ordinal_of(<T as BStackCast>::eightcc()).unwrap()
}

fn build_fixture() -> Fixture {
    let schema_path = temp_path("rtti_interp_schema");
    let data_path = temp_path("rtti_interp_data");
    let reg = rtti::sync(&schema_path).unwrap();
    let alloc = FirstFitBStackAllocator::new(BStack::open(&data_path).unwrap()).unwrap();

    // One live instance of each shape-bearing fixture type. `BStackOwned<T>`
    // carries no `Drop` impl (bstack_raii's teardown is always explicit), so
    // simply letting these bindings go out of scope leaks them on disk — which
    // is exactly what we want: a stable, never-freed set of real roots to read
    // against for the whole process lifetime.
    let _p = Point::new(&alloc, 1, 2).unwrap();
    let _w = Wrap::new(&alloc, Point::new(&alloc, 3, 4).unwrap(), 5).unwrap();
    let _v = VecArr::new(
        &alloc,
        &[10, 20, 30],
        [1, 2, 3],
        Some(Point::new(&alloc, 8, 9).unwrap()),
    )
    .unwrap();
    let _k = Kind2::new(&alloc, Kind2Data::Pair(6, 7)).unwrap();
    let cell = RCell::new(&alloc, 11).unwrap();
    let _rh = RcHolder::new(&alloc, cell).unwrap();
    let wcell = WCell::new(&alloc, 22).unwrap();
    let wh = WeakHolder::new(&alloc, 1).unwrap();
    wh.handle().set_w(&alloc, wcell.downgrade().unwrap()).unwrap();
    std::mem::forget(wcell); // keep the strong owner alive so the weak stays valid
    std::mem::forget(wh);

    let ordinals = vec![
        ord_of::<Point>(&reg),
        ord_of::<Wrap>(&reg),
        ord_of::<VecArr>(&reg),
        ord_of::<Kind2>(&reg),
        ord_of::<RcHolder>(&reg),
        ord_of::<WeakHolder>(&reg),
    ];
    let file_len = alloc.len().unwrap();

    Fixture {
        reg,
        alloc,
        ordinals,
        file_len,
    }
}

fn fixture() -> &'static Mutex<Fixture> {
    static FIXTURE: OnceLock<Mutex<Fixture>> = OnceLock::new();
    FIXTURE.get_or_init(|| Mutex::new(build_fixture()))
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    fixture_idx: u8,
    raw_offset: u64,
    /// When set, fold `raw_offset` into the live file's range (the realistic
    /// "misread a live neighbor" case); otherwise pass it through unclamped
    /// (the "wildly out of range" case, which `bstack`'s own bounds checks
    /// should already reject cleanly).
    clamp: bool,
}

fuzz_target!(|input: FuzzInput| {
    let fx = fixture();
    let fx = fx.lock().unwrap();

    let ord = fx.ordinals[input.fixture_idx as usize % fx.ordinals.len()];
    let offset = if input.clamp && fx.file_len > 0 {
        input.raw_offset % fx.file_len
    } else {
        input.raw_offset
    };

    // Must never panic/abort/OOB regardless of what `offset` actually holds.
    let _ = fx.reg.read_value(fx.alloc.stack(), ord, offset);
});
