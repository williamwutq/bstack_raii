//! H1a — RTTI schema-record byte fuzzer (FUZZ.md's `rtti_decode`).
//!
//! `decode_type` / `Shape::decode` are pure functions over untrusted on-disk
//! bytes (a type's record body, as `RttiRegistry::load_type` would hand them
//! over after just reading the framing header) — no file I/O, so this is the
//! cheapest, purely coverage-guided harness in the suite.
//!
//! Oracle: never panic/abort/OOB; a malformed record is a clean `Err`; and an
//! accepted record round-trips through `encode_type` back to an identical
//! `RttiType`.

#![no_main]

use bstack_raii::EightCC;
use bstack_raii::rtti::{decode_type, encode_type};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut tag_bytes = [0u8; 8];
    tag_bytes.copy_from_slice(&data[..8]);
    let tag = EightCC(tag_bytes);
    let body = &data[8..];

    let ty = match decode_type(tag, body) {
        Ok(ty) => ty,
        Err(_) => return, // a clean rejection of a malformed record is fine
    };

    let reencoded = encode_type(&ty).expect("a successfully decoded type must always re-encode");
    let ty2 = decode_type(tag, &reencoded).expect("re-decoding our own encoding must succeed");
    assert_eq!(ty, ty2, "encode(decode(x)) is not stable");
});
