// `Foreign::at` is `SELF`-only: it takes a single argument, a local handle, and
// resolves to a pointer within this file. Passing an explicit `FileId` —
// `Foreign::at(other_file, &block_in_this_file)` — must not compile, so safe code
// cannot mint a cross-file pointer whose offset does not name a valid `T` in the
// target file (which a later owning teardown would free in the *wrong* file). An
// explicit cross-file pointer goes through the registry-resolved `bstack_cast!` path
// or the `unsafe` `Foreign::new`.
use bstack_raii::registry::FileId;
use bstack_raii::{BStackBlock, Foreign};
fn forge<T: BStackBlock + 'static>(other_file: FileId, local: &T) {
    let _ = Foreign::at(other_file, local);
}
fn main() {}
