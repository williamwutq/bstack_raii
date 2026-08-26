//! [`TryClone`] — the allocator-less, same-type fallible clone.

use std::io;

/// Duplicate `self`, performing any fallible I/O the duplication requires,
/// **without** needing an allocator.
///
/// This is how a **shared** block is cloned: [`crate::BStackRc`] and
/// [`crate::BStackWeak`] implement it by bumping an on-disk refcount and handing
/// back another handle to the *same* block — exactly like `Rc::clone` /
/// `shared_ptr` copy in Rust / C++. A shared block is deliberately **not**
/// deep-copied into an independent [`BStackOwned`](crate::BStackOwned) via
/// [`TryCloneIn`](crate::TryCloneIn); sharing, not copying, is its defining
/// semantics.
///
/// This matters most for the **weak** case: there is no coherent deep copy of a
/// weak reference. A weak reference observes a live object's *control block*; a
/// "deep copy" would have to either point at the same live object (in which case
/// it is just another weak handle — a count bump, which is what this does) or at
/// some fresh object (in which case it observes nothing the original observed, so
/// it is not a copy at all). So `BStackWeak::try_clone` bumps the weak count and
/// returns another weak handle to the same control block — the only sound
/// meaning a weak clone can have.
pub trait TryClone: Sized {
    fn try_clone(&self) -> io::Result<Self>;
}
