//! Low-level **on-disk I/O core**: atomic primitives over a `bstack` file that sit
//! below the object model. Currently the [`refcount`] counter operations.

pub(crate) mod refcount;
