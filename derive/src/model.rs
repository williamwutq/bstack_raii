//! The struct/enum lowering **IR**: the bundle of code pieces one field (or, later,
//! one enum variant) contributes to the generated block.
//!
//! The orchestrators ([`crate::block`], [`crate::enum_`]) walk the fields, ask an
//! [`crate::emit`] function to lower each into a [`FieldParts`], and merge the parts
//! into the whole. Each shape's lowering is a self-contained function returning its
//! `FieldParts`, rather than pushing into a dozen loose `Vec`s inline.

use proc_macro2::TokenStream;
use syn::{Field, Ident, Type, Visibility};

use crate::util::Kind;

/// The invariant inputs every field-lowering `emit::*_field` function shares: the
/// enclosing struct's visibility / name / `OnDisk` type / generic parameters, plus
/// the identity of the field being lowered (`fname` / `field` / `kind`). Built once
/// per field in [`crate::block`] and passed by reference, so a new field emitter — or
/// a new input — is one struct field, not another positional argument threaded
/// through a dozen call sites (and it retires the `clippy::too_many_arguments`
/// allows).
///
/// The two inputs that genuinely differ *between calls on the same field* stay
/// explicit arguments, not fields here: `nullable` and the resolved inner/target
/// `Type` — the container dispatch passes `opt_inner` with the `Option`-derived
/// nullability, while the scalar/tuple dispatch passes the POD-adjusted `inner_ty`
/// (a POD field stores its whole type and is never null).
#[derive(Clone, Copy)]
pub(crate) struct FieldCtx<'a> {
    /// The enclosing struct's visibility (every generated method inherits it).
    pub vis: &'a Visibility,
    /// The enclosing struct's name (needed by tuple emitters for wrapper idents).
    pub name: &'a Ident,
    /// The field's name — the on-disk field / struct-literal name, and `get_<f>` stem.
    pub fname: &'a Ident,
    /// The field itself (for spans on errors and attribute inspection).
    pub field: &'a Field,
    /// The field's ownership classification (`owned` / `strong` / `weak` / `ref` / pod).
    pub kind: Kind,
    /// The generated `XOnDisk` type, for `size_of` / `offset_of`.
    pub on_disk_ty: &'a TokenStream,
    /// The struct's generic type parameters (a target mentioning one is not concrete).
    pub type_params: &'a [&'a Ident],
    /// The struct's generic const parameters (array-length generics).
    pub const_params: &'a [&'a Ident],
}

/// The invariant inputs every enum-variant-lowering `emit::*_variant` function
/// shares — the [`FieldCtx`] analogue for the `#[bstack_enum]` path. The variant's
/// own inner `Type` (or `&Variant` for a POD aggregate) and the vec variant's
/// `payload_loc` stay explicit arguments, mirroring how `FieldCtx` leaves the
/// per-call `inner_ty` / `nullable` out.
#[derive(Clone, Copy)]
pub(crate) struct VariantCtx<'a> {
    /// The variant's name (the `EData` / `EView` sum-type arm).
    pub vname: &'a Ident,
    /// The variant's discriminant literal (the `match` arm selector).
    pub disc: &'a TokenStream,
    /// The variant's ownership classification.
    pub kind: Kind,
    /// The generated owned sum type `EData` — `new` / `bstack_move!` traffic in it.
    pub data: &'a Ident,
    /// The generated borrowed sum type `EView` — `read` returns it.
    pub view: &'a Ident,
    /// The enum's `XOnDisk` type (payload offset / size).
    pub on_disk: &'a Ident,
    /// The shared payload-area length constant (bytes).
    pub payload_const: &'a Ident,
}

/// Everything one enum **variant** contributes to the generated block, grouped by
/// the match arm / helper enum it lands in. The parallel of [`FieldParts`] for the
/// `#[bstack_enum]` path: a shape-lowering `emit::*_variant` fills the relevant
/// fields, and [`expand_enum`](crate::enum_::expand_enum) [`merge`](VariantParts::merge)s
/// them across all variants.
///
/// Storage differs from a struct field (a variant packs into the shared payload area
/// at discriminant-dispatched offsets, not its own `OnDisk` field), so this is a
/// distinct bundle — but the *leaf* logic (`foreign_elem_*`, nested arrays, per-kind
/// reconstruction) is the same machinery the field emitters use.
#[derive(Default)]
pub(crate) struct VariantParts {
    /// `EData` enum variant decl(s).
    pub data_variants: Vec<TokenStream>,
    /// `EView` enum variant decl(s).
    pub view_variants: Vec<TokenStream>,
    /// `new` match arm (`EData::V(..) => (disc, payload)`).
    pub new_arms: Vec<TokenStream>,
    /// `read` match arm (`disc => EView::V(..)`).
    pub read_arms: Vec<TokenStream>,
    /// `bstack_move!` / `replace` match arm (`disc => EData::V(..)`).
    pub move_arms: Vec<TokenStream>,
    /// Raw-range hand-back arm for a variant whose old-value reconstruction can
    /// *fail* — only a `#[bstack_strong]` variant (its `strong_parts` reads the
    /// control block). `#disc => Vec<BStackRange>` of the strong child data
    /// block(s), read infallibly from the old payload `__pl`, so a whole-value
    /// enum `replace` whose post-commit old-value reconstruction faults hands the
    /// blocks back through `ReplaceError::lost_raw` instead of leaking them.
    /// Non-strong variants push nothing (their reconstruction
    /// is infallible, so `lost` is unreachable for them) and fall to an empty
    /// default.
    pub raw_arms: Vec<TokenStream>,
    /// Teardown (`__bstack_drop_children`) match arm.
    pub drop_arms: Vec<TokenStream>,
    /// Deep-clone (`__bstack_clone_children_inplace`) match arm.
    pub clone_arms: Vec<TokenStream>,
    /// This variant's payload size (folded into the max-over-variants const).
    pub payload_sizes: Vec<TokenStream>,
    /// POD element types to `__assert_pod` on.
    pub pod_types: Vec<Type>,
    /// `#[embed]`ded child types to `__assert_embeddable` on.
    pub embed_types: Vec<TokenStream>,
    /// This variant is `#[embed]` (its `new` folds the child in post-write).
    pub has_embed: bool,
    /// This variant carries a payload (drives the `read` / `move` payload read).
    pub needs_payload: bool,
    /// This variant holds a strong reference (`EData` becomes `<'e, A>`).
    pub has_shared: bool,
    /// This variant holds a weak reference (`EView` also becomes `<'e, A>`).
    pub has_weak: bool,
    /// This variant holds a `Foreign` (both enums carry the `'__e` lifetime).
    pub has_foreign: bool,
}

impl VariantParts {
    /// Fold another variant's parts into this accumulator.
    pub fn merge(&mut self, other: VariantParts) {
        self.data_variants.extend(other.data_variants);
        self.view_variants.extend(other.view_variants);
        self.new_arms.extend(other.new_arms);
        self.read_arms.extend(other.read_arms);
        self.move_arms.extend(other.move_arms);
        self.drop_arms.extend(other.drop_arms);
        self.raw_arms.extend(other.raw_arms);
        self.clone_arms.extend(other.clone_arms);
        self.payload_sizes.extend(other.payload_sizes);
        self.pod_types.extend(other.pod_types);
        self.embed_types.extend(other.embed_types);
        self.has_embed |= other.has_embed;
        self.needs_payload |= other.needs_payload;
        self.has_shared |= other.has_shared;
        self.has_weak |= other.has_weak;
        self.has_foreign |= other.has_foreign;
    }
}

/// Everything one field contributes to the generated block, grouped by the section
/// it lands in. A shape-lowering `emit::*` function fills the relevant fields and
/// leaves the rest empty; the orchestrator [`merge`](FieldParts::merge)s them.
#[derive(Default)]
pub(crate) struct FieldParts {
    /// `XOnDisk` struct field(s) (`name: Ty,`).
    pub on_disk_fields: Vec<TokenStream>,
    /// Reader / mutator methods on the handle (`get_<f>` / `set_<f>` / `replace_<f>` …).
    pub accessors: Vec<TokenStream>,
    /// Post-construction setters (weak-field wiring).
    pub setters: Vec<TokenStream>,
    /// `new` constructor: parameters, prep statements, and struct-literal inits.
    pub ctor_params: Vec<TokenStream>,
    pub ctor_preps: Vec<TokenStream>,
    pub ctor_inits: Vec<TokenStream>,
    /// One entry per constructor field that **moves in an owning handle** the
    /// caller would otherwise lose on a failed construction — an
    /// `#[bstack_owned]` / `#[bstack_strong]` scalar, an `#[embed]` child, or a
    /// vec / array of those. Each is an expression reconstructing that field's
    /// owning handle (its `bstack_move!` dual) from the in-memory `__on_disk`
    /// image; on a failed step the constructor packs them into a tuple and hands
    /// it back through `ConstructError` so the children are returned intact, not
    /// orphaned. `ctor_handback_ty` holds the matching element
    /// types (the tuple's declared type in `new`'s signature). Fields that move in
    /// *nothing* the caller owns — POD, `#[bstack_ref]`, `Foreign` (a `Copy`
    /// pointer the caller keeps), a setter-wired `#[bstack_weak]` — push to
    /// neither, so a constructor with no owning children keeps a plain
    /// `io::Result`.
    pub ctor_handback: Vec<TokenStream>,
    pub ctor_handback_ty: Vec<TokenStream>,
    /// `new` steps run *after* the block image is written (`#[embed]` copy-in).
    pub ctor_post: Vec<TokenStream>,
    /// Teardown (`__bstack_drop_children`) statements.
    pub drop_stmts: Vec<TokenStream>,
    /// Deep-clone (`__bstack_clone_children_inplace`) statements.
    pub clone_stmts: Vec<TokenStream>,
    /// `bstack_move!` pieces: capture, moved-out field type, reconstruction expr.
    pub mv_caps: Vec<TokenStream>,
    pub mv_types: Vec<TokenStream>,
    pub mv_recon: Vec<TokenStream>,
    /// Helper item definitions the field needs (POD tuple wrappers, embed assertions).
    pub wrapper_defs: Vec<TokenStream>,
    /// POD element types to `__assert_pod` on.
    pub pod_types: Vec<Type>,
    /// A field coerced `&T` → `T` (drives the one-shot coercion warning).
    pub ref_coerced: bool,
}

impl FieldParts {
    /// Fold another field's parts into this one.
    pub fn merge(&mut self, other: FieldParts) {
        self.on_disk_fields.extend(other.on_disk_fields);
        self.accessors.extend(other.accessors);
        self.setters.extend(other.setters);
        self.ctor_params.extend(other.ctor_params);
        self.ctor_preps.extend(other.ctor_preps);
        self.ctor_inits.extend(other.ctor_inits);
        self.ctor_handback.extend(other.ctor_handback);
        self.ctor_handback_ty.extend(other.ctor_handback_ty);
        self.ctor_post.extend(other.ctor_post);
        self.drop_stmts.extend(other.drop_stmts);
        self.clone_stmts.extend(other.clone_stmts);
        self.mv_caps.extend(other.mv_caps);
        self.mv_types.extend(other.mv_types);
        self.mv_recon.extend(other.mv_recon);
        self.wrapper_defs.extend(other.wrapper_defs);
        self.pod_types.extend(other.pod_types);
        self.ref_coerced |= other.ref_coerced;
    }
}
