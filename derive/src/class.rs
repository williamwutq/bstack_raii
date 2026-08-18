//! Implementation of the `#[bstack_class]` attribute macro.
//!
//! `#[bstack_class]` is a **drop-in replacement** for `#[bstack_block]` (structs) and
//! `#[bstack_enum]` (enums) that emits everything they do **plus** a persisted RTTI
//! schema:
//!
//! * it delegates the whole block-machinery codegen to [`crate::block::expand`] /
//!   [`crate::enum_::expand_enum`] (identical handle / `OnDisk` / accessors /
//!   teardown / clone / `new`); then
//! * it emits a `fn() -> RttiType` **descriptor builder** for the type and registers
//!   it into `bstack_raii::rtti::RTTI_TYPES` via `linkme`, so `rtti::sync` can persist
//!   the type's self-describing schema.
//!
//! The descriptor is assembled **at runtime** (once, at sync time) from the generated
//! `XOnDisk` type — `size_of` / `offset_of!` give the on-disk size and each field's
//! (block-relative) offset, and `<T as BStackCast>::eightcc()` gives a referenced
//! block's tag — so the emitter never re-derives layout arithmetic the block macro
//! already owns. It only translates each field / variant **shape** into an
//! `rtti::Shape` node.
//!
//! ## Class variables (`#[bstack_static]`)
//!
//! A `#[bstack_static(EXPR)]` struct field is a **class variable**: stored once inline
//! in the type's schema record (a `CLASS` shape carrying the value bytes), never as a
//! per-instance field. It is stripped from the generated block, and its `EXPR` is
//! encoded (via `bytemuck`) at descriptor-build time. `#[bstack_mut]` on it marks it
//! mutable (a fixed-size, in-place-updatable slot); otherwise it is a constant. Class
//! variables must be POD (Sized) for now.
//!
//! ## Not yet covered
//!
//! Only a **generic** `#[bstack_class]` is refused — a clear `[BSTACK0408]` compile
//! error, never a wrong descriptor — because RTTI needs one concrete `XOnDisk` layout
//! to `size_of` / `offset_of!`. Every non-generic shape is describable: `Foreign`
//! (scalar, `Option`, in a `Vec` / array / tuple — each a `Shape::Foreign` carrying
//! the target tag + ownership kind), POD and foreign tuples, class variables, and all
//! struct-field and enum-variant shapes (including complex variant payloads).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Error, Fields, Ident, ItemEnum, ItemStruct, Type};

use crate::util::*;

/// `#[bstack_class] struct` — emit the `#[bstack_block]` machinery plus an RTTI
/// descriptor builder and its `linkme` registration.
pub fn expand_struct(attr: TokenStream, input: ItemStruct) -> syn::Result<TokenStream> {
    // RTTI needs a single, concrete `XOnDisk` to `size_of` / `offset_of!`.
    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &input.generics,
            "[BSTACK0408] a generic #[bstack_class] is not supported: RTTI needs a single \
             concrete on-disk layout to describe. Use a concrete type, or #[bstack_block] \
             without RTTI.",
        ));
    }

    let parsed = parse_attr(attr.clone())?;
    let rc = parsed.mode != Mode::Plain;
    let weak = parsed.mode == Mode::RcWeak;

    let name = &input.ident;
    let on_disk = format_ident!("{}OnDisk", name);

    // One `RttiField` initializer per field, in declaration order. Instance fields
    // resolve their offset via `offset_of!(XOnDisk, f)`; `#[bstack_static]` fields
    // become a `CLASS` shape and are stripped from the generated block.
    let mut rtti_fields: Vec<TokenStream> = Vec::new();
    let mut static_pod_types: Vec<Type> = Vec::new();
    let mut has_static = false;

    match &input.fields {
        Fields::Named(named) => {
            for f in &named.named {
                let fname = f.ident.clone().expect("named field");
                let fname_str = fname.to_string();
                if let Some(expr) = bstack_static_expr(&f.attrs) {
                    has_static = true;
                    let expr = expr.map_err(|_| {
                        Error::new_spanned(
                            f,
                            format!(
                                "[BSTACK0010] field `{fname_str}`: `#[bstack_static]` needs an \
                                 initial value, e.g. `#[bstack_static(0u32)]`"
                            ),
                        )
                    })?;
                    if classify(f)? != Kind::Pod {
                        return Err(Error::new_spanned(
                            f,
                            format!(
                                "[BSTACK0010] field `{fname_str}`: a `#[bstack_static]` class \
                                 variable must be POD — it cannot also own a block \
                                 (`#[bstack_owned/strong/weak/ref]` / `#[embed]`)"
                            ),
                        ));
                    }
                    static_pod_types.push(f.ty.clone());
                    rtti_fields.push(class_field(
                        &fname_str,
                        &f.ty,
                        is_bstack_mut(&f.attrs),
                        &expr,
                    ));
                } else {
                    let kind = classify(f)?;
                    let shape = field_shape(&fname_str, f, &f.ty, kind)?;
                    rtti_fields.push(instance_field(&fname_str, &on_disk, &fname, shape));
                }
            }
        }
        Fields::Unnamed(unnamed) => {
            for (i, f) in unnamed.unnamed.iter().enumerate() {
                if is_bstack_static(&f.attrs) {
                    return Err(Error::new_spanned(
                        f,
                        "[BSTACK0010] `#[bstack_static]` class variables are only supported on \
                         named struct fields",
                    ));
                }
                let fname = format_ident!("field{i}");
                let kind = classify(f)?;
                let shape = field_shape(&fname.to_string(), f, &f.ty, kind)?;
                rtti_fields.push(instance_field(&i.to_string(), &on_disk, &fname, shape));
            }
        }
        Fields::Unit => {}
    }

    // The block machinery. If there are class variables, strip them from the struct
    // the block macro sees (they are not per-instance fields).
    let instance_input = if has_static {
        let mut cloned = input.clone();
        if let Fields::Named(named) = &mut cloned.fields {
            named.named = named
                .named
                .iter()
                .filter(|f| !is_bstack_static(&f.attrs))
                .cloned()
                .collect();
        }
        cloned
    } else {
        input.clone()
    };
    let block_ts = crate::block::expand(attr, instance_input)?;

    let name_str = name.to_string();
    let rtti_type = quote! {
        ::bstack_raii::rtti::RttiType {
            tag: <#name as ::bstack_raii::BStackCast>::eightcc(),
            name: ::std::string::String::from(#name_str),
            rc: #rc,
            weak: #weak,
            ondisk_size: ::core::mem::size_of::<#on_disk>() as u64,
            body: ::bstack_raii::rtti::RttiBody::Struct(::std::vec![
                #(#rtti_fields),*
            ]),
        }
    };
    let reg_ts = registration(name, rtti_type);
    let pod_assert = pod_assertion(&static_pod_types);

    Ok(quote! {
        #block_ts
        #pod_assert
        #reg_ts
    })
}

/// `#[bstack_class] enum` — emit the `#[bstack_enum]` machinery plus an RTTI
/// descriptor (discriminant table + per-variant shapes) and its registration.
pub fn expand_enum(attr: TokenStream, input: ItemEnum) -> syn::Result<TokenStream> {
    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &input.generics,
            "[BSTACK0408] a generic #[bstack_class] is not supported: RTTI needs a single \
             concrete on-disk layout to describe. Use a concrete type, or #[bstack_enum] \
             without RTTI.",
        ));
    }

    let parsed = parse_attr(attr.clone())?;
    let rc = parsed.mode != Mode::Plain;
    let weak = parsed.mode == Mode::RcWeak;

    // The resolved discriminant width + per-variant values (declaration order).
    let disc = crate::layout::discriminants(&input.variants, &parsed.repr)?;
    let disc_ty = &disc.ty;

    let name = &input.ident;
    let on_disk = format_ident!("{}OnDisk", name);

    let mut variant_toks: Vec<TokenStream> = Vec::with_capacity(input.variants.len());
    for (variant, value) in input.variants.iter().zip(disc.values.iter()) {
        variant_toks.push(enum_variant(variant, *value)?);
    }

    let block_ts = crate::enum_::expand_enum(attr, input.clone())?;

    let name_str = name.to_string();
    let rtti_type = quote! {
        ::bstack_raii::rtti::RttiType {
            tag: <#name as ::bstack_raii::BStackCast>::eightcc(),
            name: ::std::string::String::from(#name_str),
            rc: #rc,
            weak: #weak,
            ondisk_size: ::core::mem::size_of::<#on_disk>() as u64,
            body: ::bstack_raii::rtti::RttiBody::Enum(::bstack_raii::rtti::RttiEnum {
                disc_width: ::core::mem::size_of::<#disc_ty>() as u8,
                disc_off: ::core::mem::offset_of!(#on_disk, __bstack_disc) as u16,
                payload_off: ::core::mem::offset_of!(#on_disk, __bstack_payload) as u16,
                variants: ::std::vec![
                    #(#variant_toks),*
                ],
            }),
        }
    };
    let reg_ts = registration(name, rtti_type);

    Ok(quote! {
        #block_ts
        #reg_ts
    })
}

/// One enum variant's `RttiVariant` initializer: its name, discriminant, and payload
/// fields (offsets relative to the payload area).
fn enum_variant(variant: &syn::Variant, value: i128) -> syn::Result<TokenStream> {
    let vname = variant.ident.to_string();
    let disc = value as i64;
    let kind = classify_attrs(&variant.attrs)?;

    let fields: Vec<TokenStream> = match variant.fields.len() {
        0 => Vec::new(),
        1 => {
            // A single-field variant — annotated (`#[owned] V(T)`) or a bare
            // POD / `Vec` / array / tuple (`V(u32)`, `V(Vec<u32>)`, `V([u8; 4])`) — is
            // one field at payload offset 0, described by `field_shape` (the same
            // struct-field lowering: scalar reference, `Foreign`, container, `Option`).
            let f = variant.fields.iter().next().unwrap();
            let fname = f
                .ident
                .as_ref()
                .map(|id| id.to_string())
                .unwrap_or_else(|| "0".to_string());
            let shape = field_shape(&fname, f, &f.ty, kind)?;
            vec![quote! {
                ::bstack_raii::rtti::RttiField {
                    name: ::std::string::String::from(#fname),
                    offset: 0u32,
                    shape: #shape,
                }
            }]
        }
        _ => {
            // A multi-field POD aggregate (`V(A, B)` / `V { x, y }`): fields packed at
            // cumulative `size_of` byte offsets, declaration order (unaligned — see
            // `emit::pod_aggregate_variant`).
            let mut preceding: Vec<&Type> = Vec::new();
            let mut out = Vec::with_capacity(variant.fields.len());
            for (i, f) in variant.fields.iter().enumerate() {
                let fname = f
                    .ident
                    .as_ref()
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| i.to_string());
                let ty = &f.ty;
                let offset = if preceding.is_empty() {
                    quote!(0u32)
                } else {
                    let sizes = preceding
                        .iter()
                        .map(|t| quote!(::core::mem::size_of::<#t>()));
                    quote!((#(#sizes)+*) as u32)
                };
                out.push(quote! {
                    ::bstack_raii::rtti::RttiField {
                        name: ::std::string::String::from(#fname),
                        offset: #offset,
                        shape: ::bstack_raii::rtti::Shape::Pod {
                            width: ::core::mem::size_of::<#ty>() as u32,
                        },
                    }
                });
                preceding.push(ty);
            }
            out
        }
    };

    Ok(quote! {
        ::bstack_raii::rtti::RttiVariant {
            name: ::std::string::String::from(#vname),
            disc_value: #disc,
            fields: ::std::vec![
                #(#fields),*
            ],
        }
    })
}

/// The `Shape::<Variant>` ident for a reference ownership kind, `None` for POD.
fn kind_variant(kind: Kind) -> Option<TokenStream> {
    Some(match kind {
        Kind::Owned => quote!(Owned),
        Kind::Strong => quote!(Strong),
        Kind::Weak => quote!(Weak),
        Kind::Ref => quote!(Ref),
        Kind::Embed => quote!(Embed),
        Kind::Pod => return None,
    })
}

/// A per-instance field's `RttiField`: name, `offset_of!`-resolved offset, and shape.
fn instance_field(name: &str, on_disk: &Ident, fname: &Ident, shape: TokenStream) -> TokenStream {
    quote! {
        ::bstack_raii::rtti::RttiField {
            name: ::std::string::String::from(#name),
            offset: ::core::mem::offset_of!(#on_disk, #fname) as u32,
            shape: #shape,
        }
    }
}

/// A `#[bstack_static]` class variable's `RttiField`: a `CLASS` shape carrying its
/// (bytemuck-encoded) value; `offset` is unused (class vars are not per-instance).
fn class_field(name: &str, ty: &Type, mutable: bool, expr: &syn::Expr) -> TokenStream {
    quote! {
        ::bstack_raii::rtti::RttiField {
            name: ::std::string::String::from(#name),
            offset: 0u32,
            shape: ::bstack_raii::rtti::Shape::Class {
                mutable: #mutable,
                inner: ::std::boxed::Box::new(::bstack_raii::rtti::Shape::Pod {
                    width: ::core::mem::size_of::<#ty>() as u32,
                }),
                value: {
                    let __v: #ty = #expr;
                    ::bstack_raii::bytemuck::bytes_of(&__v).to_vec()
                },
            },
        }
    }
}

/// A `T: Pod` assertion for every `#[bstack_static]` type, so a non-POD class variable
/// gets a directed error rather than a raw `bytes_of` bound failure.
fn pod_assertion(types: &[Type]) -> TokenStream {
    if types.is_empty() {
        return quote!();
    }
    quote! {
        const _: fn() = || {
            fn __assert_static_pod<__T: ::bstack_raii::Pod>() {}
            #( __assert_static_pod::<#types>(); )*
        };
    }
}

/// The descriptor builder `fn` + its link-time registration. `#[linkme(crate = …)]`
/// points linkme at our re-export so a downstream crate needs no direct linkme
/// dependency (linkme otherwise hard-codes `::linkme`).
fn registration(name: &Ident, rtti_type: TokenStream) -> TokenStream {
    let build_fn = format_ident!("__bstack_rtti_build_{}", name);
    let reg_static = format_ident!("__BSTACK_RTTI_REG_{}", name);
    quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #build_fn() -> ::bstack_raii::rtti::RttiType {
            #rtti_type
        }

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #[::bstack_raii::rtti::distributed_slice_reexport(::bstack_raii::rtti::RTTI_TYPES)]
        #[linkme(crate = ::bstack_raii::rtti::linkme)]
        static #reg_static: ::bstack_raii::rtti::RttiRegistration =
            ::bstack_raii::rtti::RttiRegistration { build: #build_fn };
    }
}

/// Translate one field's type + ownership kind into a token stream that builds its
/// `rtti::Shape` at runtime. Recurses through `Option` / `[T; N]` / `Vec` / `String`
/// down to a leaf (`Pod` or a block reference); the ownership `kind` applies to the
/// leaf.
fn field_shape(fname: &str, field: &syn::Field, ty: &Type, kind: Kind) -> syn::Result<TokenStream> {
    // `Foreign` composes inside a `Vec` / array / tuple (its 16-byte `ForeignRepr`
    // is the container's element / member); the recursion below reaches the
    // `Foreign` leaf in each case.

    // `Option<Inner>` — nullable leaf / child.
    if let Some(inner) = option_inner(ty) {
        let inner_shape = leaf_or_container_shape(fname, ty, inner, kind)?;
        return Ok(
            quote!(::bstack_raii::rtti::Shape::Option(::std::boxed::Box::new(#inner_shape))),
        );
    }

    let _ = field;
    leaf_or_container_shape(fname, ty, ty, kind)
}

/// The non-`Option` part of the shape: a `Vec` / `String` region, an `[T; N]` array,
/// or a leaf. `orig` is the whole field type (for error spans); `ty` is the current
/// (possibly `Option`-peeled) type.
fn leaf_or_container_shape(
    fname: &str,
    orig: &Type,
    ty: &Type,
    kind: Kind,
) -> syn::Result<TokenStream> {
    // `Vec<T>` / `String` — a dynamically-sized region; the element carries the
    // field's ownership kind (POD bytes, or block offsets).
    if let Some(vi) = vec_info(ty) {
        if vi.is_string {
            return Ok(quote!(::bstack_raii::rtti::Shape::Vec(
                ::std::boxed::Box::new(::bstack_raii::rtti::Shape::Pod { width: 1 })
            )));
        }
        let elem = vec_inner(ty).expect("vec_info matched");
        let elem_shape = leaf_shape(fname, orig, elem, kind)?;
        return Ok(quote!(::bstack_raii::rtti::Shape::Vec(::std::boxed::Box::new(#elem_shape))));
    }

    // `[T; N]` (possibly nested `[[T; N]; M]`, possibly `[Option<T>; N]`).
    if matches!(ty, Type::Array(_)) {
        let (dims, leaf, nullable) = array_shape(ty)?;
        let mut acc = leaf_shape(fname, orig, leaf, kind)?;
        if nullable {
            acc = quote!(::bstack_raii::rtti::Shape::Option(::std::boxed::Box::new(#acc)));
        }
        for dim in dims.iter().rev() {
            acc = quote!(::bstack_raii::rtti::Shape::Array {
                n: (#dim) as u32,
                inner: ::std::boxed::Box::new(#acc),
            });
        }
        return Ok(acc);
    }

    leaf_shape(fname, orig, ty, kind)
}

/// The `ForeignKind` variant for a foreign field's ownership annotation, or `None`
/// (a `Foreign` must be annotated — never POD / `#[embed]`).
fn foreign_kind_variant(kind: Kind) -> Option<TokenStream> {
    Some(match kind {
        Kind::Owned => quote!(Owned),
        Kind::Strong => quote!(Strong),
        Kind::Weak => quote!(Weak),
        Kind::Ref => quote!(Ref),
        Kind::Pod | Kind::Embed => return None,
    })
}

/// A single leaf: a `Pod` value (its byte width), a cross-file `Foreign<T>` (its target
/// tag + ownership kind), or an in-file block reference (its target's tag), selected by
/// the ownership `kind`.
fn leaf_shape(fname: &str, orig: &Type, ty: &Type, kind: Kind) -> syn::Result<TokenStream> {
    // A scalar `Foreign<T>`: emit the target tag + the ownership kind.
    if let Some(target) = foreign_inner(ty) {
        let fk = foreign_kind_variant(kind).ok_or_else(|| {
            Error::new_spanned(
                orig,
                format!(
                    "[BSTACK0302] field `{fname}`: a `Foreign` needs an ownership annotation \
                     (#[bstack_owned/strong/weak/ref])"
                ),
            )
        })?;
        return Ok(quote!(::bstack_raii::rtti::Shape::Foreign {
            tag: <#target as ::bstack_raii::BStackCast>::eightcc(),
            kind: ::bstack_raii::rtti::ForeignKind::#fk,
        }));
    }
    if let Type::Tuple(tup) = ty {
        // A tuple field is a `Shape::Tuple` of its members' shapes, laid out inline at
        // cumulative offsets (matching the static macro). A member is either a
        // cross-file `Foreign` (optionally `Option`-null, using the offset-0 niche —
        // all foreign members share the field's ownership annotation) or an opaque POD
        // value of its **whole** width. A POD `Option<PodInOption>` member is *not*
        // peeled: its niche is bytemuck's, not offset-0, so it stays a single `Pod`.
        let has_foreign = tup
            .elems
            .iter()
            .any(|e| foreign_inner(option_inner(e).unwrap_or(e)).is_some());
        let fk = if has_foreign {
            Some(foreign_kind_variant(kind).ok_or_else(|| {
                Error::new_spanned(
                    orig,
                    format!(
                        "[BSTACK0302] field `{fname}`: a tuple containing a `Foreign` needs an \
                         ownership annotation (#[bstack_owned/strong/weak/ref]) naming the \
                         foreign elements' kind"
                    ),
                )
            })?)
        } else {
            None
        };
        let mut elems = Vec::with_capacity(tup.elems.len());
        for e in &tup.elems {
            let es = if let Some(target) = foreign_inner(option_inner(e).unwrap_or(e)) {
                let fk = fk.clone().expect("foreign member ⇒ has_foreign");
                let leaf = quote!(::bstack_raii::rtti::Shape::Foreign {
                    tag: <#target as ::bstack_raii::BStackCast>::eightcc(),
                    kind: ::bstack_raii::rtti::ForeignKind::#fk,
                });
                if option_inner(e).is_some() {
                    quote!(::bstack_raii::rtti::Shape::Option(::std::boxed::Box::new(#leaf)))
                } else {
                    leaf
                }
            } else {
                // Opaque POD of the whole member type (an `Option<PodInOption>` keeps
                // its bytemuck niche inline).
                quote!(::bstack_raii::rtti::Shape::Pod {
                    width: ::core::mem::size_of::<#e>() as u32,
                })
            };
            elems.push(es);
        }
        return Ok(quote!(::bstack_raii::rtti::Shape::Tuple(
            ::std::vec![#(#elems),*]
        )));
    }
    Ok(match kind {
        Kind::Pod => quote!(::bstack_raii::rtti::Shape::Pod {
            width: ::core::mem::size_of::<#ty>() as u32,
        }),
        other => {
            let variant = kind_variant(other).expect("non-POD kind");
            quote!(::bstack_raii::rtti::Shape::#variant(
                <#ty as ::bstack_raii::BStackCast>::eightcc()
            ))
        }
    })
}
