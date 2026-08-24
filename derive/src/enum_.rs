//! Implementation of the `#[bstack_enum]` attribute macro (orchestrator). The
//! shared classification / analysis / emit machinery lives in [`crate::common`].

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Error, Fields, Ident};

use crate::emit::*;
use crate::layout;
use crate::model::{VariantCtx, VariantParts};
use crate::util::*;

/// Implementation of the `#[bstack_enum]` attribute macro.
pub fn expand_enum(attr: TokenStream, input: syn::ItemEnum) -> syn::Result<TokenStream> {
    let attr = parse_attr(attr)?;
    let mode = attr.mode;

    // Generic enums (layout-preserving): a type parameter may appear only in a
    // *reference* variant (`#[bstack_owned/strong/weak/ref] V(T)`, and array / vec
    // forms) — each a bare `u64` offset (or `VecDesc`) in the payload, keeping the
    // fixed payload size independent of the parameter. A POD or `#[embed]` variant
    // stores the parameter inline, so its payload width would depend on it —
    // rejected. Const / lifetime parameters and non-plain modes are not supported.
    let type_params: Vec<&Ident> = input.generics.type_params().map(|tp| &tp.ident).collect();
    if !input.generics.params.is_empty() {
        for p in &input.generics.params {
            if !matches!(p, syn::GenericParam::Type(_)) {
                return Err(Error::new_spanned(
                    p,
                    "[BSTACK0403] a generic #[bstack_enum] currently supports only type parameters (no \
                     lifetime or const generics)",
                ));
            }
        }
        if mode != Mode::Plain {
            return Err(Error::new_spanned(
                &input.generics,
                "[BSTACK0402] a generic #[bstack_enum] currently supports plain mode only (not `rc` / \
                 `rc, weak`)",
            ));
        }
    }
    #[derive(Default)]
    struct EUsage {
        strong: bool,
        weak: bool,
        /// The param is the target of a `#[bstack_owned] V(Foreign<T>)` variant (an
        /// owned foreign deep-clone runs `try_clone_in`, needing `TryCloneIn`).
        foreign_owned: bool,
        /// The param is the target of *any* `Foreign<T>` variant; `Foreign<'a, T>`
        /// requires `T: 'static`.
        foreign: bool,
    }
    let mut eusage: Vec<(Ident, EUsage)> = type_params
        .iter()
        .map(|p| ((*p).clone(), EUsage::default()))
        .collect();
    for variant in &input.variants {
        let kind = classify_attrs(&variant.attrs)?;
        // A `#[bstack_mut]` mutator on an enum is *whole-value* (the payload's
        // meaning depends on the discriminant, so there is no per-variant field to
        // set). Put `#[bstack_mut]` on the enum itself — it is a no-op / error on a
        // variant.
        if is_bstack_mut(&variant.attrs) {
            return Err(Error::new_spanned(
                variant,
                "[BSTACK0603] #[bstack_mut] on a `#[bstack_enum]` goes on the enum itself (a whole-value \
                 `set` / `replace`), not on a variant — a variant has no separately \
                 mutable field",
            ));
        }
        for f in &variant.fields {
            if !type_mentions_any(&f.ty, &type_params) {
                continue;
            }
            let ftargets = foreign_targets_in(&f.ty);
            if ftargets.is_empty() && (kind == Kind::Pod || kind == Kind::Embed) {
                return Err(Error::new_spanned(
                    &f.ty,
                    "[BSTACK0406] a generic type parameter in a `#[bstack_enum]` variant must be a reference \
                     (`#[bstack_owned]` / `#[bstack_strong]` / `#[bstack_weak]` / `#[bstack_ref]`), \
                     not stored inline — a POD or `#[embed]` variant's payload width would depend \
                     on the parameter",
                ));
            }
            for (p, u) in eusage.iter_mut() {
                if !type_mentions_any(&f.ty, &[&*p]) {
                    continue;
                }
                let is_ftarget = ftargets.iter().any(|t| type_mentions_any(t, &[&*p]));
                if !ftargets.is_empty() && !is_ftarget {
                    return Err(Error::new_spanned(
                        &f.ty,
                        "[BSTACK0405] a generic type parameter in a non-`Foreign` position of a `Foreign` \
                         variant is not supported; use concrete types for the non-foreign parts",
                    ));
                }
                u.strong |= kind == Kind::Strong;
                u.weak |= kind == Kind::Weak;
                u.foreign_owned |= is_ftarget && kind == Kind::Owned;
                u.foreign |= is_ftarget;
            }
        }
    }
    let mut aug_generics = input.generics.clone();
    for tp in aug_generics.type_params_mut() {
        tp.bounds
            .push(syn::parse_quote!(::bstack_raii::BStackBlock));
        if let Some((_, u)) = eusage.iter().find(|(p, _)| *p == tp.ident) {
            if u.strong {
                tp.bounds
                    .push(syn::parse_quote!(::bstack_raii::BStackShared));
            }
            if u.weak {
                tp.bounds
                    .push(syn::parse_quote!(::bstack_raii::BStackWeakable));
            }
            if u.foreign_owned {
                tp.bounds.push(syn::parse_quote!(::bstack_raii::TryCloneIn));
            }
            if u.foreign {
                tp.bounds.push(syn::parse_quote!('static));
            }
        }
    }
    let (enum_impl_g, enum_ty_g, enum_where) = aug_generics.split_for_impl();
    let (enum_decl_g, enum_decl_ty_g, enum_decl_where) = input.generics.split_for_impl();
    let (enum_phantom_field, enum_phantom_ctor): (TokenStream, TokenStream) =
        if type_params.is_empty() {
            (quote!(), quote!())
        } else {
            (
                quote!(, ::core::marker::PhantomData<fn() -> (#(#type_params,)*)>),
                quote!(, ::core::marker::PhantomData),
            )
        };

    // The on-disk discriminant width + per-variant literal patterns.
    let layout::Discriminants {
        ty: disc_ty,
        pats: disc_pats,
        ..
    } = layout::discriminants(&input.variants, &attr.repr)?;

    let name = &input.ident;
    let vis = &input.vis;
    let on_disk = format_ident!("{}OnDisk", name);
    let control = format_ident!("{}OnDiskRef", name);
    let payload_const = format_ident!("__bstack_payload_{}", name);
    // `EData` is the in-memory owned form of the enum's payload — the same type is
    // used to *construct* (`E::new`) and to receive a *destructured* variant
    // (`bstack_move!`), since both hold owned handles (they are duals).
    let data = format_ident!("{}Data", name);
    let view = format_ident!("{}View", name);

    // Every variant's contributions accumulate here (the enum's Slot IR bundle):
    // the `EData` / `EView` variant decls, the `new` / `read` / `move` / `drop` /
    // `clone` match arms, per-variant payload sizes, POD / embed assertion types, and
    // the flags that shape the companion enums' generics (a strong/weak variant makes
    // `EData` `<'e, A>`; a weak one also makes `EView`; a `Foreign` variant makes both
    // carry `'__e`; `#[embed]` folds the child in post-write; `needs_payload` drives
    // the payload read). Enum embed is always concrete (a generic embed variant is
    // rejected in `EUsage`), so `embed_types` is asserted `BStackEmbeddable`.
    let mut vp = VariantParts::default();

    for (i, variant) in input.variants.iter().enumerate() {
        let disc = &disc_pats[i];
        let vname = &variant.ident;
        let kind = classify_attrs(&variant.attrs)?;
        // The invariant inputs every `emit::*_variant` lowering shares, bundled once
        // (the parallel of `FieldCtx` for the enum path).
        let vctx = VariantCtx {
            vname,
            disc,
            kind,
            data: &data,
            view: &view,
            on_disk: &on_disk,
            payload_const: &payload_const,
        };

        match &variant.fields {
            // Annotated single-field tuple `#[..] V(T)`: an owned / strong / weak /
            // ref child stored as a `u64` offset. (Unit, un-annotated single-POD,
            // multi-field tuple, and struct variants are POD aggregates, below.)
            // Annotated single-field `#[..] V(T)`, OR a POD `V(Vec<T>)` / `V(String)`
            // (which needs the vec machinery, not the byte-packed POD aggregate).
            Fields::Unnamed(f)
                if f.unnamed.len() == 1
                    && (kind != Kind::Pod
                        || vec_info(&f.unnamed.first().unwrap().ty).is_some()
                        || foreign_inner(&f.unnamed.first().unwrap().ty).is_some()) =>
            {
                vp.needs_payload = true;
                let ty = &f.unnamed.first().unwrap().ty;

                // Annotated **vector** variant `#[..] V(Vec<T>)` / `V(Vec<[T; N]>)`:
                // a `VecDesc` (16 bytes) in the payload naming a data block — the
                // per-variant mirror of a `#[bstack_owned/strong/weak/ref] Vec<..>`
                // struct field. A `Vec<[T; N]>` stores its offsets FLAT (like the
                // struct case), reshaped to `Vec<[[T;..];..]>` on read.
                let payload_loc = quote! {
                    ::bstack_raii::__private::checked_field_offset(self.0.start(), ::core::mem::offset_of!(#on_disk, __bstack_payload) as u64)?
                };
                if let Some(p) = vec_variant(&vctx, ty, &payload_loc)? {
                    vp.merge(p);
                    continue;
                }

                // Annotated **array** variant `#[..] V([T; N])`: N block references
                // stored inline in the payload as `[u64; N]` (N*8 bytes), the
                // per-element mirror of a `#[bstack_owned/strong/weak/ref] V(T)`.
                // Annotated **array** variant `#[..] V([T; N])` — possibly nested
                // `[[..]; ..]`, and (for owned/strong/ref) possibly with `Option<T>`
                // leaves. Block refs are stored **flat** in the payload as
                // `[u64; TOTAL]` (TOTAL*8 bytes), the per-element mirror of a
                // `#[bstack_owned/strong/weak/ref] V(T)`; `#[embed]` stores each
                // child's whole on-disk form verbatim.
                if let Some(p) = array_variant(&vctx, ty)? {
                    vp.merge(p);
                    continue;
                }

                // Annotated **foreign** variant `#[..] V(Foreign<T>)`: a cross-file
                // wide pointer stored as a 16-byte `ForeignPtr` in the payload. The
                // annotation names the target's ownership in its own file (teardown /
                // clone dispatch cross-file, like a scalar `Foreign` struct field).
                // Concrete target only for now; container-in-variant is not handled.
                if let Some(p) = foreign_variant(&vctx, ty)? {
                    vp.merge(p);
                    continue;
                }

                // Annotated **foreign tuple** variant `#[..] V((A, Foreign<T>, ..))`:
                // POD elements packed inline, each foreign element a 16-byte
                // `ForeignPtr`, all at cumulative byte offsets in the payload (the
                // per-variant mirror of a `#[ann] (A, Foreign<T>)` struct field). The
                // annotation names the foreign elements' ownership.
                if let Some(p) = foreign_tuple_variant(&vctx, ty)? {
                    vp.merge(p);
                    continue;
                }

                vp.merge(single_block_variant(&vctx, ty)?);
            }
            // A POD aggregate: unit, an all-POD tuple `V(A, B, ..)`, or an all-POD
            // struct `V { x: A, .. }`. The fields are packed sequentially into the
            // payload (declaration order). This is sound because the payload is
            // read/written **unaligned**, so field alignment is irrelevant — the
            // packed byte sequence of POD fields is itself just POD bytes.
            _ => {
                vp.merge(pod_aggregate_variant(&vctx, variant)?);
            }
        }
    }

    // Unpack the accumulated parts into the names the assembly below reads.
    let VariantParts {
        data_variants,
        view_variants,
        new_arms,
        read_arms,
        move_arms,
        drop_arms,
        raw_arms,
        clone_arms,
        payload_sizes,
        pod_types,
        embed_types,
        has_embed: enum_has_embed,
        needs_payload,
        has_shared,
        has_weak,
        has_foreign,
    } = vp;

    // EightCC tag: readable prefix over a hash of `crate ++ type_name` (as structs).
    let type_name = name.to_string();
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let hash = fnv1a64(&format!("{crate_name}\0{type_name}"));
    let prefix = attr.tag.as_ref().map_or_else(
        || auto_prefix(&type_name),
        |t| t.bytes().collect::<Vec<u8>>(),
    );
    let tag = build_tag(hash, &prefix);
    let data_base = eightcc_expr(&tag.bytes);
    // For a generic enum, fold each type argument's tag into the discriminant so
    // distinct instantiations get distinct tags (mixed at runtime in `eightcc()`).
    let data_mixed = if type_params.is_empty() {
        data_base
    } else {
        let mixes = type_params
            .iter()
            .map(|p| quote!(.mix(<#p as ::bstack_raii::BStackCast>::eightcc())));
        quote!(#data_base #(#mixes)*)
    };
    // Fold the module path so same-named enums in different modules stay distinct
    // (issue 2). Runtime `module_path!()`, consistent with the on-disk header and
    // the RTTI registry (both go through this `eightcc()`).
    let eightcc = quote!(#data_mixed.mix_str(::core::module_path!()));

    // Control-block tag (rc, weak). Default: the data tag with a reserved hash bit
    // toggled — same readable prefix, structurally distinct (issue 36). Explicit
    // `ctrl_tag =` override: its own readable prefix, still module-path-folded.
    let (ctrl_eightcc, ctrl_truncated) = match attr.ctrl_tag.as_ref() {
        None => (quote!(#eightcc.with_ctrl_bit()), false),
        Some(t) => {
            let ctrl_prefix = t.bytes().collect::<Vec<u8>>();
            let ctrl_tag = build_tag(hash, &ctrl_prefix);
            // [BSTACK0006] An explicit `ctrl_tag` equal to the data tag collapses
            // the data/control distinction. The default (`with_ctrl_bit`) can't
            // reach this, so only the override is checked.
            if ctrl_tag.bytes == tag.bytes {
                return Err(Error::new_spanned(
                    &input.ident,
                    "[BSTACK0006] the explicit `ctrl_tag` equals the data tag — a \
                     control block would then pass every data-block identity check; \
                     choose a distinct `ctrl_tag` (or omit it to auto-derive one)",
                ));
            }
            let ctrl_base = eightcc_expr(&ctrl_tag.bytes);
            (
                quote!(#ctrl_base.mix_str(::core::module_path!())),
                ctrl_tag.truncated,
            )
        }
    };

    // Refcount / control machinery, mirroring the struct rc modes: an injected
    // field after the header, `BStackShared` (rc / rc,weak), and (rc, weak) a
    // control block + `BStackWeakable`. `new` returns `BStackRc` for rc modes.
    let injected_ondisk = match mode {
        Mode::Plain => quote!(),
        Mode::Rc => quote!(__bstack_refcount: u64,),
        Mode::RcWeak => quote!(__bstack_ctrl: u64,),
    };
    let new_ret = match mode {
        Mode::Plain => quote!(::bstack_raii::BStackOwned<Self>),
        _ => quote!(::bstack_raii::BStackRc<'__e, Self, __A>),
    };
    // `EData` type name — `new`'s `data` parameter and `bstack_move!` output.
    // The companion `EData` / `EView` enums are generic over `<'e, A>` (when a
    // strong/weak variant needs it) AND over the enum's own type parameters. These
    // helpers assemble a use (`EData<'e, A, T>`) or a decl (`<'e, A, T: Bound>`)
    // with the right subset present.
    let etp_args: Vec<TokenStream> = type_params.iter().map(|p| quote!(#p)).collect();
    let etp_decl: Vec<TokenStream> = aug_generics.type_params().map(|tp| quote!(#tp)).collect();
    // `want_lt` and `want_a` are decoupled: a strong/weak variant needs *both* the
    // lifetime and `__A` (`BStackRc<'__e, T, __A>`), but a `Foreign` variant needs only
    // the lifetime (`Foreign<'__e, T>` binds a `SELF` pointer to the read/move borrow,
    // carrying no allocator) — so an enum with a `Foreign` variant but no shared/weak
    // one gets `'__e` without an unused `__A` param.
    let comp_ty = |id: &Ident, lt: &TokenStream, want_lt: bool, want_a: bool| -> TokenStream {
        let mut args: Vec<TokenStream> = Vec::new();
        if want_lt {
            args.push(lt.clone());
        }
        if want_a {
            args.push(quote!(__A));
        }
        args.extend(etp_args.iter().cloned());
        if args.is_empty() {
            quote!(#id)
        } else {
            quote!(#id < #(#args),* >)
        }
    };
    let comp_decl = |lt: &TokenStream, want_lt: bool, want_a: bool| -> TokenStream {
        let mut parts: Vec<TokenStream> = Vec::new();
        if want_lt {
            parts.push(lt.clone());
        }
        if want_a {
            parts.push(quote!(__A: ::bstack_raii::BStackRaiiAllocator));
        }
        parts.extend(etp_decl.iter().cloned());
        if parts.is_empty() {
            quote!()
        } else {
            quote!(< #(#parts),* >)
        }
    };
    let data_ty = comp_ty(&data, &quote!('__e), has_shared || has_foreign, has_shared);
    // The `new` constructor. Plain / `rc` are one atomic write (the injected
    // refcount is baked into the image); `(rc, weak)` allocates data + control and
    // commits both images in one `set_batched`, with the control back-pointer
    // baked into the data image — no separate back-pointer write, no half-wired
    // transient state.
    let enum_header = quote! {
        __bstack_header: ::bstack_raii::BlockHeader {
            size: ::core::mem::size_of::<#on_disk>() as u64,
            tag: <Self as ::bstack_raii::BStackCast>::eightcc(),
        },
    };
    let enum_size = quote!(::core::mem::size_of::<#on_disk>() as u64);
    // For an `#[embed]` variant, `new` declares a captured child range (set by the
    // active variant's arm) and, after writing the OnDisk with a zeroed payload,
    // `BStack::copy`s the child into the payload and frees its shell.
    let (embed_decl, embed_post) = if enum_has_embed {
        (
            // Each entry: (child source range, byte size, destination offset WITHIN
            // the payload). A scalar `#[embed] V(Child)` pushes one entry at offset
            // 0; an `#[embed] V([Child; N])` (or nested) pushes one per flat slot.
            quote!(let mut __embed_copy:
                ::std::vec::Vec<(::bstack_raii::BStackRange, u64, u64)> =
                ::std::vec::Vec::new();),
            quote! {
                for (__cr, __sz, __doff) in __embed_copy {
                    allocator.stack().copy(
                        __cr.start(),
                        __data.start()
                            + ::core::mem::offset_of!(#on_disk, __bstack_payload) as u64
                            + __doff,
                        __sz,
                    )?;
                    unsafe { ::bstack_raii::dealloc_range(allocator, __cr)?; }
                }
            },
        )
    } else {
        (quote!(), quote!())
    };
    // Hand-back for the enum constructor: when a variant owns
    // children (`drop_arms` non-empty), a failed `new` reconstructs the consumed
    // `EData` (via the same per-variant `#move_arms` `bstack_move!` uses) and hands
    // it back through `ConstructError` rather than orphaning it. Excluded for an
    // enum with an `#[embed]` variant: its payload slot is a zeroed placeholder at
    // construction time (the child is copied in post-write), so `#move_arms` — which
    // re-homes from the payload bytes — cannot reconstruct it; such an enum keeps a
    // plain `io::Result` (it never had failure reclaim, so this is no regression).
    let enum_has_owning = !drop_arms.is_empty() && !enum_has_embed;
    let reconstruct_decl = if enum_has_owning {
        // The `#move_arms` name `bstack_move`'s `'__mv`; re-home them to the
        // constructor's `'__e` so the reconstruction type-checks inside `new`.
        let move_arms_e: Vec<TokenStream> = move_arms
            .iter()
            .map(|a| rename_lifetime(a, "__mv", "__e"))
            .collect();
        quote! {
            #[allow(unused_variables)]
            let __alloc = allocator;
            #[allow(unused_variables)]
            let __reconstruct = |__disc: #disc_ty, __pl: [u8; #payload_const]|
                -> ::std::io::Result<#data_ty>
            {
                ::std::result::Result::Ok(match __disc {
                    #(#move_arms_e)*
                    _ => return ::std::result::Result::Err(::std::io::Error::new(
                        ::std::io::ErrorKind::InvalidData,
                        "bstack_enum: invalid discriminant",
                    )),
                })
            };
        }
    } else {
        quote!()
    };
    // The failure epilogue (with `__e: io::Error`, `__disc`, `__payload` in scope):
    // reconstruct the consumed `EData` and hand it back, or `lost` if that itself
    // faults.
    let on_fail = if enum_has_owning {
        quote! {
            return match __reconstruct(__disc, __payload) {
                ::std::result::Result::Ok(__hb) =>
                    ::core::result::Result::Err(::bstack_raii::ConstructError::recovered(__e, __hb)),
                ::std::result::Result::Err(_) =>
                    ::core::result::Result::Err(::bstack_raii::ConstructError::lost(__e)),
            };
        }
    } else {
        quote!(return ::std::result::Result::Err(__e);)
    };
    let enum_ret = |ok: &TokenStream| -> TokenStream {
        if enum_has_owning {
            quote!(::core::result::Result<#ok, ::bstack_raii::ConstructError<#data_ty>>)
        } else {
            quote!(::std::io::Result<#ok>)
        }
    };

    let enum_new = match mode {
        Mode::Plain | Mode::Rc => {
            let injected_init = if let Mode::Rc = mode {
                quote!(__bstack_refcount: 1u64,)
            } else {
                quote!()
            };
            let finish = if let Mode::Rc = mode {
                quote! {
                    ::std::result::Result::Ok(unsafe {
                        ::bstack_raii::BStackRc::from_raw(
                            ::bstack_raii::BStackRef::from_range(__data),
                            ::core::option::Option::None,
                            allocator,
                        )
                    })
                }
            } else {
                quote! {
                    ::std::result::Result::Ok(unsafe {
                        ::bstack_raii::BStackOwned::from_raw(
                            unsafe { <Self as ::bstack_raii::BStackBlock>::from_range(__data) },
                        )
                    })
                }
            };
            let ret_ty = enum_ret(&new_ret);
            quote! {
                #vis fn new<'__e, __A: ::bstack_raii::BStackRaiiAllocator>(
                    allocator: &'__e __A,
                    data: #data_ty,
                ) -> #ret_ty {
                    #reconstruct_decl
                    #embed_decl
                    let (__disc, __payload): (#disc_ty, [u8; #payload_const]) = match data {
                        #(#new_arms)*
                    };
                    let __on_disk = #on_disk {
                        #enum_header
                        #injected_init
                        __bstack_disc: __disc,
                        __bstack_payload: __payload,
                    };
                    let mut __slice = match allocator.alloc(#enum_size) {
                        ::std::result::Result::Ok(__s) => __s,
                        // Hand the consumed value back.
                        ::std::result::Result::Err(__e) => { #on_fail }
                    };
                    let __data = __slice.as_range();
                    if let ::std::result::Result::Err(__e) =
                        __slice.write_range(0, ::bstack_raii::bytemuck::bytes_of(&__on_disk))
                    {
                        let _ = allocator.dealloc(__slice);
                        // Hand the consumed value back.
                        #on_fail
                    }
                    #embed_post
                    #finish
                }
            }
        }
        Mode::RcWeak => {
            let ctrl_size = quote! {
                ::core::mem::size_of::<<Self as ::bstack_raii::BStackWeakable>::Control>() as u64
            };
            let ret_ty = enum_ret(&quote!(::bstack_raii::BStackRc<'__e, Self, __A>));
            quote! {
                #vis fn new<'__e, __A: ::bstack_raii::BStackRaiiAllocator>(
                    allocator: &'__e __A,
                    data: #data_ty,
                ) -> #ret_ty {
                    #reconstruct_decl
                    #embed_decl
                    let (__disc, __payload): (#disc_ty, [u8; #payload_const]) = match data {
                        #(#new_arms)*
                    };
                    let __blocks = match ::bstack_raii::BStackRaiiAllocator::alloc_many(allocator, &[#enum_size, #ctrl_size]) {
                        ::std::result::Result::Ok(__b) => __b,
                        // Hand the consumed value back.
                        ::std::result::Result::Err(__e) => { #on_fail }
                    };
                    let __data = __blocks[0];
                    let __ctrl = __blocks[1];
                    let __on_disk = #on_disk {
                        #enum_header
                        __bstack_ctrl: __ctrl.start(),
                        __bstack_disc: __disc,
                        __bstack_payload: __payload,
                    };
                    let __ctrl_payload = ::bstack_raii::__private::build_control_payload(
                        #ctrl_eightcc,
                        __data.start(),
                        #ctrl_size,
                    );
                    let __writes: [(u64, ::std::vec::Vec<u8>); 2] = [
                        (
                            __data.start(),
                            ::bstack_raii::bytemuck::bytes_of(&__on_disk).to_vec(),
                        ),
                        (__ctrl.start(), __ctrl_payload),
                    ];
                    if let ::std::result::Result::Err(__e) =
                        allocator.stack().set_batched(__writes)
                    {
                        // SAFETY (emitted): `__data` / `__ctrl` are the two blocks this
                        // constructor just allocated and failed to initialize.
                        let _ = unsafe { ::bstack_raii::BStackRaiiAllocator::free_many(allocator, [__data, __ctrl]) };
                        // Hand the consumed value back.
                        #on_fail
                    }
                    #embed_post
                    ::std::result::Result::Ok(unsafe {
                        ::bstack_raii::BStackRc::from_raw(
                            ::bstack_raii::BStackRef::from_range(__data),
                            ::core::option::Option::Some(__ctrl),
                            allocator,
                        )
                    })
                }
            }
        }
    };
    let shared_impl = shared_impl(mode, name);
    // A plain enum is self-contained, so it may be `#[embed]`ded; an `rc` enum is not.
    let embeddable_impl = if mode == Mode::Plain {
        quote! {
            impl #enum_impl_g ::bstack_raii::__private::BStackEmbeddable for #name #enum_ty_g #enum_where {}
        }
    } else {
        quote!()
    };
    let weakable_items = weakable_items(mode, name, &control, &ctrl_eightcc, vis);

    let allow_deprecated = input.attrs.iter().any(is_allow_deprecated);
    let ctrl_truncated = mode == Mode::RcWeak && ctrl_truncated;
    let overlong_warning = if (tag.truncated || ctrl_truncated)
        && !(attr.allow_overlong || allow_deprecated)
    {
        let warn_fn = format_ident!("__bstack_tag_overlong_{}", name);
        let msg = format!(
            "#[bstack_enum] on `{type_name}`: a tag override longer than 8 bytes was truncated; \
             add `allow(overlong_tag)` to silence"
        );
        quote! {
            #[doc(hidden)]
            #[allow(dead_code, non_snake_case)]
            fn #warn_fn() {
                #[deprecated(note = #msg)]
                fn overlong_tag() {}
                overlong_tag();
            }
        }
    } else {
        quote!()
    };

    // `read` / `bstack_drop` only need the payload bytes when a variant carries one.
    let read_payload = if needs_payload {
        quote!(let __pl = __od.__bstack_payload;)
    } else {
        quote!()
    };
    // Body of `__bstack_drop_children(__range, allocator)`: free the active
    // variant's owned child (if any). `__range` is this block's range (its own,
    // or — when `#[embed]`ded — its slot in the parent).
    let drop_children_body = if drop_arms.is_empty() {
        quote!()
    } else {
        quote! {
            let __stack = allocator.stack();
            let mut __buf = [0u8; ::core::mem::size_of::<#on_disk>()];
            let __r = unsafe { ::bstack_raii::BStackRef::<Self>::from_range(__range) };
            let __od: #on_disk = *__r.read_on_disk(__stack, &mut __buf)?;
            let __disc = __od.__bstack_disc;
            let __pl = __od.__bstack_payload;
            match __disc {
                #(#drop_arms)*
                // A known discriminant whose variant owns nothing: nothing to free.
                #(#disc_pats)|* => {}
                // An unknown discriminant is corruption. `read` and `bstack_move!`
                // already reject it; silently returning `Ok` here would report a
                // successful teardown that freed nothing (a hidden leak).
                _ => {
                    return ::std::result::Result::Err(::std::io::Error::new(
                        ::std::io::ErrorKind::InvalidData,
                        "bstack_enum: invalid discriminant",
                    ));
                }
            }
        }
    };

    // Body of `__bstack_clone_into`: real deep clone for a plain enum, a runtime
    // error for `rc` / `rc, weak` (its injected refcount / control block would
    // need re-initialization). The whole OnDisk is byte-copied; the active
    // variant's payload is then fixed up (owned → deep clone + repoint, strong /
    // weak → refcount bump, ref → alias, embed → not yet supported).
    let (clone_children_body, clone_into_body) = if mode != Mode::Plain {
        let err = quote! {
            ::std::result::Result::Err(::std::io::Error::new(
                ::std::io::ErrorKind::Unsupported,
                "TryCloneIn: a reference-counted (`rc` / `rc, weak`) enum block is shared, \
                 not deep-cloned — duplicate its handle with `BStackRc::try_clone` \
                 (see the `TryClone` trait)",
            ))
        };
        (err.clone(), err)
    } else {
        let dispatch = if clone_arms.is_empty() {
            quote!()
        } else {
            quote! {
                let __disc = __od.__bstack_disc;
                let mut __pl = __od.__bstack_payload;
                match __disc {
                    #(#clone_arms)*
                    // A known discriminant whose variant owns nothing: the verbatim
                    // byte copy is the correct clone.
                    #(#disc_pats)|* => {}
                    // An unknown discriminant must not be byte-copied: whatever the
                    // payload slot holds would be copied un-repointed, so the clone
                    // and the original would name one child — two owners.
                    _ => {
                        return ::std::result::Result::Err(::std::io::Error::new(
                            ::std::io::ErrorKind::InvalidData,
                            "bstack_enum: invalid discriminant",
                        ));
                    }
                }
                __od.__bstack_payload = __pl;
            }
        };
        let children = quote! {
            let __stack = allocator.stack();
            let __src = self.0;
            let mut __buf = [0u8; ::core::mem::size_of::<#on_disk>()];
            let __r = unsafe { ::bstack_raii::BStackRef::<Self>::from_range(__src) };
            #[allow(unused_mut)]
            let mut __od: #on_disk = *__r.read_on_disk(__stack, &mut __buf)?;
            #dispatch
            ::std::result::Result::Ok(__od)
        };
        let into = quote! {
            let __od = self.__bstack_clone_children_inplace(allocator, __plan)?;
            let __dst = __plan.alloc_raw(allocator, ::core::mem::size_of::<#on_disk>() as u64)?;
            __plan.write(__dst.start(), ::bstack_raii::bytemuck::bytes_of(&__od).to_vec());
            ::std::result::Result::Ok(__dst)
        };
        (children, into)
    };
    // The public `TryCloneIn` entry point, for plain enums only.
    let enum_clone_trait = if mode == Mode::Plain {
        quote! {
            impl #enum_impl_g ::bstack_raii::TryCloneIn for #name #enum_ty_g #enum_where {
                fn try_clone_in<__A: ::bstack_raii::BStackRaiiAllocator>(
                    &self,
                    allocator: &__A,
                ) -> ::std::io::Result<::bstack_raii::BStackOwned<Self>> {
                    use ::bstack_raii::BStackBlock as _;
                    // The clone strategy (single-pass intention-first, or two-pass
                    // atomic bulk on a `BStackBulkAllocator`) is chosen inside
                    // `run_clone`, which may run this descent twice (measure + build).
                    let __dst = ::bstack_raii::ClonePlan::run_clone(allocator, |__plan| {
                        self.__bstack_clone_into(allocator, __plan)
                    })?;
                    ::std::result::Result::Ok(unsafe {
                        ::bstack_raii::BStackOwned::from_raw(
                            unsafe { <Self as ::bstack_raii::BStackBlock>::from_range(__dst) },
                        )
                    })
                }
            }
        }
    } else {
        quote!()
    };

    // `EData` is generic over `<'e, A>` when a variant holds a strong/weak
    // reference; `EView` only when a weak variant makes `read` upgrade; both are
    // also generic over the enum's own type parameters.
    let data_generics = comp_decl(&quote!('__e), has_shared || has_foreign, has_shared);
    let view_generics = comp_decl(&quote!('__e), has_weak || has_foreign, has_weak);
    let view_ty = comp_ty(&view, &quote!('__e), has_weak || has_foreign, has_weak);
    // `bstack_move!` yields the same `EData` (owned handles); `Fields` just names
    // it with the move lifetime.
    let move_fields_ty = comp_ty(&data, &quote!('__mv), has_shared || has_foreign, has_shared);
    // `bstack_move!` frees the enum shell, then rebuilds the active variant's
    // payload as an owned handle.
    let move_payload = if needs_payload {
        quote!(let __pl = __od.__bstack_payload;)
    } else {
        quote!()
    };

    // Whole-value `#[bstack_mut]` on the enum itself: `set` when no variant owns
    // anything (a wholesale overwrite is safe), else `replace` (hand the old value
    // out). Both are one crash-atomic `set` of the same-size record. Only for a
    // plain enum (an `rc` / `rc, weak` block's injected refcount / control must not
    // be clobbered) and not with `#[embed]` variants (their post-write copy / re-home
    // makes in-place replacement a separate problem).
    let enum_mut_methods = if is_bstack_mut(&input.attrs) {
        if mode != Mode::Plain {
            return Err(Error::new_spanned(
                &input.ident,
                "[BSTACK0604] #[bstack_mut] on a `#[bstack_enum]` is only supported for a plain enum — a \
                 shared (`rc` / `rc, weak`) enum's refcount / control block can't be \
                 overwritten in place; rebuild the value instead",
            ));
        }
        if enum_has_embed {
            return Err(Error::new_spanned(
                &input.ident,
                "[BSTACK0605] #[bstack_mut] is not yet supported on a `#[bstack_enum]` with an `#[embed]` \
                 variant",
            ));
        }
        let build_image = quote! {
            let (__disc, __payload): (#disc_ty, [u8; #payload_const]) = match data {
                #(#new_arms)*
            };
            let __on_disk = #on_disk {
                #enum_header
                __bstack_disc: __disc,
                __bstack_payload: __payload,
            };
        };
        if drop_arms.is_empty() {
            // No variant owns anything (pure POD / ref / foreign-ref): a wholesale
            // overwrite frees nothing and needs no hand-back.
            quote! {
                /// Overwrite the whole enum value (variant + payload) in place, as
                /// one crash-atomic `set`. Available because no variant owns
                /// anything, so nothing is stranded.
                #[allow(unused_variables)]
                #vis fn set<'__e, __A: ::bstack_raii::BStackRaiiAllocator>(
                    &self,
                    allocator: &'__e __A,
                    data: #data_ty,
                ) -> ::std::io::Result<()> {
                    #build_image
                    allocator
                        .stack()
                        .set(self.0.start(), ::bstack_raii::bytemuck::bytes_of(&__on_disk))
                }
            }
        } else {
            // Some variant owns children: install the new value and move the old
            // one out (`bstack_move!`'s per-variant reconstruction), upholding the
            // `ReplaceError` "never lose the new value" contract.
            quote! {
                /// Replace the whole enum value (variant + payload), moving the old
                /// value out. Thread-safe: the read of the old record and the write
                /// of the new happen together as one [`BStack::swap`], so concurrent
                /// callers each take the distinct value they displaced — an old
                /// variant's owned children are never double-owned.
                /// On I/O failure the *new* value is handed back through
                /// [`ReplaceError`](::bstack_raii::ReplaceError).
                #[allow(unused_variables)]
                #vis fn replace<'__e, __A: ::bstack_raii::BStackRaiiAllocator>(
                    &self,
                    allocator: &'__e __A,
                    data: #data_ty,
                ) -> ::core::result::Result<#data_ty, ::bstack_raii::ReplaceError<#data_ty>> {
                    let __alloc = allocator;
                    // Reconstruct an owned `EData` from a (disc, payload) pair — the
                    // same per-variant logic as `bstack_move!`, transferring children
                    // out (nothing freed).
                    let __reconstruct = |__disc: #disc_ty, __pl: [u8; #payload_const]|
                        -> ::std::io::Result<#data_ty>
                    {
                        ::std::result::Result::Ok(match __disc {
                            #(#move_arms)*
                            _ => return ::std::result::Result::Err(::std::io::Error::new(
                                ::std::io::ErrorKind::InvalidData,
                                "bstack_enum: invalid discriminant",
                            )),
                        })
                    };
                    // 1. Consume the new value into its on-disk image.
                    #build_image
                    // 2. Atomic exchange: install the new record and take the old one
                    //    in one locked step (same-size record).
                    let __old_bytes = match allocator
                        .stack()
                        .swap(self.0.start(), ::bstack_raii::bytemuck::bytes_of(&__on_disk))
                    {
                        ::std::result::Result::Ok(__b) => __b,
                        ::std::result::Result::Err(__e) => {
                            return ::core::result::Result::Err(
                                match __reconstruct(__disc, __payload) {
                                    ::std::result::Result::Ok(__hb) =>
                                        ::bstack_raii::ReplaceError::recovered(__e, __hb),
                                    ::std::result::Result::Err(_) =>
                                        ::bstack_raii::ReplaceError::lost(__e),
                                },
                            );
                        }
                    };
                    // 3. Decode the displaced record and reconstruct the old value.
                    let __old_od: #on_disk =
                        ::bstack_raii::bytemuck::pod_read_unaligned(&__old_bytes);
                    match __reconstruct(__old_od.__bstack_disc, __old_od.__bstack_payload) {
                        ::std::result::Result::Ok(__old) => ::core::result::Result::Ok(__old),
                        // The only fallible reconstruction is a `#[bstack_strong]`
                        // variant's `strong_parts` (a control-block read); the new value
                        // is already installed, but the old variant's strong child block
                        // is untouched at a known offset. Hand it back as a raw range so
                        // it stays recoverable (retry `strong_parts` once I/O recovers, or
                        // reclaim it) rather than leaking.
                        ::std::result::Result::Err(__e) => {
                            let __pl = __old_od.__bstack_payload;
                            let __raw: ::std::vec::Vec<::bstack_raii::BStackRange> =
                                match __old_od.__bstack_disc {
                                    #(#raw_arms)*
                                    _ => ::std::vec![],
                                };
                            ::core::result::Result::Err(
                                ::bstack_raii::ReplaceError::lost_raw(__e, __raw))
                        }
                    }
                }
            }
        }
    } else {
        quote!()
    };

    let enum_handle_def = if type_params.is_empty() {
        quote! {
            #[derive(::core::clone::Clone, ::core::marker::Copy)]
            #vis struct #name(::bstack_raii::BStackRange);
        }
    } else {
        quote! {
            #vis struct #name #enum_decl_g(
                ::bstack_raii::BStackRange #enum_phantom_field
            ) #enum_decl_where;
            impl #enum_decl_g ::core::clone::Clone for #name #enum_decl_ty_g #enum_decl_where {
                fn clone(&self) -> Self { *self }
            }
            impl #enum_decl_g ::core::marker::Copy for #name #enum_decl_ty_g #enum_decl_where {}
        }
    };
    // The payload area size const (max over all variants), folded at const-eval.
    let payload_const_def = layout::payload_const_def(vis, &payload_const, &payload_sizes);
    Ok(quote! {
        #enum_handle_def

        #payload_const_def

        impl #enum_impl_g #name #enum_ty_g #enum_where {
            /// The payload area size (bytes) — the max over all variants.
            #[doc(hidden)]
            pub const __PAYLOAD: usize = #payload_const;
        }

        #[repr(C, packed)]
        #[derive(::core::clone::Clone, ::core::marker::Copy)]
        #vis struct #on_disk {
            __bstack_header: ::bstack_raii::BlockHeader,
            #injected_ondisk
            __bstack_disc: #disc_ty,
            __bstack_payload: [u8; #payload_const],
        }
        unsafe impl ::bstack_raii::Zeroable for #on_disk {}
        unsafe impl ::bstack_raii::Pod for #on_disk {}

        const _: fn() = || {
            fn __assert_pod<__T: ::bstack_raii::Pod>() {}
            #( __assert_pod::<#pod_types>(); )*
        };

        const _: fn() = || {
            fn __assert_embeddable<__T: ::bstack_raii::__private::BStackEmbeddable>() {}
            #( __assert_embeddable::<#embed_types>(); )*
        };

        /// The in-memory owned form of the enum's payload — POD by value and
        /// each child/reference as an owned handle (owned → `BStackOwned`,
        /// strong → `BStackRc`, weak → `BStackWeak`, ref → `BStackRef`).
        ///
        /// The *same* type is passed to [`new`](#name::new) to construct a variant
        /// and returned by `bstack_move!` to destructure one (they are duals).
        #vis enum #data #data_generics {
            #(#data_variants)*
        }

        /// The result of [`read`](#name::read): the current variant, with POD
        /// values by value, owned/ref children as borrowed handles, and a weak
        /// variant upgraded to `Option<BStackRc>`.
        #vis enum #view #view_generics {
            #(#view_variants)*
        }

        impl #enum_impl_g ::bstack_raii::BStackCast for #name #enum_ty_g #enum_where {
            fn eightcc() -> ::bstack_raii::EightCC {
                #eightcc
            }
        }

        impl #enum_impl_g ::bstack_raii::BStackBlock for #name #enum_ty_g #enum_where {
            type OnDisk = #on_disk;
            unsafe fn from_range(range: ::bstack_raii::BStackRange) -> Self {
                #name(range #enum_phantom_ctor)
            }
            fn range(&self) -> ::bstack_raii::BStackRange {
                self.0
            }

            /// Read this enum's OnDisk and return a deep-cloned copy: the active
            /// variant's payload fixed up (owned child cloned into `__plan`,
            /// strong/weak bumped, ref aliased, embedded child folded in place),
            /// without allocating a block for `self`. Overrides the childless
            /// `BStackBlock` default. Used to fold an `#[embed]`ded enum inline,
            /// and by `__bstack_clone_into`.
            #[doc(hidden)]
            #[allow(unused_variables, unused_imports)]
            fn __bstack_clone_children_inplace<__A: ::bstack_raii::BStackRaiiAllocator>(
                &self,
                allocator: &__A,
                __plan: &mut ::bstack_raii::ClonePlan,
            ) -> ::std::io::Result<#on_disk> {
                use ::bstack_raii::BStackBlock as _;
                #clone_children_body
            }

            /// Deep-clone this enum into a `ClonePlan`: allocate a fresh block and
            /// stage its fixed-up payload. Returns the new block's range. Also lets
            /// an owned enum child of a struct be recursed into.
            #[doc(hidden)]
            #[allow(unused_variables)]
            fn __bstack_clone_into<__A: ::bstack_raii::BStackRaiiAllocator>(
                &self,
                allocator: &__A,
                __plan: &mut ::bstack_raii::ClonePlan,
            ) -> ::std::io::Result<::bstack_raii::BStackRange> {
                #clone_into_body
            }

            /// Free the active variant's owned child (recursively) given this
            /// block's range, **without** freeing the block itself — used when the
            /// enum is `#[embed]`ded, and by `bstack_drop`. Overrides the childless
            /// `BStackBlock` default.
            #[doc(hidden)]
            #[allow(unused_imports)]
            fn __bstack_drop_children<__A: ::bstack_raii::BStackRaiiAllocator>(
                __range: ::bstack_raii::BStackRange,
                allocator: &__A,
            ) -> ::std::io::Result<()> {
                use ::bstack_raii::BStackDrop as _;
                use ::bstack_raii::BStackBlock as _;
                #drop_children_body
                ::std::result::Result::Ok(())
            }
        }

        // The enum handle is a pure view: no `BStackDrop`. Freeing goes through an
        // affine owner (see `teardown::drop_block`).

        impl #enum_impl_g #name #enum_ty_g #enum_where {
            /// Allocate a new enum block holding `data`'s variant + payload.
            #enum_new

            /// Read the current variant. Takes the allocator (a weak variant's
            /// read upgrades through it; other variants just read the block).
            #vis fn read<'__e, __A: ::bstack_raii::BStackRaiiAllocator>(
                &self,
                allocator: &'__e __A,
            ) -> ::std::io::Result<#view_ty> {
                let mut __buf = [0u8; ::core::mem::size_of::<#on_disk>()];
                let __r = unsafe { ::bstack_raii::BStackRef::<Self>::from_range(self.0) };
                let __od: #on_disk = *__r.read_on_disk(allocator.stack(), &mut __buf)?;
                let __disc = __od.__bstack_disc;
                #read_payload
                ::std::result::Result::Ok(match __disc {
                    #(#read_arms)*
                    _ => {
                        return ::std::result::Result::Err(::std::io::Error::new(
                            ::std::io::ErrorKind::InvalidData,
                            "bstack_enum: invalid discriminant",
                        ));
                    }
                })
            }

            /// Borrow this block as an untyped slice (infallible upcast).
            #vis fn as_slice<'__s>(
                &self,
                stack: &'__s ::bstack_raii::BStack,
            ) -> ::bstack_raii::BStackSlice<'__s> {
                unsafe {
                    ::bstack_raii::BStackSlice::from_raw_range(
                        stack,
                        ::bstack_raii::BStackBlock::range(self),
                    )
                }
            }

            #enum_mut_methods
        }

        #shared_impl
        #embeddable_impl
        #weakable_items

        impl #enum_impl_g ::bstack_raii::BStackMove for #name #enum_ty_g #enum_where {
            type Fields<'__mv, __A: ::bstack_raii::BStackRaiiAllocator> = #move_fields_ty;
            fn bstack_move<'__mv, __A: ::bstack_raii::BStackRaiiAllocator>(
                owned: ::bstack_raii::BStackOwned<Self>,
                __alloc: &'__mv __A,
            ) -> ::std::io::Result<Self::Fields<'__mv, __A>> {
                let __inner = owned.into_inner();
                let __range = ::bstack_raii::BStackBlock::range(&__inner);
                let mut __buf = [0u8; ::core::mem::size_of::<#on_disk>()];
                let __r = unsafe { ::bstack_raii::BStackRef::<Self>::from_range(__range) };
                let __od: #on_disk = *__r.read_on_disk(__alloc.stack(), &mut __buf)?;
                let __disc = __od.__bstack_disc;
                #move_payload
                let __result = match __disc {
                    #(#move_arms)*
                    _ => {
                        return ::std::result::Result::Err(::std::io::Error::new(
                            ::std::io::ErrorKind::InvalidData,
                            "bstack_enum: invalid discriminant",
                        ));
                    }
                };
                // Free the enum shell only; the moved-out payload stays live.
                unsafe { ::bstack_raii::dealloc_range(__alloc, __range)?; }
                ::std::result::Result::Ok(__result)
            }
        }

        #enum_clone_trait
        #overlong_warning
    })
}
