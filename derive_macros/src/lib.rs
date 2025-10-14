use core::panic;
use std::collections::HashSet;

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Attribute, Data, DeriveInput, Fields, Item, ItemStruct, Meta, MetaList, Token, Type, parse,
    parse_macro_input, punctuated::Punctuated,
};

static DEFAULT_FIELD_NAME_FOR_EDGE_DEST: &str = "dest_node";
static DEFUALT_FIELD_NAME_FOR_EDGE_TYPE: &str = "edge_type";

#[proc_macro_derive(GenericEdge, attributes(generic_edge, edge_dest, edge_type))]
pub fn derive_generic_edge(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    assert_ne!(
        DEFAULT_FIELD_NAME_FOR_EDGE_DEST,
        DEFUALT_FIELD_NAME_FOR_EDGE_TYPE
    );

    let mut input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    // check if #[repr(C)] is present
    let has_repr_c = input.attrs.iter().any(|attr: &Attribute| {
        if attr.path().is_ident("repr") {
            if let Meta::List(MetaList { tokens, .. }) = &attr.meta {
                tokens.to_string().contains("C")
            } else {
                false
            }
        } else {
            false
        }
    });

    if !has_repr_c {
        return syn::Error::new_spanned(
            input.ident,
            "Types deriving `GenericEdge` must be annotated with #[repr(C)]",
        )
        .to_compile_error()
        .into();
    }

    let _data = if let Data::Struct(ds) = input.clone().data {
        ds
    } else {
        return syn::Error::new_spanned(input.ident, "Only structs may derive `GenericEdge`")
            .to_compile_error()
            .into();
    };

    let mut impl_new_for_self = None;
    let mut impl_default_for_self = None;
    let mut quoted_dest_access = None;
    let mut quoted_edge_type_access = None;
    let mut quoted_mut_dest = None;
    let mut quoted_mut_edge_type = None;
    let mut quoted_dest_type = None;
    let mut quoted_edge_type = None;

    // --- Extract info from struct ---
    if let Data::Struct(s) = &input.data {
        match &s.fields {
            // ✅ Named fields (normal struct or bitfield struct with visible fields)
            Fields::Named(fields) => {
                let mut dest_access = None;
                let mut edge_type_access = None;
                let mut dest_access_real = None;
                let mut edge_type_access_real = None;
                let mut dest_setter = None;
                let mut edge_setter = None;
                let mut dest_access_fallback = None;
                let mut edge_type_access_fallback = None;
                let mut dest_access_type_fallback = None;
                let mut edge_type_access_type_fallback = None;

                // search fields for attributes & fallbakcs
                for field in &fields.named {
                    let ident = field.ident.as_ref();
                    let ty = field.ty.clone();

                    // setup fallback if field name matches
                    if let Some(field_name) = ident {
                        if field_name == DEFAULT_FIELD_NAME_FOR_EDGE_DEST {
                            dest_access_fallback = Some(field_name);
                            dest_access_type_fallback = Some(ty);
                        } else if field_name == DEFUALT_FIELD_NAME_FOR_EDGE_TYPE {
                            edge_type_access_fallback = Some(field_name);
                            edge_type_access_type_fallback = Some(ty);
                        }
                    }

                    // Look for #[edge_dest] or #[edge_type] attributes
                    for attr in &field.attrs {
                        if attr.path().is_ident("edge_dest") {
                            let (getter, setter, ty) = parse_generic_edge_attr(attr);
                            dest_access = Some(getter.map_or_else(
                                || {
                                    let ident = match ident {
                                        Some(i) => i,
                                        None => {
                                            return syn::Error::new_spanned(name.clone(), "Struct derives `GenericEdge`, but no getter method or field was found for attribute `edge_dest`")
                                                .to_compile_error();
                                        }
                                    };
                                    quote! {
                                        self.#ident
                                    }
                                },
                                |g| quote! { self.#g() },
                            ));
                            dest_access_real = ident;
                            let type_of_dest = if let Some(t) = ty {
                                quoted_dest_type = Some(quote! {#t});
                                t
                            } else {
                                let t = field.ty.clone();
                                quoted_dest_type = Some(quote! {#t});
                                t
                            };
                            dest_setter = setter.map(|g| quote! { .#g( #type_of_dest::try_from(new_edge_dest).unwrap() ) });
                        }
                        if attr.path().is_ident("edge_type") {
                            let (getter, setter, ty) = parse_generic_edge_attr(attr);
                            edge_type_access = Some(getter.map_or_else(
                                || {
                                    let ident = match ident {
                                        Some(i) => i,
                                        None => {
                                            return syn::Error::new_spanned(name.clone(), "Struct derives `GenericEdge`, but no getter method or field was found for attribute `edge_type`")
                                                .to_compile_error();
                                        }
                                    };
                                    quote! {
                                        self.#ident
                                    }
                                },
                                |g| quote! { self.#g() },
                            ));
                            edge_type_access_real = ident;
                            let type_of_edge_type = if let Some(t) = ty {
                                quoted_edge_type = Some(quote! {#t});
                                t
                            } else {
                                let t = field.ty.clone();
                                quoted_edge_type = Some(quote! {#t});
                                t
                            };
                            edge_setter = setter
                                .map(|g| quote! { .#g( #type_of_edge_type::from(new_edge_type) ) });
                        }
                    }
                }

                // setup getters
                if dest_access.is_none() {
                    match (dest_access_fallback, dest_access_type_fallback) {
                        (Some(field), Some(field_ty)) => {
                            quoted_dest_type = Some(quote! { #field_ty });
                            quoted_dest_access = Some(quote! {self.#field});
                        }
                        _ => {
                            return syn::Error::new_spanned(
                                name.clone(),
                                format!(
                                    "Missing #[edge_dest] attribute or fallback field `{}`",
                                    DEFAULT_FIELD_NAME_FOR_EDGE_DEST
                                ),
                            )
                            .to_compile_error()
                            .into();
                        }
                    }
                } else {
                    quoted_dest_access = dest_access;
                }
                if edge_type_access.is_none() {
                    match (edge_type_access_fallback, edge_type_access_type_fallback) {
                        (Some(field), Some(field_ty)) => {
                            quoted_edge_type = Some(quote! { #field_ty });
                            quoted_edge_type_access = Some(quote! {self.#field});
                        }
                        _ => {
                            return syn::Error::new_spanned(
                                name.clone(),
                                format!(
                                    "Missing #[edge_type] attribute or fallback field `{}`",
                                    DEFUALT_FIELD_NAME_FOR_EDGE_TYPE
                                ),
                            )
                            .to_compile_error()
                            .into();
                        }
                    }
                } else {
                    quoted_edge_type_access = edge_type_access;
                }

                // setup setters, new() and default()
                let dest_ty = if let Some(ty) = quoted_dest_type.clone() {
                    ty
                } else {
                    panic!("can't proceed as edge_type clone failed")
                };
                let edge_ty = if let Some(ty) = quoted_edge_type.clone() {
                    ty
                } else {
                    panic!("can't proceed as edge_type clone failed")
                };
                // if named struct decide how to  implement method new(edge_dest: u64, edge_type: u64)
                match (
                    dest_setter,
                    edge_setter,
                    dest_access_real,
                    edge_type_access_real,
                    dest_access_fallback,
                    edge_type_access_fallback,
                ) {
                    (Some(d_s), Some(e_s), a, b, c, d) => {
                        // // interpolation d_s
                        // quote! { .#g( #type_of_edge_type::from(edge_type) ) }
                        quoted_mut_dest = Some(quote! { self #d_s; self });
                        quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        match (a, b, c, d) {
                            (Some(d_r), Some(e_r), _, _) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap() , #e_r : #edge_ty::from(edge_type) } },
                                );
                            }
                            (None, Some(e_r), Some(d_f), _) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap() , #e_r : #edge_ty::from(edge_type) } },
                                );
                            }
                            (Some(d_r), _, _, Some(e_f)) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap() , #e_f : #edge_ty::from(edge_type) } },
                                );
                            }
                            (None, None, Some(d_f), Some(e_f)) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap() , #e_f : #edge_ty::from(edge_type) } },
                                );
                            }
                            _ => {
                                return syn::Error::new_spanned(
                                    name.clone(),
                                    "Named struct specified setter, but necessary fields (and their respective fallbacks) for default initialization"
                                    )
                                    .to_compile_error()
                                    .into();
                            }
                        };
                    }
                    (d_s, e_s, Some(d_r), Some(e_r), _, _) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_r = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_r = #edge_ty::from(new_edge_type); self });
                        }

                        impl_default_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                        );

                        impl_new_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap() , #e_r : #edge_ty::from(edge_type) } },
                        );
                    }
                    (d_s, e_s, None, None, Some(d_f), Some(e_f)) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_f = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_f = #edge_ty::from(new_edge_type); self });
                        }
                        impl_default_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                        );
                        impl_new_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap(), #e_f : #edge_ty::from(edge_type) } },
                        );
                    }
                    (d_s, e_s, Some(d_r), None, _, Some(e_f)) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_r = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_f = #edge_ty::from(new_edge_type); self });
                        }
                        impl_default_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                        );
                        impl_new_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap(), #e_f : #edge_ty::from(edge_type) } },
                        );
                    }
                    // interpolation e_s
                    (d_s, e_s, None, Some(e_r), Some(d_f), _) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_f = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_r = #edge_ty::from(new_edge_type); self });
                        }
                        impl_default_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                        );
                        impl_new_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap(), #e_r : #edge_ty::from(edge_type) } },
                        );
                    }
                    (_a, _b, _c, _d, _e, _f) => {
                        return syn::Error::new_spanned(
                        name.clone(),
                            "Named struct must specify #[edge_type(getter=..., setter=..., real_type=...)], but setter was missing and no fallback initialization method was found"
                    )
                    .to_compile_error()
                    .into();
                    }
                };
            }

            // Tuple structs: Either attributes describing the methods are provided or assume (dest, edge_type)
            Fields::Unnamed(fields) => {
                // bitfield mode: Look for struct-level attributes
                if fields.unnamed.len() == 1 {
                    let dest_attr = input
                        .attrs
                        .iter()
                        .find(|a| a.path().is_ident("edge_dest"))
                        .expect(
                            "Bitfield tuple struct must specify #[edge_dest(getter=..., setter=..., real_type=...)]",
                        );
                    let edge_attr = input
                        .attrs
                        .iter()
                        .find(|a| a.path().is_ident("edge_type"))
                        .expect(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)]",
                        );

                    let (dest_getter, d_setter, dest_ty) = parse_generic_edge_attr(dest_attr);
                    let (edge_getter, e_setter, edge_ty) = parse_generic_edge_attr(edge_attr);

                    if dest_getter.is_none() || edge_getter.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)], getter was missing"
                        );
                    }
                    if d_setter.is_none() || e_setter.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)], setter was missing"
                        );
                    }
                    if dest_ty.is_none() || edge_ty.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)], real_type was missing"
                        );
                    }

                    if let Some(d_s) = d_setter {
                        quoted_mut_dest = Some(quote! { self.#d_s( new_edge_dest ); self });
                    }
                    if let Some(e_s) = e_setter {
                        quoted_mut_edge_type = Some(quote! { self.#e_s( new_edge_type ); self });
                    }
                    if let Some(type_of_dest) = dest_ty {
                        quoted_dest_type = Some(quote! {#type_of_dest});
                    }
                    if let Some(type_of_edge_type) = edge_ty {
                        quoted_edge_type = Some(quote! {#type_of_edge_type});
                    }
                    quoted_dest_access = dest_getter.map(|e| quote! { self.#e() });
                    quoted_edge_type_access = edge_getter.map(|e| quote! { self.#e() });
                    impl_default_for_self = Some(quote! { #name::new(0u64, 0u64)});
                    impl_new_for_self = Some(quote! { #name::new(edge_type, edge_dest)});
                } else if fields.unnamed.len() == 2 {
                    let dest_attr = input.attrs.iter().find(|a| a.path().is_ident("edge_dest"));
                    let edge_attr = input.attrs.iter().find(|a| a.path().is_ident("edge_type"));
                    if let Some(_attr) = dest_attr {
                        // FIXME: search for getters and setters
                    }
                    if let Some(_attr) = edge_attr {
                        // FIXME: search for getters and setters
                    }
                    quoted_dest_access = Some(quote! { self.0 });
                    quoted_edge_type_access = Some(quote! { self.1 });
                    let type_of_dest = fields.unnamed[0].ty.clone();
                    let type_of_edge_type = fields.unnamed[1].ty.clone();
                    quoted_dest_type = Some(type_of_dest.clone()).map(|ty| quote! {#ty});
                    quoted_edge_type = Some(type_of_edge_type.clone()).map(|ty| quote! {#ty});
                    quoted_mut_dest = Some(
                        quote! { self.0 = #type_of_dest::try_from(new_edge_dest).unwrap(); self },
                    );
                    quoted_mut_edge_type =
                        Some(quote! { self.1 = #type_of_edge_type::from(new_edge_type); self });
                    // dest node is a primitive field edge type not (necessarily or not always?)
                    impl_default_for_self = Some(
                        quote! { #name (#type_of_dest::try_from(0u64).unwrap(), #type_of_edge_type::from(0u64))},
                    );
                    impl_new_for_self = Some(
                        quote! { #name (#type_of_dest::try_from(edge_dest).unwrap(), #type_of_edge_type::from(edge_type)) },
                    );
                } else {
                    return syn::Error::new_spanned(
                        name.clone(),
                        "Unnamed structs are only supported for GenericEdge with 1 <= len() <= 2",
                    )
                    .to_compile_error()
                    .into();
                }
            }

            // ❌ Unit structs not supported
            Fields::Unit => {
                return syn::Error::new_spanned(
                    name.clone(),
                    "Unit structs are not supported for GenericEdge",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // parse struct attributes amd override new() and/or default() with user specified
    // methods if given
    // parse #[generic_edge(constructor = "new", deafult = "default")]
    let mut already_derived = Vec::new();
    for attr in &input.attrs {
        // collect already derived macros
        if attr.path().is_ident("derive") {
            if let Meta::List(meta) = &attr.meta {
                let parsed: syn::punctuated::Punctuated<Meta, syn::Token![,]> =
                    match meta.parse_args_with(syn::punctuated::Punctuated::parse_terminated) {
                        Ok(i) => i,
                        Err(e) => panic!("error parsinf token metadata: {}", e),
                    };
                for nested in parsed {
                    if let Meta::Path(path) = nested {
                        if let Some(ident) = path.get_ident() {
                            already_derived.push(ident.to_string());
                        }
                    }
                }
            }
        }
        if attr.path().is_ident("generic_edge") {
            if let Meta::List(meta_list) = &attr.meta {
                let parsed: syn::punctuated::Punctuated<Meta, syn::Token![,]> = match meta_list
                    .parse_args_with(syn::punctuated::Punctuated::parse_terminated)
                {
                    Ok(i) => i,
                    Err(e) => panic!("error parsinf token metadata: {}", e),
                };

                for meta in parsed {
                    if let Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("build")
                            || nv.path.is_ident("builder")
                            || nv.path.is_ident("constructor")
                            || nv.path.is_ident("new")
                        {
                            if let syn::Expr::Lit(expr_lit) = nv.value {
                                if let syn::Lit::Str(lit_str) = expr_lit.lit {
                                    let constructor =
                                        syn::Ident::new(&lit_str.value(), lit_str.span());
                                    // override any previous implementation
                                    impl_new_for_self =
                                        Some(quote! {#name::#constructor(edge_dest, edge_type) });
                                }
                            }
                        } else if nv.path.is_ident("default") || nv.path.is_ident("from_void") {
                            if let syn::Expr::Lit(expr_lit) = nv.value {
                                if let syn::Lit::Str(lit_str) = expr_lit.lit {
                                    let default = syn::Ident::new(&lit_str.value(), lit_str.span());
                                    // override any previous implementation
                                    impl_default_for_self = Some(quote! {#name::#default() });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ensure accesses were found if not search for fallbacks if not error out
    let quoted_dest_access = quoted_dest_access.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Missing #[edge_dest] attribute".to_string())
            .to_compile_error()
    });

    let quoted_edge_type_access = quoted_edge_type_access.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Missing #[edge_type] attribute".to_string())
            .to_compile_error()
    });

    let impl_new_for_self = impl_new_for_self.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Couldn't build constructor for type")
            .to_compile_error()
    });

    let impl_default_for_self = impl_default_for_self.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Couldn't build default for type").to_compile_error()
    });

    // ensure necessary macros are derived (we want PartialEq, Eq & Hash) so the user may use
    // HashMaps and HashSets with any GenericEdge safely
    // FIXME: How to make this work without defining the struct multiple times?
    let required_traits = ["PartialEq", "Eq", "Hash"];
    let mut missing_derives = Vec::new();

    for required in &required_traits {
        if !already_derived.iter().any(|d| d == required) {
            missing_derives.push(syn::Ident::new(required, proc_macro2::Span::call_site()));
        }
    }
    // if missing derives exist, add them
    if !missing_derives.is_empty() {
        let derive_attr: Attribute = syn::parse_quote! {
            #[derive( #( #missing_derives ),* )]
        };
        input.attrs.push(derive_attr);
    }

    let static_type_assertions = quote! {
        const _: () = {
            struct _AssertDestTraits<
                T:
                    std::marker::Copy +
                    std::clone::Clone +
                    std::convert::TryFrom<u64> +
                    std::convert::TryFrom<usize> +
                    std::default::Default +
                    std::cmp::PartialEq +
                    std::cmp::Eq +
                    std::hash::Hash +
                    std::fmt::Debug +
                    std::fmt::Display +
                    std::cmp::PartialOrd +
                    std::cmp::Ord +
                    std::hash::Hash +
                    bytemuck::Pod +
                    bytemuck::Zeroable
                >(::core::marker::PhantomData<T>);
            struct _AssertEdgeTraits<T: GenericEdgeType>(::core::marker::PhantomData<T>);
            let _ = _AssertDestTraits::<#quoted_dest_type>(::core::marker::PhantomData);
            let _ = _AssertEdgeTraits::<#quoted_edge_type>(::core::marker::PhantomData);
        };
    };
    let expanded = quote! {

        #static_type_assertions

        impl std::marker::Copy for #name {}

        impl std::clone::Clone for #name {
            fn clone(&self) -> Self {
                *self
            }
        }

        unsafe impl bytemuck::Zeroable for #name {}

        unsafe impl bytemuck::Pod for #name {}

        impl GenericEdge<#quoted_edge_type> for #name {
            #[inline] fn new(edge_dest: u64, edge_type: u64) -> Self { #impl_new_for_self }
            #[inline] fn set_edge_dest(&mut self, new_edge_dest: u64) -> &mut Self { #quoted_mut_dest }
            #[inline] fn set_edge_type(&mut self, new_edge_type: u64) -> &mut Self { #quoted_mut_edge_type }
            #[inline] fn dest(&self) -> usize { #quoted_dest_access as usize }
            #[inline] fn e_type(&self) -> #quoted_edge_type { #quoted_edge_type::from(#quoted_edge_type_access) }
        }

        /// WARNING! Returns zeroed out `GenericEdgeType`. Beware if using out of box.
        impl std::default::Default for #name {
            fn default() -> Self {
                #impl_default_for_self
            }
        }

        impl std::cmp::PartialOrd for #name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.dest().cmp(&other.dest()))
            }
        }

        impl std::cmp::Ord for #name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.dest().cmp(&other.dest())
            }
        }
        impl std::cmp::PartialEq for #name {
            fn eq(&self, other: &Self) -> bool {
                self.dest() == other.dest() && self.e_type() == other.e_type()
            }
        }

        impl std::cmp::Eq for #name {}

        // use the SAME fields as in `PartialEq`
        impl std::hash::Hash for #name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.dest().hash(state);
                self.e_type().hash(state);
            }
        }

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Edge {{\n\tdest: {:?}\n\ttype: {:?}\n}}", self.dest() , self.e_type())
            }
        }

        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Edge {{\n\tdest: {}\n\ttype: {}\n}}", self.dest() , self.e_type())
            }
        }

    };

    // let dump_file = "/tmp/macro_damp.rs";
    // match std::fs::write(dump_file, expanded.to_string()) {
    //     Ok(_) => {}
    //     Err(e) => panic!("error dumpoing macro onto {}: {}", dump_file, e),
    // };

    TokenStream::from(expanded)
}

fn parse_generic_edge_attr(
    attr: &Attribute,
) -> (Option<syn::Ident>, Option<syn::Ident>, Option<Type>) {
    let mut getter = None;
    let mut setter = None;
    let mut ty = None;
    if let Meta::List(meta_list) = &attr.meta {
        // Parse as punctuated Meta, e.g. #[edge_type(getter = "edge_type", real_type = "EdgeType")]
        let parsed: syn::punctuated::Punctuated<Meta, syn::Token![,]> =
            match meta_list.parse_args_with(syn::punctuated::Punctuated::parse_terminated) {
                Ok(i) => i,
                Err(e) => panic!("error parsinf token metadata: {}", e),
            };

        for meta in parsed {
            if let Meta::NameValue(nv) = meta {
                if nv.path.is_ident("getter") || nv.path.is_ident("get") {
                    if let syn::Expr::Lit(expr_lit) = nv.value {
                        if let syn::Lit::Str(lit_str) = expr_lit.lit {
                            getter = Some(syn::Ident::new(&lit_str.value(), lit_str.span()));
                        }
                    }
                } else if nv.path.is_ident("ty") || nv.path.is_ident("real_type") {
                    if let syn::Expr::Lit(expr_lit) = nv.value {
                        if let syn::Lit::Str(lit_str) = expr_lit.lit {
                            ty = syn::parse_str::<Type>(&lit_str.value()).ok();
                        }
                    }
                } else if nv.path.is_ident("setter") || nv.path.is_ident("set") {
                    if let syn::Expr::Lit(expr_lit) = nv.value {
                        if let syn::Lit::Str(lit_str) = expr_lit.lit {
                            setter = Some(syn::Ident::new(&lit_str.value(), lit_str.span()));
                        }
                    }
                }
            }
        }
    }

    (getter, setter, ty)
}

#[proc_macro_derive(GenericEdgeType, attributes(generic_edge_type))]
pub fn derive_generic_edge_type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    // check if #[repr(C)] is present
    let has_repr_c = input.attrs.iter().any(|attr: &Attribute| {
        if attr.path().is_ident("repr") {
            if let Meta::List(MetaList { tokens, .. }) = &attr.meta {
                tokens.to_string().contains("C")
            } else {
                false
            }
        } else {
            false
        }
    });

    if !has_repr_c {
        return syn::Error::new_spanned(
            input.ident,
            "Types deriving `GenericEdgeType` must be annotated with #[repr(C)]",
        )
        .to_compile_error()
        .into();
    }

    // edges are directed by default
    let mut is_directed = quote! { true };

    // parse struct attributes amd override is_directed() with user specified value
    // methods if given
    // parse #[generic_edge_type(is_directed = "false")]
    for attr in &input.attrs {
        if attr.path().is_ident("generic_edge_type") {
            if let Meta::List(meta_list) = &attr.meta {
                let parsed: syn::punctuated::Punctuated<Meta, syn::Token![,]> = match meta_list
                    .parse_args_with(syn::punctuated::Punctuated::parse_terminated)
                {
                    Ok(i) => i,
                    Err(e) => panic!("error parsinf token metadata: {}", e),
                };

                for meta in parsed {
                    if let Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("is_directed") {
                            if let syn::Expr::Lit(expr_lit) = nv.value {
                                if let syn::Lit::Str(lit_str) = expr_lit.lit {
                                    let user_input =
                                        syn::Ident::new(&lit_str.value(), lit_str.span());
                                    // override any previous implementation
                                    is_directed = quote! { #user_input };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let static_type_assertions = quote! {
        const _: () = {
            struct _AssertEdgeTypeTraits<
                T:
                    std::marker::Copy +
                    std::clone::Clone +
                    std::convert::From<u64> +
                    std::convert::From<usize> +
                    std::default::Default +
                    std::cmp::PartialEq +
                    std::cmp::Eq +
                    std::hash::Hash +
                    std::fmt::Debug +
                    std::fmt::Display +
                    bytemuck::Pod +
                    bytemuck::Zeroable
            >(::core::marker::PhantomData<T>);
            let _ = _AssertEdgeTypeTraits::<#name>(::core::marker::PhantomData);
        };
    };

    let expanded = quote! {

        #static_type_assertions

        impl std::marker::Copy for #name {}

        impl std::clone::Clone for #name {
            fn clone(&self) -> Self {
                *self
            }
        }

        unsafe impl bytemuck::Zeroable for #name {}

        unsafe impl bytemuck::Pod for #name {}

        impl GenericEdgeType for #name {
            #[inline(always)] fn label(&self) -> usize {
                usize::from(*self)
            }
            #[inline(always)] fn set_label(&mut self, label: u64) {
                *self = #name::from(label);
            }
        }

        /// WARNING! Returns zeroed out `GenericEdgeType`. Beware if using out of box.
        impl std::default::Default for #name {
            fn default() -> Self {
                Self::from(0usize)
            }
        }

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EdgeType {{ label: {:?} }}", self.label())
            }
        }

        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EdgeType {{ label: {} }}", self.label())
            }
        }

        impl std::cmp::PartialEq for #name {
            fn eq(&self, other: &Self) -> bool {
                self.label() == other.label()
            }
        }

        impl std::cmp::Eq for #name {}

        // use the SAME fields as in `PartialEq`
        impl std::hash::Hash for #name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.label().hash(state);
            }
        }

        /// for pethraph / rustworkx_core compatibility
        impl rustworkx_core::petgraph::EdgeType for #name {
            fn is_directed() -> bool {
                #is_directed
            }
        }

    };

    TokenStream::from(expanded)
}
#[proc_macro_derive(E)]
pub fn derive_e(_input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    quote! {}.into()
}

#[proc_macro_derive(N)]
pub fn derive_n(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);
    // parse struct attributes amd override new() and/or default() with user specified
    // methods if given
    // parse #[generic_edge(constructor = "new", deafult = "default")]
    let mut already_derived = Vec::new();
    for attr in &input.attrs {
        // collect already derived macros
        if attr.path().is_ident("derive") {
            if let Meta::List(meta) = &attr.meta {
                let parsed: syn::punctuated::Punctuated<Meta, syn::Token![,]> =
                    match meta.parse_args_with(syn::punctuated::Punctuated::parse_terminated) {
                        Ok(i) => i,
                        Err(e) => panic!("error parsinf token metadata: {}", e),
                    };
                for nested in parsed {
                    if let Meta::Path(path) = nested {
                        if let Some(ident) = path.get_ident() {
                            already_derived.push(ident.to_string());
                        }
                    }
                }
            }
        }
    }
    // ensure necessary macros are derived (we want PartialEq, Eq & Hash) so the user may use
    // HashMaps and HashSets with any GenericEdge safely
    // FIXME: How to make this work without defining the struct multiple times?
    let required_traits = [
        "Clone",
        "Copy",
        "Debug",
        "PartialEq",
        "Eq",
        "PartialOrd",
        "Ord",
        "Hash",
    ];
    let mut missing_derives = Vec::new();

    for required in &required_traits {
        if !already_derived.iter().any(|d| d == required) {
            missing_derives.push(syn::Ident::new(required, proc_macro2::Span::call_site()));
        }
    }
    // if missing derives exist, add them
    if !missing_derives.is_empty() {
        let derive_attr: Attribute = syn::parse_quote! {
            #[derive( #( #missing_derives ),* )]
        };
        input.attrs.push(derive_attr);
    }

    quote!(#input).into()
}
fn has_repr_c_or_transparent(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("repr") && a.meta.to_token_stream().to_string().contains("C"))
        || attrs.iter().any(|a| {
            a.path().is_ident("repr")
                && a.meta.to_token_stream().to_string().contains("transparent")
        })
}

#[proc_macro_attribute]
pub fn sparkx_label(_attr: TokenStream, item: TokenStream) -> proc_macro::TokenStream {
    let mut input: Item = parse(item).expect("expected an item (struct/enum/union)");

    // Get attrs vec for the variants you support
    let attrs: &mut Vec<Attribute> = match &mut input {
        syn::Item::Struct(s) => &mut s.attrs,
        syn::Item::Enum(e) => &mut e.attrs,
        syn::Item::Union(u) => &mut u.attrs,
        other => {
            return syn::Error::new_spanned(other, "ensure_derives: only for structs/enums/unions")
                .to_compile_error()
                .into();
        }
    };

    // Collect already-present derive idents
    let mut have = HashSet::<String>::new();
    for a in attrs.iter().filter(|a| a.path().is_ident("derive")) {
        let paths: Punctuated<syn::Path, Token![,]> = a
            .parse_args_with(Punctuated::<syn::Path, Token![,]>::parse_terminated)
            .unwrap_or_default();
        for p in paths {
            if let Some(id) = p.get_ident() {
                have.insert(id.to_string());
            }
        }
    }

    // Build syn::Ident (proc_macro2::Ident) with a proc_macro2::Span
    let required = [
        "PartialEq",
        "Eq",
        "Hash",
        "Clone",
        "PartialOrd",
        "Ord",
        "Copy",
        "Debug",
    ];
    let missing: Vec<syn::Ident> = required
        .iter()
        .filter(|t| !have.contains(**t))
        .map(|t| syn::Ident::new(t, proc_macro2::Span::call_site()))
        .collect();

    if !missing.is_empty() {
        let derive_attr: syn::Attribute = syn::parse_quote! {
            #[derive( #( #missing ),* )]
        };
        attrs.push(derive_attr);
    }

    let extra = match &input {
        Item::Struct(ItemStruct {
            ident,
            generics,
            attrs,
            fields,
            ..
        }) => {
            // Basic layout guard
            if !has_repr_c_or_transparent(attrs) && !fields.is_empty() {
                return syn::Error::new_spanned(
                    ident,
                    "bytemuck::Pod/Zeroable require #[repr(C)] or #[repr(transparent)] (or be a unit struct).",
                ).to_compile_error().into();
            }

            // Build field-type bounds
            let mut pod_bounds = Vec::new();
            let mut zero_bounds = Vec::new();
            match fields {
                Fields::Named(named) => {
                    for f in &named.named {
                        let ty = &f.ty;
                        pod_bounds.push(quote!( #ty: ::bytemuck::Pod ));
                        zero_bounds.push(quote!( #ty: ::bytemuck::Zeroable ));
                    }
                }
                Fields::Unnamed(unnamed) => {
                    for f in &unnamed.unnamed {
                        let ty = &f.ty;
                        pod_bounds.push(quote!( #ty: ::bytemuck::Pod ));
                        zero_bounds.push(quote!( #ty: ::bytemuck::Zeroable ));
                    }
                }
                Fields::Unit => { /* ZST: fine */ }
            }

            let (ig, tg, wc) = generics.split_for_impl();

            // Merge our extra bounds with existing where-clause (if any)
            let pod_where = if wc.is_some() || !pod_bounds.is_empty() {
                quote!( where #wc #(#pod_bounds,)* )
            } else {
                quote!()
            };
            let zero_where = if wc.is_some() || !zero_bounds.is_empty() {
                quote!( where #wc #(#zero_bounds,)* )
            } else {
                quote!()
            };

            quote! {
                unsafe impl #ig ::bytemuck::Zeroable for #ident #tg #zero_where {}
                unsafe impl #ig ::bytemuck::Pod      for #ident #tg #pod_where   {}
            }
        }
        // Don’t try to implement for enums by default (layout/value constraints are trickier)
        _ => quote!(),
    };

    quote!(#input #extra).into() // back to proc_macro::TokenStream
}
