use crate::parser;
use crate::protos::op_def::OpDef;
use proc_macro2::Ident;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use protobuf::RepeatedField;
use quote::format_ident;
use quote::quote;
use quote::ToTokens;
use std::collections::HashMap;
use std::error::Error;
use std::io::Write;
use std::result::Result;
use tensorflow_proto::opdef::op_def::OpDef_ArgDef;
use tensorflow_proto::opdef::op_def::OpDef_AttrDef;

/// The order of arguments is well-defined because the tensorflow API identifies an argument by an index.
#[derive(Debug, Clone)]
struct Arguments<'a>(&'a RepeatedField<OpDef_ArgDef>);

impl<'a> Arguments<'a> {
    /// Snake case identifier, unique.
    fn rust_field(&self) -> Vec<Ident> {
        self.0
            .iter()
            .map(|edge| Ident::new_raw(&snake_name(&edge.name), Span::call_site()))
            .collect()
    }

    /// When non-scalar this will reference the attribute defining the length of the argument..
    fn number_attr(&self) -> Vec<Option<&'a str>> {
        self.0
            .iter()
            .map(|attr| {
                if attr.number_attr.is_empty() {
                    None
                } else {
                    Some(attr.number_attr.as_str())
                }
            })
            .collect()
    }

    /// For each argument specifies previous arguments.
    fn previous_arguments(&self) -> Vec<PreviousArguments> {
        let mut seen_edges = PreviousArguments::default();
        self.c_number_attr()
            .iter()
            .map(|edge| {
                let current = seen_edges.clone();
                if let Some(number_attr) = edge {
                    *seen_edges
                        .previous_nonscalar_attributes_count
                        .entry(number_attr.to_string())
                        .or_insert(0) += 1;
                } else {
                    seen_edges.previous_scalar_count += 1;
                }
                current
            })
            .collect()
    }

    /// Identifies the attribute to the tensorflow API.
    fn c_number_attr(&self) -> Vec<Option<&'a str>> {
        self.0
            .iter()
            .map(|edge| {
                if edge.number_attr.is_empty() {
                    None
                } else {
                    Some(edge.number_attr.as_str())
                }
            })
            .collect()
    }
}

struct Attributes<'a>(&'a OpDef, Box<dyn Fn(&OpDef_AttrDef) -> bool + 'a>);

impl<'a> Attributes<'a> {
    fn filtered<'b>(&'b self) -> impl Iterator<Item = &'a OpDef_AttrDef> + 'b {
        self.0.attr.iter().filter(move |attr| self.1(attr))
    }

    fn name(&self) -> Vec<Ident> {
        self.filtered()
            .map(|attr| Ident::new_raw(&attr.name, Span::call_site()))
            .collect()
    }

    fn types(&self) -> Vec<AttrType> {
        self.filtered()
            .map(|attr| AttrType::from_str(&attr.field_type))
            .collect()
    }

    fn c_name(&self) -> Vec<&'a str> {
        self.filtered().map(|attr| attr.name.as_str()).collect()
    }
}

/// Restructured [OpDef] for easier rust code generation.
struct Operation<'a>(&'a OpDef);

impl<'a> Arguments<'a> {}

impl<'a> Operation<'a> {
    /// All independent attributes, excluding those referenced by [Edge::number_attr].
    fn independent_attrs<'b>(&'b self) -> Attributes<'b> {
        Attributes(
            self.0,
            Box::new(move |attr| {
                self.input()
                    .number_attr()
                    .iter()
                    .all(|number_attr| *number_attr != Some(attr.name.as_str()))
                    && self
                        .output()
                        .number_attr()
                        .iter()
                        .all(|number_attr| *number_attr != Some(&attr.name))
            }),
        )
    }

    /// Identifies the operation to the Tensorflow API.
    fn c_name(&self) -> &'a str {
        &self.0.name
    }

    fn output(&self) -> Arguments<'a> {
        Arguments(&self.0.output_arg)
    }

    fn input(&self) -> Arguments<'a> {
        Arguments(&self.0.input_arg)
    }

    fn input_aliases(&self) -> Vec<Ident> {
        (0..self.0.input_arg.len())
            .map(|index| Ident::new(&format!("O{index}"), Span::call_site()))
            .collect()
    }
}

#[derive(Debug, Default, Clone)]
pub struct PreviousArguments {
    /// Number of preceeding scalar arguments.
    previous_scalar_count: usize,
    /// Previous arguments with dynamic (non-scalar) length.
    ///
    /// The length is defined
    /// specifying the length will have an entry here, mapping to the count of appearances for
    /// all previous edges.
    previous_nonscalar_attributes_count: HashMap<String, usize>,
}

impl PreviousArguments {
    /// Scalar offset of the next edge, given the name of the number_attr of the previous edge or None
    /// when the edge has only a width of one
    fn edge_index(&self) -> TokenStream {
        let scalar_edges = self.previous_scalar_count;
        self
        .previous_nonscalar_attributes_count
        .iter()
        .fold(quote!((#scalar_edges as i32)), |scalar_offset, (string, count)|
                quote!(#scalar_offset + self.op.get_attr_int(#string)? as i32 * (#count as i32))
        )
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum AttrType {
    Tensor,
    List(AttrTypePrimitive),
    Primitive(AttrTypePrimitive),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum AttrTypePrimitive {
    String,
    Float,
    Integer,
    Shape,
    Type,
    Bool,
}

impl AttrType {
    fn from_str(attr: &str) -> Self {
        match attr {
            "string" => AttrType::Primitive(AttrTypePrimitive::String),
            "int" => AttrType::Primitive(AttrTypePrimitive::Integer),
            "float" => AttrType::Primitive(AttrTypePrimitive::Float),
            "bool" => AttrType::Primitive(AttrTypePrimitive::Bool),
            "type" => AttrType::Primitive(AttrTypePrimitive::Type),
            "shape" => AttrType::Primitive(AttrTypePrimitive::Shape),
            "tensor" => AttrType::Tensor,
            "func" => AttrType::Primitive(AttrTypePrimitive::String),
            "list(string)" => AttrType::List(AttrTypePrimitive::String),
            "list(int)" => AttrType::List(AttrTypePrimitive::Integer),
            "list(float)" => AttrType::List(AttrTypePrimitive::Float),
            "list(bool)" => AttrType::List(AttrTypePrimitive::Bool),
            "list(type)" => AttrType::List(AttrTypePrimitive::Type),
            "list(shape)" => AttrType::List(AttrTypePrimitive::Shape),
            "list(func)" => AttrType::List(AttrTypePrimitive::String),
            t => {
                panic!("unrecognized field type {:?}", t)
            }
        }
    }
}

impl ToTokens for AttrTypePrimitive {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(match self {
            AttrTypePrimitive::String => quote!(::std::string::String),
            AttrTypePrimitive::Float => quote!(f32),
            AttrTypePrimitive::Integer => quote!(i64),
            AttrTypePrimitive::Shape => quote!(crate::Shape),
            AttrTypePrimitive::Type => quote!(crate::DataType),
            AttrTypePrimitive::Bool => quote!(bool),
        });
    }
}

impl ToTokens for AttrType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(match self {
            AttrType::List(attr_type_primitive) => quote!(::std::vec::Vec<#attr_type_primitive>),
            AttrType::Primitive(attr_type_primitive) => quote!(#attr_type_primitive),
            AttrType::Tensor => quote!(::std::boxed::Box<dyn crate::AnyTensor>),
        });
    }
}

fn buildfn_set_attr(
    rust_name: &Ident,
    attr_type: &AttrType,
    c_name: &str,
    node_var: &Ident,
) -> TokenStream {
    let setter = match attr_type {
        AttrType::Primitive(AttrTypePrimitive::String) => {
            quote!( #node_var.set_attr_string(#c_name, value)?; )
        }
        AttrType::Primitive(AttrTypePrimitive::Type) => {
            quote!( #node_var.set_attr_type(#c_name, *value)?; )
        }
        AttrType::Primitive(AttrTypePrimitive::Bool) => {
            quote!( #node_var.set_attr_bool(#c_name, *value)?; )
        }
        AttrType::Primitive(AttrTypePrimitive::Float) => {
            quote!( #node_var.set_attr_float(#c_name, *value)?; )
        }
        AttrType::Primitive(AttrTypePrimitive::Integer) => {
            quote!( #node_var.set_attr_int(#c_name, *value)?; )
        }
        AttrType::Primitive(AttrTypePrimitive::Shape) => {
            quote!( #node_var.set_attr_shape(#c_name, value)?; )
        }
        AttrType::List(AttrTypePrimitive::String) => {
            quote!( #node_var.set_attr_string_list(#c_name, value)?; )
        }
        AttrType::List(AttrTypePrimitive::Float) => {
            quote!( #node_var.set_attr_float_list(#c_name, value)?; )
        }
        AttrType::List(AttrTypePrimitive::Integer) => {
            quote!( #node_var.set_attr_int_list(#c_name, value)?; )
        }
        AttrType::List(AttrTypePrimitive::Type) => {
            quote!( #node_var.set_attr_type_list(#c_name, value)?; )
        }
        AttrType::List(AttrTypePrimitive::Shape) => {
            quote!( #node_var.set_attr_shape_list(#c_name, value)?; )
        }
        AttrType::Tensor => {
            quote!( #node_var.set_attr_any_tensor(#c_name, value)?; )
        }
        ty => panic!("Unrecognized attribute type for {}: {:?}", rust_name, ty),
    };
    quote! {
        if let ::std::option::Option::Some(value) = &self.#rust_name {
            #setter
        }
    }
}

fn inst_edge_method(
    rust_name: &Ident,
    offsets: &PreviousArguments,
    number_attr: Option<&str>,
    edge_type: &Ident,
    name: &str,
) -> TokenStream {
    let edge_index = offsets.edge_index();

    if let Some(name_attr) = number_attr {
        quote! {
            #[doc = "Returns a Vector of "]
            #[doc = stringify!(#rust_name)]
            #[doc = " for '"]
            #[doc = stringify!(#rust_name)]
            #[doc = "' "]
            #[doc = stringify!(#edge_type)]
            #[doc = " of this "]
            #[doc = #name]
            #[doc = " operation."]
            pub fn #rust_name(&self) -> crate::Result<Vec<crate::#edge_type>>{
                let dynamic_offset = (#edge_index) as i32;
                let mut ret = vec![];
                for i in dynamic_offset..self.op.get_attr_int(#name_attr)? as i32 {
                    ret.push(O::#edge_type(&self.op.clone(), 1));
                }

                Ok(ret)
            }
        }
    } else {
        quote! {
            #[doc = "Returns the '"]
            #[doc = stringify!(#rust_name)]
            #[doc = "' "]
            #[doc = stringify!(#edge_type)]
            #[doc = " of this "]
            #[doc = #name]
            #[doc = " operation."]
            pub fn #rust_name(&self) -> crate::Result<crate::#edge_type>{
                let offset = (#edge_index) as i32;
                Ok(O::#edge_type(&self.op, offset))
            }
        }
    }
}

fn per_op_code(op: &Operation) -> TokenStream {
    let c_name = op.c_name();
    let builder_name = Ident::new_raw(c_name, Span::call_site());
    let inst_name = format_ident!("{}Inst", c_name);
    let short_build_name = Ident::new_raw(&snake_name(c_name), Span::call_site());

    let rust_field_output = op.output().rust_field();
    let rust_field_input = op.input().rust_field();
    let previous_edges_output = op.output().previous_arguments();
    let previous_edges_input = op.input().previous_arguments();
    let c_number_attr_output = op.output().c_number_attr();
    let c_number_attr_input = op.input().c_number_attr();
    let in_type_alias = op.input_aliases();
    let attr_name = op.independent_attrs().name();
    let attr_type = op.independent_attrs().types();
    let attr_c_name = op.independent_attrs().c_name();

    let builder_var = Ident::new("builder", Span::call_site());
    let scope_var = Ident::new("scope", Span::call_site());

    let set_attrs = attr_name
        .iter()
        .zip(attr_type.iter())
        .zip(attr_c_name.iter())
        .map(|((attr_name, attr_type), attr_c_name)| {
            buildfn_set_attr(attr_name, attr_type, attr_c_name, &builder_var)
        })
        .collect::<Vec<_>>();

    let input_argument_type = c_number_attr_input
        .iter()
        .map(|input| {
            if input.is_some() {
                quote!(Vec<crate::Output>)
            } else {
                quote!(crate::Output)
            }
        })
        .collect::<Vec<_>>();

    let addinput_method = c_number_attr_input
        .iter()
        .map(|number_attr| {
            if number_attr.is_some() {
                quote!(add_input_list)
            } else {
                quote!(add_input)
            }
        })
        .collect::<Vec<_>>();

    let (dynamic_input_number_attr, dynamic_input_name): (Vec<&str>, Vec<&Ident>) =
        c_number_attr_input
            .iter()
            .zip(rust_field_input.iter())
            .filter_map(|(number_attr, rust_name)| {
                number_attr.map(|number_attr| (number_attr, rust_name))
            })
            .collect();

    let output_getter = rust_field_output
        .iter()
        .zip(previous_edges_output.iter())
        .zip(c_number_attr_output.iter())
        .map(|((rust_name, previous_edges), number_attr)| {
            inst_edge_method(
                rust_name,
                previous_edges,
                number_attr.as_deref(),
                &Ident::new("Output", Span::call_site()),
                c_name,
            )
        })
        .collect::<Vec<_>>();

    let input_getter = rust_field_input
        .iter()
        .zip(previous_edges_input.iter())
        .zip(c_number_attr_input.iter())
        .map(|((rust_name, previous_edges), number_attr)| {
            inst_edge_method(
                rust_name,
                previous_edges,
                number_attr.as_deref(),
                &Ident::new("Input", Span::call_site()),
                c_name,
            )
        });

    quote! {
        #[doc = "Builder for the `"]
        #[doc = #c_name]
        #[doc = " operation."]
        #[derive(::std::fmt::Debug, ::std::default::Default)]
        pub struct #builder_name {
            control_inputs: ::std::vec::Vec<crate::Operation>,
            #(#attr_name: ::std::option::Option<#attr_type>,)*
        }

        impl #builder_name {
            /// just no docs
            pub fn new() -> Self {
                Self::default()
            }

            #(
                #[doc = "Sets the `"]
                #[doc = #c_name]
                #[doc = "` attribute."]
                pub fn #attr_name<T: std::convert::Into<#attr_type>>(mut self, value: T) -> Self {

                    self.#attr_name = ::std::option::Option::Some(value.into());
                    self
                }
            )*

            /// Adds a control input.
            pub fn add_control_input(mut self, op: crate::Operation) -> Self {
                self.control_inputs.push(op);
                self
            }

            #[doc = "Builds the `"]
            #[doc = #c_name]
            #[doc = "` operation."]
            pub fn build<
                #(#in_type_alias: ::std::convert::Into<#input_argument_type>),*
            >(
                &self,
                #(#rust_field_input: #in_type_alias,)*
                #scope_var: &mut crate::Scope
            ) -> crate::Result<crate::Operation> {
                self.build_impl(#(#rust_field_input.into(),)* #scope_var)
            }

            // FIXME document why we need this functiom over 'build_impl'. What are
            // 'control_inputs'
            fn build_impl(
                &self,
                #(#rust_field_input: #input_argument_type,)*
                #scope_var: &mut crate::Scope
            ) -> crate::Result<crate::Operation> {
                #scope_var.new_operation(#c_name, |#builder_var| {
                    #(#builder_var.#addinput_method(&#rust_field_input);)*
                    for op in &self.control_inputs {
                        #builder_var.add_control_input(op);
                    }
                    #(#builder_var.set_attr_int(#dynamic_input_number_attr, #dynamic_input_name.len() as i64)?;)*
                    #(#set_attrs)*
                    ::std::result::Result::Ok(())
                })
            }

            #[doc = "Builds a new instance of '"]
            #[doc = #c_name]
            #[doc = "' Operation with it's Outputs and Inputs exposed as methods."]
            pub fn build_instance(
                &self,
                #(#rust_field_input: #input_argument_type,)*
                #scope_var: &mut crate::Scope
            ) -> crate::Result<#inst_name> {
                let op = #scope_var.new_operation(#c_name, |#builder_var| {
                    #(#builder_var.#addinput_method(&#rust_field_input);)*
                    #(#builder_var.set_attr_int(#dynamic_input_number_attr, #dynamic_input_name.len() as i64)?;)*
                    #(#set_attrs)*
                    Ok(())
                })?;

                Ok(#inst_name {op})
            }
        }

        #[doc = "Shorthand for `"]
        #[doc = stringify!(#builder_name)]
        #[doc ="::new().build("]
        #(#[doc = stringify!(#rust_field_input)] #[doc = ","])*
        #[doc = stringify!(#scope_var)]
        #[doc = ")`."]
        pub fn #short_build_name<
            #(#in_type_alias:  ::std::convert::Into<#input_argument_type>),*
        >(
            #(#rust_field_input: #in_type_alias,)*
            #scope_var: &mut crate::Scope
        ) -> crate::Result<crate::Operation> {
            #builder_name::new().build(#(#rust_field_input.into(),)* #scope_var)
        }

        #[doc = "An instance of'"]
        #[doc = #c_name]
        #[doc ="' Operation with it's Outputs and Inputs exposed as methods."]
        #[derive(Debug, Clone)]
        pub struct #inst_name<O> {
            #[doc = "An instance of a fully built "]
            #[doc = #c_name]
            #[doc =" Operation in a Tensorflow graph."]
            pub op: O
        }

        impl<O: Clone> #inst_name<O> {
            #(#output_getter)*
            #(#input_getter)*
        }

        impl<O> From<#inst_name<O>> for O {
            fn from(inst: #inst_name) -> crate::Operation {
                inst.op
            }
        }
    }
}

fn snake_name(name: &str) -> String {
    let mut s = String::new();
    let mut was_lower = false;
    for c in name.chars() {
        if c.is_uppercase() {
            if was_lower {
                s.push('_');
            }
            was_lower = false;
        } else {
            was_lower = true;
        }
        for cc in c.to_lowercase() {
            s.push(cc);
        }
    }
    s
}

pub fn generate<W: Write>(ops_pbtxt: &[u8], mut output: W) -> Result<(), Box<dyn Error>> {
    let ops = parser::parse(ops_pbtxt).inspect_err(|e| {
        println!("Parse error at {:?}", e.pos);
        if let Some(p) = &e.pos {
            let input = String::from_utf8_lossy(ops_pbtxt);
            println!("Previous: {}", &input[0..*p]);
            println!("Next: {}", &input[*p..]);
        }
    })?;
    write!(
        &mut output,
        "// DO NOT EDIT. Generated by tensorflow-op-codegen/src/main.rs.\n\n"
    )?;
    for op in &ops {
        if ![
            "Placeholder",
            "Add",
            "Sub",
            "Mul",
            "Assign",
            "NoOp",
            "ApplyGradientDescent",
            "RandomStandardNormal",
            "Tanh",
            "MatMul",
            "ZerosLike",
            "ApplyAdadelta",
            "ConcatV2",
        ]
        .contains(&op.name.as_str())
        {
            continue;
        }
        writeln!(&mut output, "{}", per_op_code(&Operation(op)))?;
    }
    println!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_name() {
        assert_eq!(&snake_name("foo"), "foo");
        assert_eq!(&snake_name("fooBar"), "foo_bar");
        assert_eq!(&snake_name("FooBar"), "foo_bar");
        assert_eq!(&snake_name("abcXYZ"), "abc_xyz");
        assert_eq!(&snake_name("abcXYZdef"), "abc_xyzdef");
    }
}
