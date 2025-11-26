#![allow(clippy::uninlined_format_args)]

pub mod protos {
    pub use tensorflow_proto::opdef::*;
}

pub mod parser;

pub mod ops;

pub mod eager;
