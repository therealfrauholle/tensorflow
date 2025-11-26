use std::{fs::File, path::PathBuf};

fn main() {
    let path_to_tensorflow = PathBuf::from(std::env::var("PATH_TO_TENSORFLOW").unwrap());

    let path_to_ops_pbtxt = path_to_tensorflow.join("tensorflow/core/ops/ops.pbtxt");

    let ops = std::fs::read(path_to_ops_pbtxt).unwrap();

    let mut generated = File::create("raw_ops.rs").unwrap();
    tensorflow_op_codegen::eager::generate(&ops, &mut generated).unwrap();

    let mut generated = File::create("ops_impl.rs").unwrap();
    tensorflow_op_codegen::ops::generate(&ops, &mut generated).unwrap();
}
