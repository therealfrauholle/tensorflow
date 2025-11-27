use std::{fs::File, path::PathBuf, process::Command};

fn main() {
    let path_to_tensorflow = PathBuf::from(std::env::var("PATH_TO_TENSORFLOW").unwrap());

    let path_to_ops_pbtxt = path_to_tensorflow.join("tensorflow/core/ops/ops.pbtxt");

    let ops = std::fs::read(path_to_ops_pbtxt).unwrap();

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut generated = File::create(out_dir.join("raw_ops.rs")).unwrap();
    tensorflow_op_codegen::eager::generate(&ops, &mut generated).unwrap();

    let mut generated = File::create(out_dir.join("ops_impl.rs")).unwrap();
    tensorflow_op_codegen::ops::generate(&ops, &mut generated).unwrap();
    Command::new("rustfmt")
        .arg(format!("{}", out_dir.join("ops_impl.rs").display()))
        .status()
        .unwrap()
        .success()
        .then_some(())
        .unwrap();
    Command::new("rustfmt")
        .arg(format!("{}", out_dir.join("raw_ops.rs").display()))
        .status()
        .unwrap()
        .success()
        .then_some(())
        .unwrap();
}
