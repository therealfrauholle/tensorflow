extern crate protoc_rust;

use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let tensorflow_folder = &args[1];
    let output_folder = Path::new(&args[2]);
    protoc_rust::Codegen::new()
        .out_dir(
            output_folder
                .join("tensorflow-proto/src/messages")
                .to_str()
                .expect("Unable to format output path for main crate"),
        )
        .inputs(
            [
                "third_party/xla/third_party/tsl/tsl/protobuf/coordination_config.proto",
                "third_party/xla/third_party/tsl/tsl/protobuf/rpc_options.proto",
            ]
            .iter()
            .map(|p| format!("{tensorflow_folder}/{p}"))
            .collect::<Vec<_>>(),
        )
        .include(Path::new(tensorflow_folder).join("third_party/xla/third_party/tsl"))
        .run()
        .unwrap();
    protoc_rust::Codegen::new()
        .out_dir(
            output_folder
                .join("tensorflow-proto/src/messages")
                .to_str()
                .expect("Unable to format output path for main crate"),
        )
        .inputs(
            [
                "tensorflow/core/framework/allocation_description.proto",
                "tensorflow/core/framework/attr_value.proto",
                "tensorflow/core/framework/cost_graph.proto",
                "tensorflow/core/framework/full_type.proto",
                "tensorflow/core/framework/function.proto",
                "tensorflow/core/framework/graph.proto",
                "tensorflow/core/framework/graph_debug_info.proto",
                "tensorflow/core/framework/node_def.proto",
                "tensorflow/core/framework/op_def.proto",
                "tensorflow/core/framework/resource_handle.proto",
                "tensorflow/core/framework/step_stats.proto",
                "tensorflow/core/framework/tensor.proto",
                "tensorflow/core/framework/tensor_description.proto",
                "tensorflow/core/framework/tensor_shape.proto",
                "tensorflow/core/framework/types.proto",
                "tensorflow/core/framework/variable.proto",
                "tensorflow/core/framework/versions.proto",
                "tensorflow/core/protobuf/cluster.proto",
                "tensorflow/core/protobuf/config.proto",
                "tensorflow/core/protobuf/debug.proto",
                "tensorflow/core/protobuf/meta_graph.proto",
                "tensorflow/core/protobuf/rewriter_config.proto",
                "tensorflow/core/protobuf/saved_model.proto",
                "tensorflow/core/protobuf/saved_object_graph.proto",
                "tensorflow/core/protobuf/saver.proto",
                "tensorflow/core/protobuf/struct.proto",
                "tensorflow/core/protobuf/trackable_object_graph.proto",
                "tensorflow/core/protobuf/verifier_config.proto",
            ]
            .iter()
            .map(|p| format!("{tensorflow_folder}/{p}"))
            .collect::<Vec<_>>(),
        )
        .include(tensorflow_folder)
        .include(Path::new(tensorflow_folder).join("third_party/xla/third_party/tsl"))
        .run()
        .unwrap();
    protoc_rust::Codegen::new()
        .out_dir(
            output_folder
                .join("tensorflow-proto/src/opdef")
                .to_str()
                .expect("Unable to format output path for ops crate"),
        )
        .inputs(
            [
                "tensorflow/core/framework/attr_value.proto",
                "tensorflow/core/framework/full_type.proto",
                "tensorflow/core/framework/op_def.proto",
                "tensorflow/core/framework/resource_handle.proto",
                "tensorflow/core/framework/tensor.proto",
                "tensorflow/core/framework/tensor_shape.proto",
                "tensorflow/core/framework/types.proto",
            ]
            .iter()
            .map(|p| format!("{tensorflow_folder}/{p}"))
            .collect::<Vec<_>>(),
        )
        .include(tensorflow_folder)
        .run()
        .unwrap();
}
