# 🦀 Standalone Rust Inference

Currently tested only with OpenVINO Padim model with normalization within the graph.
Uses local clone of [OpenVINO Rust Bindings](https://github.com/intel/openvino-rs/) as bindings for 2.0 API haven't been released yet.
Probably won't work on your machine.

```bash
cd tools/experimental/rust_inference
cargo run ~/datasets/MVTec/bottle/test/contamination/000.png ./model/model.xml
```
