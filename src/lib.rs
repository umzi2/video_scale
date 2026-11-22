//! Video upscaling with a recurrent VSR model through ONNX Runtime and TensorRT.
//!
//! The CLI in `src/main.rs` is a thin shell around these modules:
//!
//! - [`config`] reads the `*.toml` run description.
//! - [`video`] streams frames through FFmpeg and publishes the result atomically.
//! - [`temporal`] splits the stream into overlapping windows and cross-fades them.
//! - [`upscaler`] runs one window through the model, tiling oversized frames.
//! - [`tile`] plans the tiles and their cross-fade weights.
//! - [`convert`] converts between RGB24 frames and the model's fp16 planes.
//! - [`model`] loads the ONNX graph, lowering an unreleased opset if needed.

pub mod config;
pub mod convert;
pub mod model;
pub mod temporal;
pub mod tile;
pub mod upscaler;
pub mod video;
