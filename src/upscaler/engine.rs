//! The fixed-shape TensorRT session and its reusable fp16 input tensor.

use super::Window;
use crate::{convert, model};
use anyhow::{Context, Result, ensure};
use half::f16;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::{fs, path::Path};

pub(super) struct Engine {
    session: Session,
    window: Window,
    input: Vec<f16>,
}

impl Engine {
    pub(super) fn new(model: &Path, cache: &Path, window: Window) -> Result<Self> {
        let session = create_session(model, cache, window.width, window.height, window.chunk)?;
        Ok(Self {
            session,
            window,
            input: Vec::new(),
        })
    }

    /// Consume the output while it is borrowed from ORT (no tile-sized copy).
    pub(super) fn run(&mut self, rgb: &[u8], consume: impl FnOnce(&[f16])) -> Result<()> {
        let Window {
            width,
            height,
            chunk,
        } = self.window;
        convert::to_nchw(rgb, width, height, chunk, &mut self.input);
        let tensor =
            TensorRef::from_array_view(([1, chunk, 3, height, width], self.input.as_slice()))
                .context("wrapping the model input")?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .context("running the model")?;
        let (shape, values) = outputs[0]
            .try_extract_tensor::<f16>()
            .context("reading the model output")?;

        let expected = [1, chunk, 3, 2 * height, 2 * width];
        ensure!(
            shape
                .iter()
                .copied()
                .eq(expected.iter().map(|dim| *dim as i64)),
            "model output {shape:?} does not match {expected:?}"
        );
        consume(values);
        Ok(())
    }
}

fn create_session(
    model_path: &Path,
    cache_dir: &Path,
    width: usize,
    height: usize,
    chunk: usize,
) -> Result<Session> {
    let environment = ort::init()
        .with_name("video_scale")
        .build()
        .context("initializing ONNX Runtime")?;
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating the engine cache {}", cache_dir.display()))?;
    let cache = cache_dir.to_string_lossy();
    // One profile point: the window size is fixed for the whole run.
    let shape = format!("input:1x{chunk}x3x{height}x{width}");

    let session = Session::builder(&environment)
        .map_err(|error| anyhow::anyhow!("creating the ONNX Runtime session: {error}"))?
        .with_execution_providers([
            ep::TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(&*cache)
                .with_timing_cache(true)
                .with_timing_cache_path(&*cache)
                .with_profile_min_shapes(shape.clone())
                .with_profile_opt_shapes(shape.clone())
                .with_profile_max_shapes(shape)
                .build()
                .error_on_failure(),
            ep::CUDA::default().build().error_on_failure(),
        ])
        .map_err(|error| {
            anyhow::anyhow!("registering the TensorRT and CUDA execution providers: {error}")
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| anyhow::anyhow!("setting the graph optimization level: {error}"))?
        .commit_from_memory(&model::model_bytes(model_path)?)
        .with_context(|| format!("loading {}", model_path.display()))?;
    Ok(session)
}
