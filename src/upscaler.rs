//! One ONNX Runtime session (TensorRT, falling back to CUDA) for a fixed window.

use anyhow::{Context, Result, ensure};
use half::f16;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::{fs, path::Path};

use crate::convert;

/// A model bound to one frame geometry and window size.
///
/// TensorRT compiles for a single shape, so the geometry is fixed when the
/// session is built and the engine is cached for it.
pub struct Upscaler {
    session: Session,
    width: usize,
    height: usize,
    chunk: usize,
    input: Vec<f16>,
}

impl Upscaler {
    pub fn new(
        model_path: &Path,
        cache_dir: &Path,
        width: usize,
        height: usize,
        chunk: usize,
    ) -> Result<Self> {
        ensure!(
            width > 0 && height > 0 && chunk > 0,
            "invalid window geometry {width}x{height}, {chunk} frames"
        );
        let session = create_session(model_path, cache_dir, width, height, chunk)?;
        Ok(Self {
            session,
            width,
            height,
            chunk,
            input: Vec::new(),
        })
    }

    /// `frames` is `chunk` RGB24 frames; `out` receives `chunk` planar
    /// `[3, 2h, 2w]` fp16 frames straight from the model.
    pub fn infer(&mut self, frames: &[u8], out: &mut Vec<f16>) -> Result<()> {
        let (width, height, chunk) = (self.width, self.height, self.chunk);
        ensure!(
            frames.len() == chunk * convert::frame_len(width, height),
            "infer needs exactly {chunk} frames of {width}x{height}"
        );
        convert::to_nchw(frames, width, height, chunk, &mut self.input);

        let input = TensorRef::from_array_view(([1, chunk, 3, height, width], &*self.input))
            .context("wrapping the model input")?;
        let outputs = self
            .session
            .run(ort::inputs![input])
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
        out.clear();
        out.extend_from_slice(values);
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
        .commit_from_memory(&crate::model::model_bytes(model_path)?)
        .with_context(|| format!("loading {}", model_path.display()))?;
    Ok(session)
}
