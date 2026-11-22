//! TensorRT (falling back to CUDA) session for one fixed window size.

use anyhow::{Result, ensure};
use half::f16;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::{fs, path::Path};

use crate::{convert, model};

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
        cache: &Path,
        width: usize,
        height: usize,
        chunk: usize,
    ) -> Result<Self> {
        let env = ort::init().with_name("video_scale").build()?;
        fs::create_dir_all(cache)?;
        let cache = cache.to_string_lossy();
        // One profile point: the window size is fixed for the whole run.
        let shape = format!("input:1x{chunk}x3x{height}x{width}");
        let session = Session::builder(&env)?
            .with_execution_providers([
                ep::TensorRT::default()
                    .with_fp16(true)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&*cache)
                    .with_timing_cache(true)
                    .with_timing_cache_path(&*cache)
                    .with_profile_min_shapes(shape.clone())
                    .with_profile_opt_shapes(shape.clone())
                    .with_profile_max_shapes(shape.clone())
                    .build()
                    .error_on_failure(),
                ep::CUDA::default().build().error_on_failure(),
            ])
            .map_err(|e| anyhow::anyhow!("registering execution providers: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("setting optimization level: {e}"))?
            .commit_from_memory(&model::model_bytes(model_path)?)
            .map_err(|e| anyhow::anyhow!("loading {:?}: {e}", model_path))?;
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
        let (w, h, t) = (self.width, self.height, self.chunk);
        ensure!(
            frames.len() == t * w * h * 3,
            "infer needs exactly {t} frames"
        );
        convert::to_nchw(frames, w, h, t, &mut self.input);
        let input = TensorRef::from_array_view(([1usize, t, 3, h, w], &*self.input))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f16>()?;
        let expected = [1, t, 3, 2 * h, 2 * w];
        ensure!(
            shape.iter().copied().eq(expected.iter().map(|d| *d as i64)),
            "model output {shape:?} does not match {expected:?}"
        );
        out.clear();
        out.extend_from_slice(values);
        Ok(())
    }
}
