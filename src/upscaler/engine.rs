//! The fixed-shape TensorRT session and its reusable fp16 input tensor.

use super::Window;
use crate::{convert, model};
use anyhow::{Context, Result, ensure};
use half::f16;
use ort::{
    ep,
    logging::LogLevel,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::{
    fs,
    path::Path,
    sync::Arc,
    time::{Instant, SystemTime},
};

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
        .with_logger(Arc::new(ort_log))
        .build()
        .context("initializing ONNX Runtime")?;
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating the engine cache {}", cache_dir.display()))?;
    let cache = cache_dir.to_string_lossy();
    // One profile point: the window size is fixed for the whole run.
    let shape = format!("input:1x{chunk}x3x{height}x{width}");
    eprintln!(
        "TensorRT FP16 with CUDA fallback; model {}; profile {shape}; cache {}",
        model_path.display(),
        cache_dir.display()
    );
    let cached = engine_stamps(cache_dir);
    let started = Instant::now();

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
    eprintln!(
        "session ready in {:.1} s ({})",
        started.elapsed().as_secs_f32(),
        if engine_stamps(cache_dir) == cached {
            "engine cache reused"
        } else {
            "engine built and cached"
        }
    );
    Ok(session)
}

/// ONNX Runtime messages at warning level and above; provider fallbacks are
/// reported here.
fn ort_log(level: LogLevel, category: &str, _id: &str, _location: &str, message: &str) {
    if level >= LogLevel::Warning {
        eprintln!("{category} {message}");
    }
}

/// Snapshot of the cached engines: a changed stamp means one was rebuilt.
fn engine_stamps(cache: &Path) -> Vec<(String, SystemTime)> {
    let Ok(entries) = fs::read_dir(cache) else {
        return Vec::new();
    };
    let mut stamps: Vec<_> = entries
        .flatten()
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "engine")
        })
        .filter_map(|entry| {
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((entry.file_name().to_string_lossy().into_owned(), modified))
        })
        .collect();
    stamps.sort();
    stamps
}

#[cfg(test)]
mod tests {
    use super::engine_stamps;
    use std::fs;

    #[test]
    fn engine_stamps_report_only_cached_engines() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.engine"), b"x").unwrap();
        fs::write(dir.path().join("a.profile"), b"x").unwrap();
        let stamps = engine_stamps(dir.path());
        assert_eq!(stamps.len(), 1);
        assert_eq!(stamps[0].0, "a.engine");
        assert!(engine_stamps(&dir.path().join("missing")).is_empty());
    }
}
