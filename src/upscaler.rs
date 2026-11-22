//! One ONNX Runtime session (TensorRT, falling back to CUDA) for a fixed window.
//!
//! When the full frame does not fit in GPU memory, the frame is split into tiles
//! and the session is compiled for the tile size instead; the tiles are blended
//! back into a full frame here.

use anyhow::{Context, Result, ensure};
use half::f16;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::{fs, path::Path};

use crate::{convert, model, tile};

/// Frame geometry of one run.
#[derive(Clone, Copy)]
struct Window {
    width: usize,
    height: usize,
    chunk: usize,
}

/// A model bound to one frame geometry and window size.
pub struct Upscaler {
    session: Session,
    window: Window,
    input: Vec<f16>,
    tiled: Option<Tiled>,
}

impl Upscaler {
    /// `max_tile` limits the tile edge in source pixels; `None` runs whole frames.
    pub fn new(
        model_path: &Path,
        cache_dir: &Path,
        width: usize,
        height: usize,
        chunk: usize,
        max_tile: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            width > 0 && height > 0 && chunk > 0,
            "invalid window geometry {width}x{height}, {chunk} frames"
        );
        let plan = max_tile.map(|max_tile| tile::Plan::new(width, height, max_tile));
        let (tile_width, tile_height) =
            plan.as_ref().map_or((width, height), tile::Plan::tile_size);
        // The engine is compiled once for the tile size and reused for every window.
        let session = create_session(model_path, cache_dir, tile_width, tile_height, chunk)?;
        let window = Window {
            width,
            height,
            chunk,
        };
        let tiled = plan
            .filter(|plan| !plan.is_single_tile())
            .map(|plan| Tiled::new(plan, window));
        Ok(Self {
            session,
            window,
            input: Vec::new(),
            tiled,
        })
    }

    /// `frames` is `chunk` RGB24 frames; `out` receives `chunk` planar
    /// `[3, 2h, 2w]` fp16 frames straight from the model.
    pub fn infer(&mut self, frames: &[u8], out: &mut Vec<f16>) -> Result<()> {
        let window = self.window;
        ensure!(
            frames.len() == window.chunk * convert::frame_len(window.width, window.height),
            "infer needs exactly {} frames of {}x{}",
            window.chunk,
            window.width,
            window.height
        );
        convert::to_nchw(
            frames,
            window.width,
            window.height,
            window.chunk,
            &mut self.input,
        );

        match self.tiled.as_mut() {
            None => infer_once(&mut self.session, &self.input, window, out),
            Some(tiled) => run_tiled(
                &mut self.session,
                tiled,
                &mut self.input,
                frames,
                window,
                out,
            ),
        }
    }
}

/// Per-tile buffers, kept between windows to avoid reallocating.
struct Tiled {
    tiles: Vec<tile::Tile>,
    /// One tile of `chunk` frames, RGB24.
    frames: Vec<u8>,
    /// One tile's model output, `[chunk, 3, 2h, 2w]` fp16.
    planes: Vec<f16>,
    /// The whole output frame accumulated in f32, `[chunk, 3, 2h, 2w]`.
    accumulate: Vec<f32>,
    /// Summed blend weights, `[2h, 2w]`, shared by all planes.
    weight_sum: Vec<f32>,
}

impl Tiled {
    fn new(plan: tile::Plan, window: Window) -> Self {
        let (tile_width, tile_height) = plan.tile_size();
        let tile_pixels = tile_width * tile_height;
        let out_frame = 4 * window.width * window.height;
        Self {
            tiles: plan.tiles(),
            frames: vec![0; window.chunk * tile_pixels * 3],
            planes: Vec::new(),
            accumulate: vec![0.0; window.chunk * 3 * out_frame],
            weight_sum: vec![0.0; out_frame],
        }
    }
}

/// Runs one window as a set of tiles and blends them into whole frames.
fn run_tiled(
    session: &mut Session,
    tiled: &mut Tiled,
    input: &mut Vec<f16>,
    frames: &[u8],
    window: Window,
    out: &mut Vec<f16>,
) -> Result<()> {
    let Tiled {
        tiles,
        frames: tile_frames,
        planes,
        accumulate,
        weight_sum,
    } = tiled;
    accumulate.fill(0.0);
    weight_sum.fill(0.0);
    let out_width = 2 * window.width;
    let plane_len = weight_sum.len();

    for tile in tiles.iter() {
        copy_tile(frames, window, tile, tile_frames);
        let tile_window = Window {
            width: tile.width,
            height: tile.height,
            chunk: window.chunk,
        };
        convert::to_nchw(tile_frames, tile.width, tile.height, window.chunk, input);
        infer_once(&mut *session, input, tile_window, planes)?;
        add_tile(
            tile,
            window.chunk,
            out_width,
            plane_len,
            planes,
            accumulate,
            weight_sum,
        );
    }

    out.clear();
    out.resize(accumulate.len(), f16::ZERO);
    for (plane, values) in accumulate.chunks_exact(plane_len).enumerate() {
        let destination = &mut out[plane * plane_len..][..plane_len];
        for ((destination, value), weight) in
            destination.iter_mut().zip(values).zip(weight_sum.iter())
        {
            *destination = if *weight > 0.0 {
                f16::from_f32(value / weight)
            } else {
                f16::ZERO
            };
        }
    }
    Ok(())
}

/// Copies one tile of every frame from the window into a packed buffer.
fn copy_tile(frames: &[u8], window: Window, tile: &tile::Tile, out: &mut [u8]) {
    let row_bytes = tile.width * 3;
    for frame in 0..window.chunk {
        for row in 0..tile.height {
            let source = ((frame * window.height + tile.y + row) * window.width + tile.x) * 3;
            let destination = (frame * tile.height + row) * row_bytes;
            out[destination..destination + row_bytes]
                .copy_from_slice(&frames[source..source + row_bytes]);
        }
    }
}

/// Adds one tile's planes, weighted per pixel, to the output accumulator.
fn add_tile(
    tile: &tile::Tile,
    chunk: usize,
    out_width: usize,
    plane_len: usize,
    planes: &[f16],
    accumulate: &mut [f32],
    weight_sum: &mut [f32],
) {
    let width = 2 * tile.width;
    let height = 2 * tile.height;
    let (x, y) = (2 * tile.x, 2 * tile.y);

    // The weights are the same for every plane and frame, so sum them once.
    for row in 0..height {
        let weights = &tile.weights[row * width..][..width];
        let sums = &mut weight_sum[(y + row) * out_width + x..][..width];
        for (sum, weight) in sums.iter_mut().zip(weights) {
            *sum += weight;
        }
    }

    for plane in 0..chunk * 3 {
        let source = &planes[plane * width * height..][..width * height];
        let target = &mut accumulate[plane * plane_len..][..plane_len];
        for row in 0..height {
            let weights = &tile.weights[row * width..][..width];
            let values = &source[row * width..][..width];
            let destination = &mut target[(y + row) * out_width + x..][..width];
            for ((destination, value), weight) in
                destination.iter_mut().zip(values).zip(weights.iter())
            {
                *destination += value.to_f32() * weight;
            }
        }
    }
}

/// Runs one window through the session and copies the output planes into `out`.
fn infer_once(
    session: &mut Session,
    input: &[f16],
    window: Window,
    out: &mut Vec<f16>,
) -> Result<()> {
    let Window {
        width,
        height,
        chunk,
    } = window;
    let tensor = TensorRef::from_array_view(([1, chunk, 3, height, width], input))
        .context("wrapping the model input")?;
    let outputs = session
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
    out.clear();
    out.extend_from_slice(values);
    Ok(())
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
