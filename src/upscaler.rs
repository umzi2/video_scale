//! One ONNX Runtime session (TensorRT, falling back to CUDA) for a fixed window.
//!
//! When the full frame does not fit in GPU memory, the frame is split into tiles
//! and the session is compiled for the tile size instead; the tiles are blended
//! back into a full frame here.

mod engine;
mod tiled;

use self::{engine::Engine, tiled::Tiled};
use crate::{convert, tile::Plan};
use anyhow::{Result, ensure};
use half::f16;
use std::path::Path;

/// Frame geometry of one run.
#[derive(Clone, Copy)]
struct Window {
    width: usize,
    height: usize,
    chunk: usize,
}

/// A model bound to one frame geometry and window size.
pub struct Upscaler {
    engine: Engine,
    window: Window,
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
        let plan = max_tile
            .map(|limit| Plan::new(width, height, limit))
            .filter(|plan| !plan.is_single_tile());
        let window = Window {
            width,
            height,
            chunk,
        };
        let (width, height) = plan.as_ref().map_or((width, height), Plan::tile_size);
        let engine = Engine::new(
            model_path,
            cache_dir,
            Window {
                width,
                height,
                chunk,
            },
        )?;
        let tiled = plan.map(|plan| Tiled::new(plan, window));
        Ok(Self {
            engine,
            window,
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
        match &mut self.tiled {
            Some(tiled) => tiled.run(&mut self.engine, frames, window, out),
            None => self.engine.run(frames, |planes| {
                out.clear();
                out.extend_from_slice(planes);
            }),
        }
    }
}
