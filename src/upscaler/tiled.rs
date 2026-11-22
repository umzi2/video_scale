//! Spatial blending for one fixed tile layout, reused across temporal windows.

use super::{Window, engine::Engine};
use crate::tile::{Plan, Tile};
use anyhow::Result;
use half::f16;

pub(super) struct Tiled {
    tiles: Vec<Tile>,
    rgb: Vec<u8>,
    sums: Vec<f32>,
    weights: Vec<f32>,
}

impl Tiled {
    pub(super) fn new(plan: Plan, window: Window) -> Self {
        let (width, height) = plan.tile_size();
        let tiles = plan.tiles();
        let mut weights = vec![0.0; 4 * window.width * window.height];
        for tile in &tiles {
            let rows = output_rows(tile, &mut weights, 2 * window.width);
            for (row, mask) in rows.zip(tile.weights.chunks_exact(2 * tile.width)) {
                for (sum, weight) in row.iter_mut().zip(mask) {
                    *sum += weight;
                }
            }
        }
        Self {
            tiles,
            rgb: vec![0; 3 * window.chunk * width * height],
            sums: vec![0.0; 3 * window.chunk * weights.len()],
            weights,
        }
    }

    pub(super) fn run(
        &mut self,
        engine: &mut Engine,
        frames: &[u8],
        window: Window,
        out: &mut Vec<f16>,
    ) -> Result<()> {
        self.sums.fill(0.0);
        for tile in &self.tiles {
            copy_tile(frames, window, tile, &mut self.rgb);
            engine.run(&self.rgb, |planes| {
                add_tile(
                    tile,
                    planes,
                    &mut self.sums,
                    2 * window.width,
                    self.weights.len(),
                );
            })?;
        }
        out.clear();
        out.reserve(self.sums.len());
        for plane in self.sums.chunks_exact(self.weights.len()) {
            out.extend(
                plane
                    .iter()
                    .zip(&self.weights)
                    .map(|(sum, weight)| f16::from_f32(sum / weight)),
            );
        }
        Ok(())
    }
}

/// Rows this upscaled tile occupies in a single full-resolution output plane.
fn output_rows<'a>(
    tile: &Tile,
    plane: &'a mut [f32],
    stride: usize,
) -> impl Iterator<Item = &'a mut [f32]> {
    let x = 2 * tile.x;
    let end = x + 2 * tile.width;
    plane
        .chunks_exact_mut(stride)
        .skip(2 * tile.y)
        .take(2 * tile.height)
        .map(move |row| &mut row[x..end])
}

fn add_tile(tile: &Tile, values: &[f16], sums: &mut [f32], stride: usize, plane_len: usize) {
    let width = 2 * tile.width;
    for (source, target) in values
        .chunks_exact(tile.weights.len())
        .zip(sums.chunks_exact_mut(plane_len))
    {
        let rows = output_rows(tile, target, stride);
        for ((row, samples), mask) in rows
            .zip(source.chunks_exact(width))
            .zip(tile.weights.chunks_exact(width))
        {
            for ((sum, value), weight) in row.iter_mut().zip(samples).zip(mask) {
                *sum += value.to_f32() * weight;
            }
        }
    }
}

fn copy_tile(frames: &[u8], window: Window, tile: &Tile, rgb: &mut [u8]) {
    let stride = 3 * window.width;
    let row_bytes = 3 * tile.width;
    let x = 3 * tile.x;
    let tile_frames = rgb.chunks_exact_mut(row_bytes * tile.height);
    for (frame, target) in frames.chunks_exact(stride * window.height).zip(tile_frames) {
        let rows = frame.chunks_exact(stride).skip(tile.y);
        for (row, dest) in rows.zip(target.chunks_exact_mut(row_bytes)) {
            dest.copy_from_slice(&row[x..x + row_bytes]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_identity_preserves_every_frame_channel_and_edge() {
        let window = Window {
            width: 17,
            height: 13,
            chunk: 2,
        };
        let len = 3 * window.width * window.height * window.chunk;
        let frames: Vec<_> = (0..len).map(|i| (i % 251) as u8).collect();
        let mut tiled = Tiled::new(Plan::new(window.width, window.height, 8), window);
        for tile in &tiled.tiles {
            copy_tile(&frames, window, tile, &mut tiled.rgb);
            let mut planes = Vec::new();
            for frame in tiled.rgb.chunks_exact(3 * tile.width * tile.height) {
                for c in 0..3 {
                    for y in 0..2 * tile.height {
                        for x in 0..2 * tile.width {
                            let pixel = y / 2 * tile.width + x / 2;
                            planes.push(f16::from_f32(frame[3 * pixel + c] as f32));
                        }
                    }
                }
            }
            add_tile(
                tile,
                &planes,
                &mut tiled.sums,
                2 * window.width,
                tiled.weights.len(),
            );
        }
        for (index, plane) in tiled.sums.chunks_exact(tiled.weights.len()).enumerate() {
            for (pos, (sum, weight)) in plane.iter().zip(&tiled.weights).enumerate() {
                let y = pos / (2 * window.width);
                let x = pos % (2 * window.width);
                let pixel = (index / 3 * window.height + y / 2) * window.width + x / 2;
                assert_eq!(
                    f16::from_f32(sum / weight).to_f32(),
                    frames[3 * pixel + index % 3] as f32
                );
            }
        }
    }
}
