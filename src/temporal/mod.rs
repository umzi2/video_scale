//! Streaming a video through a recurrent model in fixed-size windows.
//!
//! The model sees `chunk` frames at a time and reproduces them at twice the
//! resolution. Neighbouring windows share `overlap` frames; instead of throwing
//! one copy away, both copies are kept and cross-faded, so every frame keeps a
//! contribution from both windows.
//!
//! Cross-fading happens on the model's fp16 planes (accumulated as f32) and only
//! the finished frame is quantised to RGB24, so no precision is lost between
//! windows.

#[cfg(test)]
mod tests;

use crate::convert::{accumulate_planes, blend_to_u8, frame_len, planes_to_u8, scale_planes};
use anyhow::{Result, bail, ensure};
use half::f16;
use std::{
    collections::VecDeque,
    io::{Read, Write},
};

/// Frame geometry and window layout of one run.
#[derive(Clone, Copy)]
struct Layout {
    width: usize,
    height: usize,
    chunk: usize,
    overlap: usize,
}

impl Layout {
    fn new(width: usize, height: usize, chunk: usize, overlap: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("frame dimensions must be positive, got {width}x{height}");
        }
        if chunk == 0 {
            bail!("chunk_size must be positive");
        }
        if overlap >= chunk {
            bail!("require 0 <= overlap < chunk_size, got {overlap} >= {chunk}");
        }
        Ok(Self {
            width,
            height,
            chunk,
            overlap,
        })
    }

    fn frame_bytes(self) -> usize {
        frame_len(self.width, self.height)
    }

    fn out_frame_bytes(self) -> usize {
        4 * self.frame_bytes()
    }

    /// Distance between the starts of two consecutive windows.
    fn stride(self) -> usize {
        self.chunk - self.overlap
    }
}

/// Sin² weights that split a frame between the two windows predicting it.
///
/// Frame `k` of an overlap (`k` in `1..=overlap`) gets `sin²(π/2 · k/(overlap+1))`
/// from the newer window; the older window gets the complementary weight from the
/// other end of its window, so the weights of a frame always sum to one.
struct Crossfade {
    ramp: Vec<f32>,
    chunk: usize,
    overlap: usize,
    stride: usize,
}

impl Crossfade {
    fn new(chunk: usize, overlap: usize) -> Self {
        let ramp = (1..=overlap)
            .map(|k| {
                let angle = k as f64 * std::f64::consts::PI / (2.0 * (overlap + 1) as f64);
                angle.sin().powi(2) as f32
            })
            .collect();
        Self {
            ramp,
            chunk,
            overlap,
            stride: chunk - overlap,
        }
    }

    /// Whether frame `index` also appears in the next window.
    fn carries(&self, index: usize, more: bool) -> bool {
        more && index >= self.stride
    }

    /// This window's weight for frame `index`.
    fn weight(&self, index: usize, first: bool, more: bool) -> f32 {
        let mut weight = 1.0;
        if !first && index < self.overlap {
            weight *= self.ramp[index];
        }
        if self.carries(index, more) {
            weight *= self.ramp[self.chunk - 1 - index];
        }
        weight
    }
}

/// A frame several windows contribute to, accumulated in f32 planes.
struct Blended {
    planes: Vec<f32>,
    weight: f32,
}

/// Reads whole frames into `buf`; returns how many arrived.
fn read_frames(source: &mut impl Read, buf: &mut [u8], frame: usize) -> Result<usize> {
    let mut read = 0;
    while read < buf.len() {
        match source.read(&mut buf[read..])? {
            0 => break,
            n => read += n,
        }
    }
    if read % frame != 0 {
        bail!("truncated RGB24 frame: {read} bytes, frame is {frame}");
    }
    Ok(read / frame)
}

/// Repeats the last available frame into the rest of a short window.
fn pad_short_window(window: &mut [u8], valid: usize, chunk: usize, frame: usize) {
    let (loaded, padding) = window[..chunk * frame].split_at_mut(valid * frame);
    let last = &loaded[loaded.len() - frame..];
    for slot in padding.chunks_exact_mut(frame) {
        slot.copy_from_slice(last);
    }
}

fn emit_planes(planes: &[f16], out: &mut [u8], sink: &mut impl Write) -> Result<()> {
    planes_to_u8(planes, 1.0, out);
    sink.write_all(out)?;
    Ok(())
}

fn emit_blended(blended: &Blended, out: &mut [u8], sink: &mut impl Write) -> Result<()> {
    blend_to_u8(&blended.planes, 1.0 / blended.weight, out);
    sink.write_all(out)?;
    Ok(())
}

/// Reads RGB24 frames from `source`, upscales them window by window through
/// `infer` and writes the result to `sink`.
///
/// `infer` receives `chunk` frames of `width x height` and must fill its output
/// with `chunk` planar `[3, 2h, 2w]` fp16 frames.
pub fn process(
    source: &mut impl Read,
    sink: &mut impl Write,
    width: usize,
    height: usize,
    chunk: usize,
    overlap: usize,
    mut infer: impl FnMut(&[u8], &mut Vec<f16>) -> Result<()>,
) -> Result<()> {
    let layout = Layout::new(width, height, chunk, overlap)?;
    let (frame, out_frame, stride) = (
        layout.frame_bytes(),
        layout.out_frame_bytes(),
        layout.stride(),
    );
    let crossfade = Crossfade::new(chunk, overlap);

    // One frame of lookahead tells a full final window from one with a successor.
    let mut window = vec![0u8; (chunk + 1) * frame];
    let mut loaded = 0;
    let mut first = true;
    let mut result: Vec<f16> = Vec::new();
    let mut out = vec![0u8; out_frame];
    let mut queue = VecDeque::<Blended>::new();

    loop {
        loaded += read_frames(source, &mut window[loaded * frame..], frame)?;
        if loaded == 0 {
            return Ok(());
        }
        let more = loaded > chunk;
        let valid = loaded.min(chunk);
        if valid < chunk {
            pad_short_window(&mut window, valid, chunk, frame);
        }

        result.clear();
        infer(&window[..chunk * frame], &mut result)?;
        ensure!(
            result.len() == chunk * out_frame,
            "model returned {} values, expected {} for {chunk} frames",
            result.len(),
            chunk * out_frame
        );

        let carried = queue.len();
        for index in 0..valid {
            let planes = &result[index * out_frame..(index + 1) * out_frame];
            let keep = crossfade.carries(index, more);
            let weight = crossfade.weight(index, first, more);
            if index < carried {
                let mut blended = queue.pop_front().unwrap();
                accumulate_planes(&mut blended.planes, planes, weight);
                blended.weight += weight;
                if !keep {
                    emit_blended(&blended, &mut out, sink)?;
                }
                // Completed frames precede still-pending ones on the next pass
                // through the queue; reuse their storage for the new tail below.
                if more {
                    queue.push_back(blended);
                }
            } else if keep {
                let mut accumulated = if first {
                    vec![0.0; out_frame]
                } else {
                    queue.pop_front().unwrap().planes
                };
                scale_planes(planes, weight, &mut accumulated);
                queue.push_back(Blended {
                    planes: accumulated,
                    weight,
                });
            } else {
                emit_planes(planes, &mut out, sink)?;
            }
        }

        if !more {
            return Ok(());
        }
        // Keep the overlap plus the lookahead frame; they seed the next window.
        window.copy_within(stride * frame..(chunk + 1) * frame, 0);
        loaded = overlap + 1;
        first = false;
    }
}
