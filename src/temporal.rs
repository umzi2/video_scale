//! Split a video stream into windows of `chunk` frames for a recurrent model,
//! optionally cross-fading the frames that neighbouring windows predict twice.

use anyhow::{Result, bail};
use std::collections::VecDeque;
use std::io::{Read, Write};

/// Fill `buf` with whole frames from `source`; returns how many frames arrived.
fn fill(source: &mut impl Read, buf: &mut [u8], frame: usize) -> Result<usize> {
    let mut size = 0;
    while size < buf.len() {
        match source.read(&mut buf[size..])? {
            0 => break,
            n => size += n,
        }
    }
    if size % frame != 0 {
        bail!("truncated RGB24 input frame");
    }
    Ok(size / frame)
}

pub fn process(
    source: &mut impl Read,
    sink: &mut impl Write,
    width: usize,
    height: usize,
    chunk: usize,
    overlap: usize,
    mut infer: impl FnMut(&[u8], &mut Vec<f32>) -> Result<()>,
) -> Result<()> {
    if width == 0 || height == 0 || overlap >= chunk {
        bail!("expected positive dimensions and 0 <= overlap < chunk_size");
    }
    let frame = width * height * 3;
    let out_frame = frame * 4;
    let stride = chunk - overlap;
    // sin² ramp: complementary weights for the two halves of an overlap.
    let ramp: Vec<f32> = (1..=overlap)
        .map(|i| {
            (i as f64 * std::f64::consts::PI / (2.0 * (overlap + 1) as f64))
                .sin()
                .powi(2) as f32
        })
        .collect();
    // Real frames plus one lookahead frame telling a full final window from a successor.
    let mut window = vec![0u8; (chunk + 1) * frame];
    let mut result = Vec::new();
    let mut out = vec![0u8; out_frame];
    let mut pending: VecDeque<(Vec<f32>, f32)> = VecDeque::new();
    let mut used = 0;
    let mut first = true;
    loop {
        used += fill(source, &mut window[used * frame..], frame)?;
        if used == 0 {
            return Ok(());
        }
        let more = used > chunk;
        let valid = used.min(chunk);
        if valid < chunk {
            let last = ((valid - 1) * frame)..(valid * frame);
            let last = window[last].to_vec();
            for slot in window[valid * frame..chunk * frame].chunks_mut(frame) {
                slot.copy_from_slice(&last);
            }
        }
        result.clear();
        infer(&window[..chunk * frame], &mut result)?;
        if result.len() != chunk * out_frame {
            bail!(
                "model returned {} values, expected {}",
                result.len(),
                chunk * out_frame
            );
        }
        let carried = pending.len();
        for i in 0..valid {
            let mut weight = 1.0;
            if !first && i < overlap {
                weight *= ramp[i];
            }
            let keep = more && i >= stride;
            if keep {
                weight *= ramp[chunk - 1 - i];
            }
            let rgb = &result[i * out_frame..(i + 1) * out_frame];
            let (pixels, total) = if i < carried || keep {
                let mut blend = rgb.to_vec();
                if weight != 1.0 {
                    blend.iter_mut().for_each(|p| *p *= weight);
                }
                let mut total = weight;
                if i < carried {
                    let (previous, previous_weight) = pending.pop_front().unwrap();
                    blend.iter_mut().zip(previous).for_each(|(p, q)| *p += q);
                    total += previous_weight;
                }
                if keep {
                    pending.push_back((blend, total));
                    continue;
                }
                (blend, total)
            } else {
                (rgb.to_vec(), 1.0)
            };
            let scale = 1.0 / total;
            for (dst, src) in out.iter_mut().zip(pixels) {
                *dst = (src * scale * 255.0).clamp(0.0, 255.0).round() as u8;
            }
            sink.write_all(&out)?;
        }
        if !more {
            return Ok(());
        }
        window.copy_within(stride * frame..(chunk + 1) * frame, 0);
        used = overlap + 1;
        first = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upscale(frames: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames.len() * 4);
        for y in 0..height * 2 {
            for x in 0..width * 2 {
                for c in 0..3 {
                    let v = frames[((y / 2) * width + x / 2) * 3 + c];
                    out.push(v as f32 / 255.0);
                }
            }
        }
        out
    }

    #[test]
    fn schedule_padding_and_exact_frames() {
        for (chunk, overlap) in [(1, 0), (4, 0), (4, 2), (4, 3)] {
            for length in 0..14 {
                let (width, height) = (3, 2);
                let frame = width * height * 3;
                let frames: Vec<u8> = (0..length * frame).map(|i| i as u8).collect();
                let mut sink = Vec::new();
                let mut calls = 0usize;
                let stride = chunk - overlap;
                {
                    let mut source: &[u8] = &frames;
                    process(
                        &mut source,
                        &mut sink,
                        width,
                        height,
                        chunk,
                        overlap,
                        |window, dst| {
                            let start = calls * stride;
                            let filled = frames.len().min((start + chunk) * frame) - start * frame;
                            assert_eq!(
                                &window[..filled],
                                &frames[start * frame..start * frame + filled]
                            );
                            if filled < chunk * frame {
                                let last = &window[filled - frame..filled];
                                assert!(
                                    window[filled..chunk * frame]
                                        .chunks(frame)
                                        .all(|w| w == last)
                                );
                            }
                            for f in 0..chunk {
                                dst.extend(upscale(
                                    &window[f * frame..(f + 1) * frame],
                                    width,
                                    height,
                                ));
                            }
                            calls += 1;
                            Ok(())
                        },
                    )
                    .unwrap();
                }
                assert_eq!(
                    sink,
                    (0..length)
                        .flat_map(
                            |f| upscale(&frames[f * frame..(f + 1) * frame], width, height)
                                .into_iter()
                                .map(|v| (v * 255.0).round() as u8)
                        )
                        .collect::<Vec<u8>>()
                );
                let expected_calls = if length == 0 {
                    0
                } else {
                    1 + (length.saturating_sub(chunk) + stride - 1) / stride
                };
                assert_eq!(calls, expected_calls);
                assert_eq!(sink.len(), length * frame * 4);
            }
        }
    }

    #[test]
    fn pairwise_sinusoidal_crossfade() {
        let mut sink = Vec::new();
        let mut source: &[u8] = &[0u8; 6 * 3];
        let values = [0.0f32, 1.0];
        let mut call = 0;
        process(&mut source, &mut sink, 1, 1, 5, 3, |_, dst| {
            dst.extend(std::iter::repeat_n(values[call], 5 * 4 * 3));
            call += 1;
            Ok(())
        })
        .unwrap();
        let expected: Vec<u8> = [0u8, 0, 37, 128, 218, 255]
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, 12))
            .collect();
        assert_eq!(sink, expected);
    }

    #[test]
    fn four_intersecting_windows_keep_every_contribution() {
        let a = (2.0 - 2f32.sqrt()) / 4.0;
        let b = 0.5;
        let c = (2.0 + 2f32.sqrt()) / 4.0;
        let weights = [
            [1.0, 0.0, 0.0, 0.0],
            [c, a, 0.0, 0.0],
            [b, b * c, a, 0.0],
            [a, b * c, b * c, a],
            [0.0, a, b * c, b],
            [0.0, 0.0, a, c],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for active in 0..4 {
            let mut sink = Vec::new();
            let mut source: &[u8] = &[0u8; 7 * 3];
            let mut call = 0;
            process(&mut source, &mut sink, 1, 1, 4, 3, |_, dst| {
                dst.extend(std::iter::repeat_n(
                    if call == active { 1.0 } else { 0.0 },
                    4 * 4 * 3,
                ));
                call += 1;
                Ok(())
            })
            .unwrap();
            for (row, w) in weights.iter().enumerate() {
                let sum: f32 = w.iter().sum();
                let value = (w[active] / sum * 255.0).round() as u8;
                assert!(
                    sink[row * 12..(row + 1) * 12].iter().all(|v| *v == value),
                    "frame {row} active {active}"
                );
            }
        }
    }

    #[test]
    fn clip_and_quantize_after_blending() {
        let colors = [[-0.25f32, 1.25, 0.6 / 255.0], [0.75, 0.25, 1.6 / 255.0]];
        let mut sink = Vec::new();
        let mut source: &[u8] = &[0u8; 3 * 3];
        let mut call = 0;
        process(&mut source, &mut sink, 1, 1, 2, 1, |_, dst| {
            let color = colors[call];
            for _ in 0..2 {
                dst.extend(color.iter().flat_map(|c| std::iter::repeat_n(*c, 4)));
            }
            call += 1;
            Ok(())
        })
        .unwrap();
        let expected: Vec<u8> = [[0u8, 255, 1], [64, 191, 1], [191, 64, 2]]
            .iter()
            .flat_map(|row| row.iter().flat_map(|v| std::iter::repeat_n(*v, 4)))
            .collect();
        assert_eq!(sink, expected);
    }

    #[test]
    fn rejects_partial_frames_and_bad_geometry() {
        let mut sink = Vec::new();
        let mut source: &[u8] = &[0u8; 4];
        assert!(process(&mut source, &mut sink, 1, 1, 2, 0, |_, _| Ok(())).is_err());
        for (w, h, chunk, overlap) in [
            (0, 1, 1, 0),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
            (1, 1, 3, 3),
            (1, 1, 3, 4),
        ] {
            let mut sink = Vec::new();
            let mut source: &[u8] = &[];
            assert!(process(&mut source, &mut sink, w, h, chunk, overlap, |_, _| Ok(())).is_err());
        }
    }
}
