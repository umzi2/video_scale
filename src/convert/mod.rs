//! Conversions between RGB24 frames and the model's planar frames.
//!
//! A frame is `height * width * 3` interleaved bytes. The model reads
//! `[t, 3, h, w]` fp16 planes in `[0, 1]` and writes `[t, 3, 2h, 2w]`. Blending
//! happens on those planes, so the interleaved layout is touched only when a
//! frame enters or leaves the pipeline.
//!
//! Every function here asserts that the buffers match the requested geometry and
//! uses an AVX2 kernel when the CPU has one.

#[cfg(target_arch = "x86_64")]
mod simd;

#[cfg(test)]
mod tests;

use half::f16;

/// Bytes in one RGB24 frame.
#[inline]
pub fn frame_len(width: usize, height: usize) -> usize {
    width * height * 3
}

/// Converts `t` interleaved RGB24 frames into `[t, 3, h, w]` fp16 in `[0, 1]`.
///
/// # Panics
///
/// Panics unless `frames` holds exactly `t` frames.
pub fn to_nchw(frames: &[u8], width: usize, height: usize, t: usize, out: &mut Vec<f16>) {
    assert_eq!(
        frames.len(),
        t * frame_len(width, height),
        "expected {t} RGB24 frames of {width}x{height}"
    );
    out.resize(frames.len(), f16::ZERO);

    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: `available` checked that AVX2 and F16C are usable.
        unsafe { simd::to_nchw(frames, width, height, t, out) };
        return;
    }

    for f in 0..t {
        let frame = &frames[f * frame_len(width, height)..][..frame_len(width, height)];
        for (c, plane) in out[f * 3 * width * height..][..3 * width * height]
            .chunks_exact_mut(width * height)
            .enumerate()
        {
            for (destination, pixel) in plane.iter_mut().zip(frame.as_chunks::<3>().0) {
                *destination = f16::from_f32(pixel[c] as f32 / 255.0);
            }
        }
    }
}

/// One frame's fp16 planes into `out`, scaled by `weight` and overwriting it.
///
/// # Panics
///
/// Panics unless `out` has the same length as `planes`.
pub fn scale_planes(planes: &[f16], weight: f32, out: &mut [f32]) {
    assert_eq!(planes.len(), out.len(), "plane buffers differ in length");

    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: `available` checked that AVX2 and F16C are usable.
        unsafe { simd::weighted(planes, weight, out, false) };
        return;
    }

    for (destination, source) in out.iter_mut().zip(planes) {
        *destination = source.to_f32() * weight;
    }
}

/// `acc += planes * weight`.
///
/// # Panics
///
/// Panics unless `acc` has the same length as `planes`.
pub fn accumulate_planes(acc: &mut [f32], planes: &[f16], weight: f32) {
    assert_eq!(planes.len(), acc.len(), "plane buffers differ in length");

    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: `available` checked that AVX2 and F16C are usable.
        unsafe { simd::weighted(planes, weight, acc, true) };
        return;
    }

    for (accumulated, source) in acc.iter_mut().zip(planes) {
        *accumulated += source.to_f32() * weight;
    }
}

/// Interleaves three fp16 planes into one RGB24 frame, clamping to `[0, 1]`.
///
/// # Panics
///
/// Panics unless `planes` describes exactly `out.len() / 3` pixels.
pub fn planes_to_u8(planes: &[f16], scale: f32, out: &mut [u8]) {
    let channels = split_channels(planes, out);
    let mut done = 0;

    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: `available` checked that AVX2 and F16C are usable, and the
        // channels all cover at least `done` pixels.
        done = unsafe { simd::interleave_f16(channels, scale, out) };
    }

    interleave(channels, scale, out, done);
}

/// Interleaves three f32 planes into one RGB24 frame.
///
/// # Panics
///
/// Panics unless `planes` describes exactly `out.len() / 3` pixels.
pub fn blend_to_u8(planes: &[f32], scale: f32, out: &mut [u8]) {
    let channels = split_channels(planes, out);
    let mut done = 0;

    #[cfg(target_arch = "x86_64")]
    if simd::available() {
        // SAFETY: `available` checked that AVX2 is usable, and the channels all
        // cover at least `done` pixels.
        done = unsafe { simd::interleave_f32(channels, scale, out) };
    }

    interleave(channels, scale, out, done);
}

/// Splits a planar frame into its three channels, checking the geometry.
#[inline]
fn split_channels<'a, T>(planes: &'a [T], out: &[u8]) -> [&'a [T]; 3] {
    assert_eq!(out.len() % 3, 0, "RGB24 buffer is not a whole frame");
    let pixels = out.len() / 3;
    assert_eq!(planes.len(), 3 * pixels, "expected three planes");
    let (r, rest) = planes.split_at(pixels);
    let (g, b) = rest.split_at(pixels);
    [r, g, b]
}

/// Interleaves and quantises channels from pixel `skip` on.
fn interleave<T: Copy + Into<f32>>(channels: [&[T]; 3], scale: f32, out: &mut [u8], skip: usize) {
    for i in skip..out.len() / 3 {
        for (channel, plane) in channels.iter().enumerate() {
            out[i * 3 + channel] = quantise(plane[i].into() * scale);
        }
    }
}

/// `[0, 1]` to 8-bit, rounding half away from zero.
#[inline]
fn quantise(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0).round() as u8
}
