//! Layout/precision conversions between video frames and model tensors.
//!
//! Frames are RGB24 (`h * w * 3` bytes); the model works on `[1, t, 3, h, w]`
//! fp16 planes in `[0, 1]`. Blending happens on those planes, so only the final
//! quantisation interleaves channels back into RGB24.

// The SIMD kernels are entered only after `is_x86_feature_detected!`, which is the
// contract each `unsafe fn` below documents; individual intrinsics stay unannotated.
#![allow(unsafe_op_in_unsafe_fn)]

use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m128i, __m256};

/// Bytes of one RGB24 frame.
#[inline]
pub fn frame_len(w: usize, h: usize) -> usize {
    w * h * 3
}

/// `frames` is `t` interleaved RGB24 frames; writes `[1, t, 3, h, w]` fp16 in `[0, 1]`.
pub fn to_nchw(frames: &[u8], w: usize, h: usize, t: usize, out: &mut Vec<f16>) {
    out.clear();
    out.resize(t * 3 * w * h, f16::ZERO);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c") {
        // SAFETY: feature detection above.
        unsafe { to_nchw_avx2(frames, w, h, t, out) };
        return;
    }
    to_nchw_scalar(frames, w, h, t, out);
}

pub fn to_nchw_scalar(frames: &[u8], w: usize, h: usize, t: usize, out: &mut [f16]) {
    let frame = frame_len(w, h);
    for f in 0..t {
        let plane = &frames[f * frame..(f + 1) * frame];
        for c in 0..3 {
            let dst = &mut out[(f * 3 + c) * w * h..(f * 3 + c + 1) * w * h];
            for (d, p) in dst.iter_mut().zip(plane.as_chunks::<3>().0) {
                *d = f16::from_f32(p[c] as f32 / 255.0);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn to_nchw_avx2(frames: &[u8], w: usize, h: usize, t: usize, out: &mut [f16]) {
    use std::arch::x86_64::*;
    let frame = frame_len(w, h);
    let pixels = w * h;
    let scale = _mm256_set1_ps(1.0 / 255.0);
    let mask = _mm256_set1_epi32(0xff);
    // Byte offsets of eight consecutive pixels in the interleaved frame.
    let offsets = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    for f in 0..t {
        let src = unsafe { frames.as_ptr().add(f * frame) };
        for c in 0..3 {
            let dst = unsafe { out.as_mut_ptr().add((f * 3 + c) * pixels) };
            let mut x = 0;
            // One extra pixel of slack: the gather loads four bytes per pixel.
            while x + 9 <= pixels {
                let packed =
                    unsafe { _mm256_i32gather_epi32::<1>(src.add(3 * x) as *const i32, offsets) };
                let shifted = match c {
                    0 => packed,
                    1 => _mm256_srli_epi32::<8>(packed),
                    _ => _mm256_srli_epi32::<16>(packed),
                };
                let values = _mm256_and_si256(shifted, mask);
                let floats = _mm256_mul_ps(_mm256_cvtepi32_ps(values), scale);
                let halves = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(floats);
                unsafe { _mm_storeu_si128(dst.add(x) as *mut __m128i, halves) };
                x += 8;
            }
            while x < pixels {
                unsafe {
                    *dst.add(x) = f16::from_f32(src.add(x * 3 + c).read() as f32 / 255.0);
                }
                x += 1;
            }
        }
    }
}

/// Converts one frame's three fp16 planes into `out` with `weight`, overwriting it.
pub fn store_planes(planes: &[f16], weight: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c") {
        // SAFETY: feature detection above.
        unsafe { weighted_avx2(planes, weight, out, false) };
        return;
    }
    for (d, s) in out.iter_mut().zip(planes) {
        *d = s.to_f32() * weight;
    }
}

/// `acc += planes * weight`.
pub fn accumulate_planes(acc: &mut [f32], planes: &[f16], weight: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c") {
        // SAFETY: feature detection above.
        unsafe { weighted_avx2(planes, weight, acc, true) };
        return;
    }
    for (a, s) in acc.iter_mut().zip(planes) {
        *a += s.to_f32() * weight;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn weighted_avx2(planes: &[f16], weight: f32, out: &mut [f32], add: bool) {
    use std::arch::x86_64::*;
    let n = out.len();
    let w = _mm256_set1_ps(weight);
    let mut i = 0;
    while i + 8 <= n {
        let halves = unsafe { _mm_loadu_si128(planes.as_ptr().add(i) as *const __m128i) };
        let mut floats = _mm256_mul_ps(_mm256_cvtph_ps(halves), w);
        if add {
            let acc = unsafe { _mm256_loadu_ps(out.as_ptr().add(i)) };
            floats = _mm256_add_ps(acc, floats);
        }
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), floats) };
        i += 8;
    }
    while i < n {
        let v = planes[i].to_f32() * weight;
        out[i] = if add { out[i] + v } else { v };
        i += 1;
    }
}

/// Interleaves three fp16 planes into one RGB24 frame, clamping to `[0, 1]`.
pub fn planes_to_u8(planes: &[f16], scale: f32, out: &mut [u8]) {
    let pixels = out.len() / 3;
    let (r, gb) = planes.split_at(pixels);
    let (g, b) = gb.split_at(pixels);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c") {
        // SAFETY: feature detection above.
        unsafe { interleave_f16(r, g, b, scale, out) };
        return;
    }
    for i in 0..pixels {
        out[i * 3] = quantise(r[i].to_f32() * scale);
        out[i * 3 + 1] = quantise(g[i].to_f32() * scale);
        out[i * 3 + 2] = quantise(b[i].to_f32() * scale);
    }
}

/// Interleaves three f32 planes into one RGB24 frame.
pub fn blend_to_u8(planes: &[f32], scale: f32, out: &mut [u8]) {
    let pixels = out.len() / 3;
    let (r, gb) = planes.split_at(pixels);
    let (g, b) = gb.split_at(pixels);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: feature detection above.
        unsafe { interleave_f32(r, g, b, scale, out) };
        return;
    }
    for i in 0..pixels {
        out[i * 3] = quantise(r[i] * scale);
        out[i * 3 + 1] = quantise(g[i] * scale);
        out[i * 3 + 2] = quantise(b[i] * scale);
    }
}

/// `[0, 1]` -> 8-bit, rounding half away from zero.
#[inline]
fn quantise(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0).round() as u8
}

/// Interleaves sixteen 8-bit channel vectors into 48 bytes of RGB24.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn interleave3(r: __m128i, g: __m128i, b: __m128i, out: *mut u8) {
    use std::arch::x86_64::*;
    let sources = [r, g, b];
    for chunk in 0..3 {
        let mut bytes = _mm_setzero_si128();
        for (channel, source) in sources.iter().enumerate() {
            let mut mask = [0x80u8; 16];
            for (lane, slot) in mask.iter_mut().enumerate() {
                let index = chunk * 16 + lane;
                if index % 3 == channel {
                    *slot = (index / 3) as u8;
                }
            }
            let mask = _mm_loadu_si128(mask.as_ptr().cast());
            bytes = _mm_or_si128(bytes, _mm_shuffle_epi8(*source, mask));
        }
        _mm_storeu_si128(out.add(chunk * 16).cast(), bytes);
    }
}

/// Quantises eight f16 values into the low eight bytes of a vector.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn quantise8(source: __m128i, scale: __m256) -> __m128i {
    use std::arch::x86_64::*;
    let floats = _mm256_mul_ps(_mm256_cvtph_ps(source), scale);
    let zero = _mm256_setzero_ps();
    let max = _mm256_set1_ps(255.0);
    let rounded = _mm256_add_ps(
        _mm256_min_ps(_mm256_max_ps(floats, zero), max),
        _mm256_set1_ps(0.5),
    );
    let lo = _mm_cvttps_epi32(_mm256_castps256_ps128(rounded));
    let hi = _mm_cvttps_epi32(_mm256_extractf128_ps(rounded, 1));
    _mm_packus_epi16(_mm_packus_epi32(lo, hi), _mm_setzero_si128())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn interleave_f16(r: &[f16], g: &[f16], b: &[f16], scale: f32, out: &mut [u8]) {
    use std::arch::x86_64::*;
    let pixels = out.len() / 3;
    let factor = _mm256_set1_ps(scale * 255.0);
    let mut i = 0;
    while i + 16 <= pixels {
        let load = |plane: &[f16]| unsafe {
            (
                _mm_loadu_si128(plane.as_ptr().add(i).cast()),
                _mm_loadu_si128(plane.as_ptr().add(i + 8).cast()),
            )
        };
        let (r0, r1) = load(r);
        let (g0, g1) = load(g);
        let (b0, b1) = load(b);
        let r8 = _mm_unpacklo_epi64(quantise8(r0, factor), quantise8(r1, factor));
        let g8 = _mm_unpacklo_epi64(quantise8(g0, factor), quantise8(g1, factor));
        let b8 = _mm_unpacklo_epi64(quantise8(b0, factor), quantise8(b1, factor));
        unsafe { interleave3(r8, g8, b8, out.as_mut_ptr().add(i * 3)) };
        i += 16;
    }
    while i < pixels {
        out[i * 3] = quantise(r[i].to_f32() * scale);
        out[i * 3 + 1] = quantise(g[i].to_f32() * scale);
        out[i * 3 + 2] = quantise(b[i].to_f32() * scale);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantise8_f32(source: __m256, scale: __m256) -> __m128i {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let max = _mm256_set1_ps(255.0);
    let rounded = _mm256_add_ps(
        _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(source, scale), zero), max),
        _mm256_set1_ps(0.5),
    );
    let lo = _mm_cvttps_epi32(_mm256_castps256_ps128(rounded));
    let hi = _mm_cvttps_epi32(_mm256_extractf128_ps(rounded, 1));
    _mm_packus_epi16(_mm_packus_epi32(lo, hi), _mm_setzero_si128())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave_f32(r: &[f32], g: &[f32], b: &[f32], scale: f32, out: &mut [u8]) {
    use std::arch::x86_64::*;
    let pixels = out.len() / 3;
    let factor = _mm256_set1_ps(scale * 255.0);
    let mut i = 0;
    while i + 16 <= pixels {
        let load =
            |plane: &[f32], offset: usize| unsafe { _mm256_loadu_ps(plane.as_ptr().add(offset)) };
        let r8 = _mm_unpacklo_epi64(
            quantise8_f32(load(r, i), factor),
            quantise8_f32(load(r, i + 8), factor),
        );
        let g8 = _mm_unpacklo_epi64(
            quantise8_f32(load(g, i), factor),
            quantise8_f32(load(g, i + 8), factor),
        );
        let b8 = _mm_unpacklo_epi64(
            quantise8_f32(load(b, i), factor),
            quantise8_f32(load(b, i + 8), factor),
        );
        unsafe { interleave3(r8, g8, b8, out.as_mut_ptr().add(i * 3)) };
        i += 16;
    }
    while i < pixels {
        out[i * 3] = quantise(r[i] * scale);
        out[i * 3 + 1] = quantise(g[i] * scale);
        out[i * 3 + 2] = quantise(b[i] * scale);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_small_frame() {
        let (w, h, t) = (2, 1, 2);
        let (ow, oh) = (2 * w, 2 * h);
        let frames: Vec<u8> = (0..t * w * h * 3).map(|i| (i * 7) as u8).collect();
        let mut planes = Vec::new();
        to_nchw(&frames, w, h, t, &mut planes);
        assert_eq!(planes.len(), t * 3 * w * h);
        let pixel = |f: usize, y: usize, x: usize, c: usize| frames[((f * h + y) * w + x) * 3 + c];
        assert_eq!(
            planes[5 * w * h + 1],
            f16::from_f32(pixel(1, 0, 1, 2) as f32 / 255.0)
        );

        // 2x nearest-neighbour upscale, still planar.
        let mut upscaled = vec![f16::ZERO; t * 3 * ow * oh];
        for f in 0..t {
            for c in 0..3 {
                let src = &planes[(f * 3 + c) * w * h..(f * 3 + c + 1) * w * h];
                let dst = &mut upscaled[(f * 3 + c) * ow * oh..(f * 3 + c + 1) * ow * oh];
                for y in 0..oh {
                    for x in 0..ow {
                        dst[y * ow + x] = src[(y / 2) * w + x / 2];
                    }
                }
            }
        }
        let planes_per_frame = 3 * ow * oh;
        for f in 0..t {
            let mut rgb = vec![0u8; ow * oh * 3];
            planes_to_u8(
                &upscaled[f * planes_per_frame..(f + 1) * planes_per_frame],
                1.0,
                &mut rgb,
            );
            for (i, value) in rgb.iter().enumerate() {
                let p = i / 3;
                let (y, x) = (p / ow, p % ow);
                assert_eq!(*value, pixel(f, y / 2, x / 2, i % 3), "frame {f} pixel {p}");
            }
        }
    }

    #[test]
    fn scalar_interleave_matches_simd() {
        let (w, h) = (5, 3);
        let pixels = w * h;
        let planes: Vec<f16> = (0..3 * pixels)
            .map(|i| f16::from_f32(((i * 37) % 256) as f32 / 255.0))
            .collect();
        let mut vector = vec![0u8; pixels * 3];
        planes_to_u8(&planes, 1.0, &mut vector);
        let mut scalar = vec![0u8; pixels * 3];
        for i in 0..pixels {
            scalar[i * 3] = quantise(planes[i].to_f32());
            scalar[i * 3 + 1] = quantise(planes[pixels + i].to_f32());
            scalar[i * 3 + 2] = quantise(planes[2 * pixels + i].to_f32());
        }
        assert_eq!(vector, scalar);
    }

    #[test]
    fn scalar_matches_avx2() {
        let (w, h, t) = (13, 7, 2);
        let frames: Vec<u8> = (0..t * w * h * 3).map(|i| (i * 31 % 256) as u8).collect();
        let mut scalar = vec![f16::ZERO; t * 3 * w * h];
        to_nchw_scalar(&frames, w, h, t, &mut scalar);
        let mut vector = Vec::new();
        to_nchw(&frames, w, h, t, &mut vector);
        assert_eq!(scalar, vector);
    }
}
