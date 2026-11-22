//! AVX2/F16C kernels behind [`super`]'s safe API.
//!
//! Every function is `unsafe` because it uses intrinsics that require CPU
//! features; [`available`] is the feature test callers must pass first. Buffers
//! are sized by the safe wrappers before they get here.

use half::f16;

use std::arch::x86_64::*;

/// Whether this CPU can run the kernels below.
#[inline]
pub(super) fn available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c")
}

/// `t` RGB24 frames into `[t, 3, h, w]` fp16; see [`super::to_nchw`].
pub(super) unsafe fn to_nchw(
    frames: &[u8],
    width: usize,
    height: usize,
    t: usize,
    out: &mut [f16],
) {
    // SAFETY: the caller checked AVX2/F16C and sized both buffers for `t` frames.
    unsafe {
        let pixels = width * height;
        let frame = pixels * 3;
        let scale = _mm256_set1_ps(1.0 / 255.0);
        let mask = _mm256_set1_epi32(0xff);
        // Byte offsets of eight consecutive pixels in the interleaved frame.
        let offsets = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);

        for f in 0..t {
            let src = frames.as_ptr().add(f * frame);
            for c in 0..3 {
                let dst = out.as_mut_ptr().add((f * 3 + c) * pixels);
                let mut x = 0;
                // One pixel of slack: the gather always reads four bytes per pixel.
                while x + 9 <= pixels {
                    let packed = _mm256_i32gather_epi32::<1>(src.add(3 * x).cast(), offsets);
                    let shifted = match c {
                        0 => packed,
                        1 => _mm256_srli_epi32::<8>(packed),
                        _ => _mm256_srli_epi32::<16>(packed),
                    };
                    let values = _mm256_and_si256(shifted, mask);
                    let floats = _mm256_mul_ps(_mm256_cvtepi32_ps(values), scale);
                    let halves = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(floats);
                    _mm_storeu_si128(dst.add(x).cast(), halves);
                    x += 8;
                }
                while x < pixels {
                    *dst.add(x) = f16::from_f32(*src.add(x * 3 + c) as f32 / 255.0);
                    x += 1;
                }
            }
        }
    }
}

/// `out = planes * weight` (`add = false`) or `out += planes * weight`.
pub(super) unsafe fn weighted(planes: &[f16], weight: f32, out: &mut [f32], add: bool) {
    // SAFETY: the caller checked AVX2/F16C and sized both buffers equally.
    unsafe {
        let n = out.len();
        let scale = _mm256_set1_ps(weight);
        let mut i = 0;
        while i + 8 <= n {
            let halves = _mm_loadu_si128(planes.as_ptr().add(i).cast());
            let mut values = _mm256_mul_ps(_mm256_cvtph_ps(halves), scale);
            if add {
                values = _mm256_add_ps(_mm256_loadu_ps(out.as_ptr().add(i)), values);
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(i), values);
            i += 8;
        }
        while i < n {
            let value = planes[i].to_f32() * weight;
            out[i] = if add { out[i] + value } else { value };
            i += 1;
        }
    }
}

/// Quantises 16 fp16 pixels into interleaved RGB24; returns the pixels written.
pub(super) unsafe fn interleave_f16(channels: [&[f16]; 3], scale: f32, out: &mut [u8]) -> usize {
    // SAFETY: the caller checked AVX2/F16C; `channels` cover `out.len() / 3` pixels.
    unsafe {
        let pixels = out.len() / 3;
        let factor = _mm256_set1_ps(scale * 255.0);
        let mut i = 0;
        while i + 16 <= pixels {
            let mut quantised = [_mm_setzero_si128(); 3];
            for (channel, plane) in channels.iter().enumerate() {
                let low = _mm_loadu_si128(plane.as_ptr().add(i).cast());
                let high = _mm_loadu_si128(plane.as_ptr().add(i + 8).cast());
                quantised[channel] =
                    _mm_unpacklo_epi64(quantise_f16(low, factor), quantise_f16(high, factor));
            }
            interleave_rgb(quantised, out.as_mut_ptr().add(i * 3));
            i += 16;
        }
        i
    }
}

/// Quantises 16 f32 pixels into interleaved RGB24; returns the pixels written.
pub(super) unsafe fn interleave_f32(channels: [&[f32]; 3], scale: f32, out: &mut [u8]) -> usize {
    // SAFETY: the caller checked AVX2; `channels` cover `out.len() / 3` pixels.
    unsafe {
        let pixels = out.len() / 3;
        let factor = _mm256_set1_ps(scale * 255.0);
        let mut i = 0;
        while i + 16 <= pixels {
            let mut quantised = [_mm_setzero_si128(); 3];
            for (channel, plane) in channels.iter().enumerate() {
                let low = _mm256_loadu_ps(plane.as_ptr().add(i));
                let high = _mm256_loadu_ps(plane.as_ptr().add(i + 8));
                quantised[channel] =
                    _mm_unpacklo_epi64(quantise_f32(low, factor), quantise_f32(high, factor));
            }
            interleave_rgb(quantised, out.as_mut_ptr().add(i * 3));
            i += 16;
        }
        i
    }
}

/// Eight f16 values into the low eight bytes of a vector, clamped and rounded.
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn quantise_f16(values: __m128i, factor: __m256) -> __m128i {
    // SAFETY: the caller checked AVX2/F16C.
    unsafe { quantise_floats(_mm256_mul_ps(_mm256_cvtph_ps(values), factor)) }
}

/// Eight f32 values into the low eight bytes of a vector, clamped and rounded.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn quantise_f32(values: __m256, factor: __m256) -> __m128i {
    // SAFETY: the caller checked AVX2.
    unsafe { quantise_floats(_mm256_mul_ps(values, factor)) }
}

/// Clamps `[0, 1]`-scaled floats to bytes, rounding half away from zero.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn quantise_floats(values: __m256) -> __m128i {
    // SAFETY: the caller checked AVX2; only register operations are involved.
    let clamped = _mm256_min_ps(
        _mm256_max_ps(values, _mm256_setzero_ps()),
        _mm256_set1_ps(255.0),
    );
    let rounded = _mm256_add_ps(clamped, _mm256_set1_ps(0.5));
    let low = _mm_cvttps_epi32(_mm256_castps256_ps128(rounded));
    let high = _mm_cvttps_epi32(_mm256_extractf128_ps(rounded, 1));
    _mm_packus_epi16(_mm_packus_epi32(low, high), _mm_setzero_si128())
}

/// Interleaves 16 bytes per channel into 48 bytes of RGB24.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn interleave_rgb(channels: [__m128i; 3], out: *mut u8) {
    // SAFETY: `out` has room for 48 bytes; masks only select bytes 0..16.
    unsafe {
        // Byte `n` of the output is pixel `n / 3`, channel `n % 3`.
        for chunk in 0..3 {
            let mut bytes = _mm_setzero_si128();
            for (channel, source) in channels.iter().enumerate() {
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
}
