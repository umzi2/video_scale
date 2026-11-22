use super::*;

/// Reference `[t, 3, h, w]` conversion without the vector kernels.
fn nchw_reference(frames: &[u8], width: usize, height: usize, t: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; t * 3 * width * height];
    for f in 0..t {
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let byte = frames[((f * height + y) * width + x) * 3 + c];
                    out[(f * 3 + c) * width * height + y * width + x] =
                        f16::from_f32(byte as f32 / 255.0);
                }
            }
        }
    }
    out
}

#[test]
fn nchw_matches_reference() {
    for (width, height, t) in [(13, 7, 2), (1, 1, 1), (16, 3, 1), (5, 3, 3)] {
        let frames: Vec<u8> = (0..t * width * height * 3)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        let mut planes = Vec::new();
        to_nchw(&frames, width, height, t, &mut planes);
        assert_eq!(planes, nchw_reference(&frames, width, height, t));
    }
}

#[test]
fn pixels_round_trip_through_planes() {
    let (width, height) = (3, 2);
    let frames: Vec<u8> = (0..width * height * 3).map(|i| (i * 17) as u8).collect();
    let mut planes = Vec::new();
    to_nchw(&frames, width, height, 1, &mut planes);

    // 2x nearest-neighbour upscale, still planar.
    let (ow, oh) = (2 * width, 2 * height);
    let mut upscaled = vec![f16::ZERO; 3 * ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..3 {
                upscaled[c * ow * oh + y * ow + x] =
                    planes[c * width * height + (y / 2) * width + x / 2];
            }
        }
    }

    let mut rgb = vec![0u8; ow * oh * 3];
    planes_to_u8(&upscaled, 1.0, &mut rgb);
    for (i, value) in rgb.iter().enumerate() {
        let pixel = i / 3;
        let (y, x) = (pixel / ow, pixel % ow);
        let source = ((y / 2) * width + x / 2) * 3 + i % 3;
        assert_eq!(*value, frames[source], "pixel {pixel}");
    }
}

#[test]
fn interleave_matches_scalar() {
    let pixels = 5 * 3;
    let planes: Vec<f16> = (0..3 * pixels)
        .map(|i| f16::from_f32(((i * 37) % 256) as f32 / 255.0))
        .collect();
    let mut vector = vec![0u8; pixels * 3];
    planes_to_u8(&planes, 1.0, &mut vector);
    for i in 0..pixels {
        assert_eq!(vector[i * 3], quantise(planes[i].to_f32()));
        assert_eq!(vector[i * 3 + 1], quantise(planes[pixels + i].to_f32()));
        assert_eq!(vector[i * 3 + 2], quantise(planes[2 * pixels + i].to_f32()));
    }
}

#[test]
fn blends_and_quantises_out_of_range_values() {
    let planes = [f16::from_f32(-0.5), f16::from_f32(0.0), f16::from_f32(1.5)];
    let mut accumulated = [0.25f32, 0.25, 0.25];
    accumulate_planes(&mut accumulated, &planes, 0.5);
    assert_eq!(accumulated, [0.0, 0.25, 1.0]);

    let mut overwritten = [f32::NAN; 3];
    scale_planes(&planes, 2.0, &mut overwritten);
    assert_eq!(overwritten, [-1.0, 0.0, 3.0]);

    // Three 3-pixel planes: red below 0, green at 0, blue above 1.
    let planes = [
        f16::from_f32(-0.25),
        f16::from_f32(-0.25),
        f16::from_f32(-0.25),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.25),
        f16::from_f32(1.25),
        f16::from_f32(1.25),
    ];
    let mut rgb = vec![0u8; 9];
    planes_to_u8(&planes, 1.0, &mut rgb);
    assert_eq!(rgb, [0, 0, 255, 0, 0, 255, 0, 0, 255]);
}

#[test]
#[should_panic(expected = "expected 2 RGB24 frames")]
fn rejects_wrong_frame_count() {
    let frames = vec![0u8; 3 * 4 * 4];
    to_nchw(&frames, 4, 4, 2, &mut Vec::new());
}
