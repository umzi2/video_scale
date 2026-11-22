use super::*;

/// 2x nearest-neighbour upscale of one RGB24 frame, as fp16 planes.
fn upscale(frames: &[u8], width: usize, height: usize) -> Vec<f16> {
    let mut out = Vec::with_capacity(frames.len() * 4);
    for c in 0..3 {
        for y in 0..height * 2 {
            for x in 0..width * 2 {
                let value = frames[((y / 2) * width + x / 2) * 3 + c];
                out.push(f16::from_f32(value as f32 / 255.0));
            }
        }
    }
    out
}

/// Expected RGB24 bytes: the same 2x nearest-neighbour upscale.
fn expected(frames: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(frames.len() * 4);
    for y in 0..height * 2 {
        for x in 0..width * 2 {
            for c in 0..3 {
                out.push(frames[((y / 2) * width + x / 2) * 3 + c]);
            }
        }
    }
    out
}

#[test]
fn schedule_padding_and_exact_frames() {
    for (chunk, overlap) in [(1, 0), (4, 0), (4, 1), (4, 2), (4, 3)] {
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
                            dst.extend(upscale(&window[f * frame..(f + 1) * frame], width, height));
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
                    .flat_map(|f| expected(&frames[f * frame..(f + 1) * frame], width, height))
                    .collect::<Vec<u8>>()
            );
            let expected_calls = if length == 0 {
                0
            } else {
                1 + length.saturating_sub(chunk).div_ceil(stride)
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
        dst.extend(std::iter::repeat_n(f16::from_f32(values[call]), 5 * 4 * 3));
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
                f16::from_f32(if call == active { 1.0 } else { 0.0 }),
                4 * 4 * 3,
            ));
            call += 1;
            Ok(())
        })
        .unwrap();
        for (row, row_weights) in weights.iter().enumerate() {
            let sum: f32 = row_weights.iter().sum();
            let value = (row_weights[active] / sum * 255.0).round() as u8;
            assert!(
                sink[row * 12..(row + 1) * 12].iter().all(|v| *v == value),
                "frame {row}, active window {active}"
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
        for _ in 0..2 {
            for channel in colors[call] {
                dst.extend(std::iter::repeat_n(f16::from_f32(channel), 4));
            }
        }
        call += 1;
        Ok(())
    })
    .unwrap();
    let expected: Vec<u8> = [[0u8, 255, 1], [64, 191, 1], [191, 64, 2]]
        .iter()
        .flat_map(|pixel| (0..4).flat_map(move |_| pixel.iter().copied()))
        .collect();
    assert_eq!(sink, expected);
}

#[test]
fn rejects_partial_frames_and_bad_geometry() {
    let mut sink = Vec::new();
    let mut source: &[u8] = &[0u8; 4];
    assert!(process(&mut source, &mut sink, 1, 1, 2, 0, |_, _| Ok(())).is_err());

    for (width, height, chunk, overlap) in [
        (0, 1, 1, 0),
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (1, 1, 3, 3),
        (1, 1, 3, 4),
    ] {
        let mut sink = Vec::new();
        let mut source: &[u8] = &[];
        assert!(
            process(
                &mut source,
                &mut sink,
                width,
                height,
                chunk,
                overlap,
                |_, _| Ok(())
            )
            .is_err()
        );
    }
}

#[test]
fn rejects_model_output_of_the_wrong_size() {
    let mut sink = Vec::new();
    let mut source: &[u8] = &[0u8; 2 * 3];
    let error = process(&mut source, &mut sink, 1, 1, 2, 0, |_, dst| {
        dst.push(f16::ONE);
        Ok(())
    })
    .unwrap_err();
    assert!(error.to_string().contains("model returned 1 values"));
}

#[test]
fn crossfade_weights_are_complementary() {
    for overlap in 1..5 {
        let chunk = 3 * overlap;
        let stride = chunk - overlap;
        let crossfade = Crossfade::new(chunk, overlap);
        for index in 0..overlap {
            // The newer window sees the frame at `index`, the older one at `stride + index`.
            let newer = crossfade.weight(index, false, true);
            let older = crossfade.weight(stride + index, true, true);
            assert!(
                (newer + older - 1.0).abs() < 1e-6,
                "overlap {overlap}, index {index}: {newer} + {older}"
            );
        }
    }
}
