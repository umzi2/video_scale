//! Stage-by-stage benchmarks of the upscaling pipeline.
//!
//! `cargo bench --bench pipeline` runs the CPU stages. Adding a model turns on the
//! TensorRT stage:
//!
//! `BENCH_MODEL=2x_umzi_anime_smovsr_fp16.onnx cargo bench --bench pipeline -- tensorrt`
//!
//! `BENCH_CACHE` overrides the engine cache directory (default `.trt-cache`);
//! `BENCH_T` sets the window size of the TensorRT stage (default 24).
//! Filtering works as usual: `cargo bench --bench pipeline -- convert/rgb24`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use std::{env, path::PathBuf};
use video_scale::{convert, temporal, upscaler::Upscaler};

const W: usize = 640;
const H: usize = 360;
const T: usize = 5;
const OVERLAP: usize = 2;

fn model_path() -> Option<PathBuf> {
    env::var_os("BENCH_MODEL").map(PathBuf::from)
}

fn fixture() -> (Vec<u8>, Vec<f16>) {
    let frame = convert::frame_len(W, H);
    let out_frame = 4 * frame;
    let frames = (0..T * frame).map(|i| (i % 251) as u8).collect();
    let planes = (0..T * out_frame)
        .map(|i| f16::from_f32((i % 1000) as f32 / 1000.0))
        .collect();
    (frames, planes)
}

fn convert_benches(c: &mut Criterion) {
    let (frames, planes) = fixture();
    let frame = convert::frame_len(W, H);
    let out_frame = 4 * frame;
    let mut nchw = Vec::new();
    let mut weighted = vec![0f32; T * out_frame];
    let mut one_frame = vec![0f32; out_frame];
    let mut rgb = vec![0u8; out_frame];

    let mut g = c.benchmark_group("convert");
    // Read one RGB24 byte and write one fp16 value per channel sample.
    g.throughput(Throughput::Bytes((T * frame * 3) as u64));
    g.bench_function("rgb24_to_nchw", |b| {
        b.iter(|| convert::to_nchw(&frames, W, H, T, &mut nchw))
    });

    g.throughput(Throughput::Bytes((4 * planes.len()) as u64));
    g.bench_function("copy_output_planes", |b| {
        b.iter(|| {
            nchw.clear();
            nchw.extend_from_slice(&planes);
        })
    });

    g.throughput(Throughput::Bytes((T * out_frame * 6) as u64));
    g.bench_function("planes_to_weighted_f32", |b| {
        b.iter(|| {
            for f in 0..T {
                let p = &planes[f * out_frame..(f + 1) * out_frame];
                let o = &mut weighted[f * out_frame..(f + 1) * out_frame];
                convert::scale_planes(p, 0.37, o);
            }
        })
    });

    g.throughput(Throughput::Bytes((out_frame * 6) as u64));
    g.bench_function("accumulate_frame", |b| {
        b.iter(|| convert::accumulate_planes(&mut one_frame, &planes[..out_frame], 0.63))
    });

    g.throughput(Throughput::Bytes((out_frame * 3) as u64));
    g.bench_function("planes_to_rgb24", |b| {
        b.iter(|| convert::planes_to_u8(&planes[..out_frame], 1.0, &mut rgb))
    });

    g.throughput(Throughput::Bytes((out_frame * 5) as u64));
    g.bench_function("blend_to_rgb24", |b| {
        b.iter(|| convert::blend_to_u8(&weighted[..out_frame], 0.5, &mut rgb))
    });
    g.finish();
}

fn temporal_benches(c: &mut Criterion) {
    let (_, planes) = fixture();
    let frame = convert::frame_len(W, H);
    let stride = T - OVERLAP;
    let frames = T + 10 * stride;
    let source = vec![0u8; frames * frame];
    let mut sink = Vec::with_capacity(source.len() * 4);

    let mut g = c.benchmark_group("temporal");
    g.throughput(Throughput::Elements(frames as u64));
    g.bench_function("streaming_windows_stub_model", |b| {
        b.iter(|| {
            sink.clear();
            let mut input = &source[..];
            temporal::process(&mut input, &mut sink, W, H, T, OVERLAP, |_, out| {
                out.clear();
                out.extend_from_slice(&planes);
                Ok(())
            })
            .unwrap();
        })
    });
    g.finish();
}

fn tensorrt_bench(c: &mut Criterion) {
    let Some(model) = model_path() else {
        return;
    };
    let cache = env::var_os("BENCH_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".trt-cache"));
    let t = env::var("BENCH_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let frame = convert::frame_len(W, H);
    let frames: Vec<u8> = (0..t * frame).map(|i| (i % 251) as u8).collect();
    let mut upscaler = match Upscaler::new(&model, &cache, W, H, t, None) {
        Ok(upscaler) => upscaler,
        Err(error) => {
            eprintln!("skipping TensorRT bench: {error:#}");
            return;
        }
    };
    let mut out = Vec::new();
    upscaler.infer(&frames, &mut out).unwrap();

    // Explicit frame-rate report: criterion shows per-iteration time, these are the numbers
    // that describe a video pass (frames/s and the cost of a 2160-frame clip).
    let iters = 5;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        upscaler.infer(&frames, &mut out).unwrap();
    }
    let window_seconds = start.elapsed().as_secs_f64() / iters as f64;
    let per_frame = window_seconds / t as f64;
    println!(
        "\n[T={t}] window {:.2} ms, {:.3} ms per frame, {:.1} frames/s, 2160 frames (no overlap) ~ {:.1} s",
        window_seconds * 1e3,
        per_frame * 1e3,
        1.0 / per_frame,
        per_frame * 2160.0
    );

    let mut g = c.benchmark_group("tensorrt");
    // One warm call dominates the time; a handful of samples keeps the suite short.
    g.sample_size(10);
    g.throughput(Throughput::Elements(t as u64));
    g.bench_function(format!("infer_{t}_frames"), |b| {
        b.iter(|| upscaler.infer(&frames, &mut out).unwrap())
    });
    g.finish();
}

criterion_group!(benches, convert_benches, temporal_benches, tensorrt_bench);
criterion_main!(benches);
