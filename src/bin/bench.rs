//! Stage-by-stage benchmark of the upscaling pipeline.
//!
//! cargo run --release --bin bench -- --width 640 --height 360 --chunk 5 \
//!     --iters 30 --model 2x_umzi_anime_smovsr_fp16.onnx

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant},
};
use video_scale::{convert, temporal, upscaler::Upscaler};

struct Args {
    width: usize,
    height: usize,
    chunk: usize,
    overlap: usize,
    iters: usize,
    model: Option<PathBuf>,
    cache: PathBuf,
}

fn parse() -> Args {
    let mut args = Args {
        width: 640,
        height: 360,
        chunk: 5,
        overlap: 2,
        iters: 50,
        model: None,
        cache: ".trt-cache".into(),
    };
    let mut it = env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .unwrap_or_else(|| panic!("{flag} needs a value"))
                .parse()
                .unwrap_or_else(|_| panic!("bad value for {flag}"))
        };
        match flag.as_str() {
            "--width" => args.width = value(),
            "--height" => args.height = value(),
            "--chunk" => args.chunk = value(),
            "--overlap" => args.overlap = value(),
            "--iters" => args.iters = value(),
            "--model" => args.model = Some(it.next().expect("--model needs a path").into()),
            "--cache" => args.cache = it.next().expect("--cache needs a path").into(),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(args.overlap < args.chunk);
    args
}

fn timed(iters: usize, mut run: impl FnMut()) -> Duration {
    // Warm up caches and branch predictors before the measured loop.
    run();
    let start = Instant::now();
    for _ in 0..iters {
        run();
    }
    start.elapsed() / iters as u32
}

fn report(name: &str, per: Duration, bytes: usize, frames: usize) {
    let ms = per.as_secs_f64() * 1e3;
    let gbs = bytes as f64 / per.as_secs_f64() / 1e9;
    let fps = frames as f64 / per.as_secs_f64();
    println!("{name:<32} {ms:9.3} ms  {gbs:6.2} GB/s  {fps:8.1} fps");
}

fn main() -> anyhow::Result<()> {
    let args = parse();
    let (w, h, t) = (args.width, args.height, args.chunk);
    let frame = convert::frame_len(w, h);
    let (ow, oh) = (2 * w, 2 * h);
    let out_frame = ow * oh * 3;
    println!(
        "window {w}x{h} -> {ow}x{oh}, T={}, overlap={}, iters={}",
        t, args.overlap, args.iters
    );

    let frames: Vec<u8> = (0..t * frame).map(|i| (i % 251) as u8).collect();
    let planes: Vec<half::f16> = (0..t * out_frame)
        .map(|i| half::f16::from_f32((i % 1000) as f32 / 1000.0))
        .collect();
    let mut scratch_f16: Vec<half::f16> = Vec::new();
    let mut scratch_f32 = vec![0.0f32; t * out_frame];
    let mut rgb = vec![0u8; out_frame];
    let mut sink = Vec::new();
    let mut window = planes.clone();

    let to = timed(args.iters, || {
        convert::to_nchw(&frames, w, h, t, &mut scratch_f16)
    });
    report("rgb24 -> nchw fp16", to, t * frame * 3 + t * frame, t);

    let copy = timed(args.iters, || {
        scratch_f16.clear();
        scratch_f16.extend_from_slice(&planes);
    });
    report("copy model output planes", copy, 2 * t * out_frame * 2, t);

    let weights = timed(args.iters, || {
        for f in 0..t {
            let p = &planes[f * out_frame..(f + 1) * out_frame];
            let o = &mut scratch_f32[f * out_frame..(f + 1) * out_frame];
            convert::store_planes(p, 0.37, o);
        }
    });
    report("fp16 planes -> weighted f32", weights, t * out_frame * 6, t);

    let accumulate = timed(args.iters, || {
        convert::accumulate_planes(&mut scratch_f32, &planes, 0.63);
    });
    report("accumulate one frame", accumulate, out_frame * 6, 1);

    let quantise = timed(args.iters, || convert::planes_to_u8(&planes, 1.0, &mut rgb));
    report("fp16 planes -> rgb24", quantise, out_frame * 3, 1);

    let blend = timed(args.iters, || {
        convert::blend_to_u8(&scratch_f32[..out_frame], 0.5, &mut rgb)
    });
    report("f32 planes -> rgb24", blend, out_frame * 5, 1);

    // Whole stream path with a stub model: windowing, blending and quantisation.
    let stride = t - args.overlap;
    let source: Vec<u8> = vec![0u8; (t + 10 * stride) * frame];
    sink.reserve(source.len() * 4);
    let process = timed(args.iters, || {
        sink.clear();
        let mut input = &source[..];
        temporal::process(&mut input, &mut sink, w, h, t, args.overlap, |_, out| {
            out.clear();
            out.extend_from_slice(&planes);
            Ok(())
        })
        .unwrap();
    });
    report(
        "streaming windows (stub model)",
        process,
        0,
        t + 10 * stride,
    );

    if let Some(model) = args.model {
        let mut upscaler = Upscaler::new(&model, &args.cache, w, h, t)?;
        upscaler.infer(&frames, &mut window)?;
        let infer = timed(args.iters, || {
            upscaler.infer(&frames, &mut window).unwrap();
        });
        report("tensorrt inference", infer, t * out_frame * 4, t);
        let cpu = process.as_secs_f64() / (t + 10 * stride) as f64;
        let per_frame = cpu + infer.as_secs_f64() / t as f64;
        println!(
            "\nper input frame: {:.3} ms (cpu+trt), projected 2160 frames: {:.1} s",
            per_frame * 1e3,
            per_frame * 2160.0
        );
    }
    Ok(())
}
