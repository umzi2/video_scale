mod temporal;

use anyhow::{Context, Result, bail, ensure};
use half::f16;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, Stdio},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    model: PathBuf,
    cache_dir: PathBuf,
    chunk_size: usize,
    overlap: usize,
    /// Directory with CUDA/cuDNN/TensorRT libraries, if they are not installed system-wide.
    #[serde(default)]
    lib_dir: Option<PathBuf>,
    encode: Encode,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Encode {
    codec: String,
    pixel_format: String,
    audio: String,
    #[serde(default)]
    options: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct Video {
    width: u32,
    height: u32,
    avg_frame_rate: String,
    #[serde(default)]
    side_data_list: Vec<SideData>,
}

#[derive(Deserialize)]
struct SideData {
    rotation: Option<i32>,
}

fn probe(input: &Path) -> Result<Video> {
    #[derive(Deserialize)]
    struct Probe {
        streams: Vec<Video>,
    }
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:stream_side_data=rotation",
            "-of",
            "json",
        ])
        .arg(input)
        .output()
        .context("starting ffprobe")?;
    ensure!(
        out.status.success(),
        "ffprobe: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut v = serde_json::from_slice::<Probe>(&out.stdout)?
        .streams
        .into_iter()
        .next()
        .context("input has no video stream")?;
    ensure!(v.width > 0 && v.height > 0, "invalid input dimensions");
    let (n, d) = v
        .avg_frame_rate
        .split_once('/')
        .context("missing frame rate")?;
    ensure!(
        n.parse::<u64>()? > 0 && d.parse::<u64>()? > 0,
        "invalid frame rate"
    );
    if let Some(angle) = v.side_data_list.iter().find_map(|s| s.rotation) {
        ensure!(angle % 90 == 0, "unsupported video rotation: {angle}");
        if angle.rem_euclid(180) == 90 {
            std::mem::swap(&mut v.width, &mut v.height);
        }
    }
    Ok(v)
}

/// ONNX Runtime 1.31 rejects models stamped with an unreleased opset (28 here).
/// Every operator in this model has a stable schema far below opset 26, so the
/// `ai.onnx` version is rewritten in memory before loading.
fn model_bytes(model: &Path) -> Result<Vec<u8>> {
    let mut bytes = fs::read(model).with_context(|| format!("reading {}", model.display()))?;
    let mut pos = 0;
    let mut patched = 0;
    while pos < bytes.len() {
        let (key, next) = varint(&bytes, pos)?;
        let (field, wire) = (key >> 3, key & 7);
        match wire {
            0 => pos = varint(&bytes, next)?.1,
            1 => pos = next + 8,
            5 => pos = next + 4,
            2 => {
                let (len, start) = varint(&bytes, next)?;
                let end = start + len as usize;
                ensure!(end <= bytes.len(), "truncated ONNX model");
                if field == 8 && patch_opset(&mut bytes[start..end])? {
                    patched += 1;
                }
                pos = end;
            }
            _ => bail!("unsupported ONNX field {field}"),
        }
    }
    ensure!(
        patched == 1,
        "expected one ai.onnx opset import, patched {patched}"
    );
    Ok(bytes)
}

fn varint(bytes: &[u8], mut pos: usize) -> Result<(u64, usize)> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(pos).context("truncated ONNX varint")?;
        value |= u64::from(byte & 0x7f) << shift;
        pos += 1;
        if byte & 0x80 == 0 {
            return Ok((value, pos));
        }
    }
    bail!("invalid ONNX varint");
}

/// Rewrite `OperatorSetIdProto`'s version for `ai.onnx` if it is newer than 26.
fn patch_opset(entry: &mut [u8]) -> Result<bool> {
    let (mut domain, mut version) = (None, None);
    let mut pos = 0;
    while pos < entry.len() {
        let (key, next) = varint(entry, pos)?;
        let (field, wire) = (key >> 3, key & 7);
        match (field, wire) {
            (1, 2) => {
                let (len, start) = varint(entry, next)?;
                domain = Some(&entry[start..start + len as usize]);
                pos = start + len as usize;
            }
            (2, 0) => {
                let (value, end) = varint(entry, next)?;
                version = Some((value, next, end));
                pos = end;
            }
            (_, 0) => pos = varint(entry, next)?.1,
            (_, 2) => {
                let (len, start) = varint(entry, next)?;
                pos = start + len as usize;
            }
            (_, 5) => pos = next + 4,
            _ => bail!("unsupported ONNX opset field {field}"),
        }
    }
    let Some((value, start, end)) = version else {
        return Ok(false);
    };
    let is_onnx = domain.is_none_or(|d| d.is_empty() || d == b"ai.onnx");
    if !is_onnx || value <= 26 {
        return Ok(false);
    }
    ensure!(
        end - start == 1,
        "unexpected opset version encoding: {value}"
    );
    entry[start] = 26;
    Ok(true)
}

/// TensorRT (falling back to CUDA) session for one fixed window size.
struct Upscaler {
    session: Session,
    width: usize,
    height: usize,
    chunk: usize,
}

impl Upscaler {
    fn new(model: &Path, cache: &Path, width: usize, height: usize, chunk: usize) -> Result<Self> {
        let env = ort::init().with_name("video_scale").build()?;
        fs::create_dir_all(cache)?;
        let cache = cache.to_string_lossy();
        // One profile point: the window size is fixed for the whole run.
        let shape = format!("input:1x{chunk}x3x{height}x{width}");
        let session = Session::builder(&env)?
            .with_execution_providers([
                ep::TensorRT::default()
                    .with_fp16(true)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&*cache)
                    .with_timing_cache(true)
                    .with_timing_cache_path(&*cache)
                    .with_profile_min_shapes(shape.clone())
                    .with_profile_opt_shapes(shape.clone())
                    .with_profile_max_shapes(shape.clone())
                    .build()
                    .error_on_failure(),
                ep::CUDA::default().build().error_on_failure(),
            ])
            .map_err(|e| anyhow::anyhow!("registering execution providers: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("setting optimization level: {e}"))?
            .commit_from_memory(&model_bytes(model)?)
            .map_err(|e| anyhow::anyhow!("loading {model:?}: {e}"))?;
        Ok(Self {
            session,
            width,
            height,
            chunk,
        })
    }

    /// `frames` is `chunk` RGB24 frames; splits the result into `chunk` RGB frames.
    fn infer(&mut self, frames: &[u8], out: &mut Vec<f32>) -> Result<()> {
        let (w, h, t) = (self.width, self.height, self.chunk);
        let frame = w * h * 3;
        ensure!(frames.len() == t * frame, "infer needs exactly {t} frames");
        // NHWC uint8 -> NCHW fp16 in [0,1], the model's input layout.
        let mut data = Vec::with_capacity(t * 3 * h * w);
        for f in 0..t {
            let plane = &frames[f * frame..(f + 1) * frame];
            for c in 0..3 {
                data.extend(
                    plane
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| f16::from_f32(p[c] as f32 / 255.0)),
                );
            }
        }
        let input = Tensor::from_array(([1usize, t, 3, h, w], data))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f16>()?;
        let expected = [1, t, 3, 2 * h, 2 * w];
        let dims: Vec<i64> = shape.iter().copied().collect();
        ensure!(
            dims == expected.iter().map(|d| *d as i64).collect::<Vec<_>>(),
            "model output {shape:?} does not match {expected:?}"
        );
        let (oh, ow) = (2 * h, 2 * w);
        out.clear();
        out.resize(t * oh * ow * 3, 0.0);
        for f in 0..t {
            for c in 0..3 {
                for y in 0..oh {
                    for x in 0..ow {
                        out[((f * oh + y) * ow + x) * 3 + c] =
                            values[((f * 3 + c) * oh + y) * ow + x].to_f32();
                    }
                }
            }
        }
        Ok(())
    }
}

// Reap the other pipeline stages even when a later spawn or wait fails.
struct Stage(Child);

impl Drop for Stage {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Stage {
    fn wait(&mut self, name: &str) -> Result<()> {
        let status = self.0.wait()?;
        ensure!(status.success(), "{name} exited with {status}");
        Ok(())
    }
}

fn ffmpeg() -> Command {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-xerror"]);
    cmd
}

fn process(path: &Path) -> Result<()> {
    let cfg: Config = toml::from_str(
        &fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .context("invalid config")?;
    ensure!(
        cfg.overlap < cfg.chunk_size,
        "require 0 <= overlap < chunk_size"
    );
    let base = path.canonicalize()?;
    let base = base.parent().context("missing config directory")?;
    if let Some(dir) = &cfg.lib_dir {
        let dir = base.join(dir).canonicalize().context("lib_dir not found")?;
        // The loader caches LD_LIBRARY_PATH at startup, so append it and start over.
        let previous = env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        let already = env::split_paths(&previous).any(|p| p == dir);
        if !already {
            let mut paths = vec![dir];
            paths.extend(env::split_paths(&previous));
            let mut cmd = Command::new(env::current_exe()?);
            cmd.args(env::args_os().skip(1))
                .env("LD_LIBRARY_PATH", env::join_paths(paths)?);
            let error = std::os::unix::process::CommandExt::exec(&mut cmd);
            return Err(error).context("re-executing with lib_dir");
        }
    }
    let input = base
        .join(cfg.input)
        .canonicalize()
        .context("input not found")?;
    let model = base
        .join(cfg.model)
        .canonicalize()
        .context("model not found")?;
    let output = base.join(cfg.output);
    ensure!(
        !output.try_exists()?,
        "output already exists: {}",
        output.display()
    );
    let ext = output
        .extension()
        .and_then(|x| x.to_str())
        .context("output needs a container extension")?;
    let video = probe(&input)?;
    let (width, height) = (video.width as usize, video.height as usize);
    let size = format!("{}x{}", width * 2, height * 2);
    let parent = output.parent().context("output needs a parent directory")?;
    fs::create_dir_all(parent)?;
    // Encode beside the destination so a failed run never publishes a partial video.
    let tmp = tempfile::Builder::new()
        .prefix(".video-scale-")
        .suffix(&format!(".{ext}"))
        .tempfile_in(parent)?;
    eprintln!(
        "{width}x{height} -> {size}, {} fps; T={}, overlap={}",
        video.avg_frame_rate, cfg.chunk_size, cfg.overlap
    );

    let mut decoder = Stage(
        ffmpeg()
            .arg("-i")
            .arg(&input)
            .args([
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .spawn()
            .context("starting FFmpeg decoder")?,
    );
    let mut enc = ffmpeg();
    enc.args([
        "-y",
        "-stats",
        "-stats_period",
        "5",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        &size,
        "-framerate",
        &video.avg_frame_rate,
        "-i",
        "pipe:0",
    ]);
    if cfg.encode.audio != "none" {
        enc.arg("-i").arg(&input);
    }
    enc.args([
        "-map",
        "0:v:0",
        "-c:v",
        &cfg.encode.codec,
        "-pix_fmt",
        &cfg.encode.pixel_format,
    ]);
    if cfg.encode.audio != "none" {
        enc.args(["-map", "1:a?", "-c:a", &cfg.encode.audio]);
    }
    for (name, value) in cfg.encode.options {
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_:-".contains(&b))
                && !name.starts_with('-'),
            "invalid FFmpeg option name: {name}"
        );
        enc.arg(format!("-{name}")).arg(value);
    }
    let mut encoder = Stage(
        enc.arg(tmp.path())
            .stdin(Stdio::piped())
            .spawn()
            .context("starting FFmpeg encoder")?,
    );

    let mut upscaler = Upscaler::new(
        &model,
        &base.join(cfg.cache_dir),
        width,
        height,
        cfg.chunk_size,
    )?;
    let mut source = decoder.0.stdout.take().unwrap();
    let mut sink = encoder.0.stdin.take().unwrap();
    temporal::process(
        &mut source,
        &mut sink,
        width,
        height,
        cfg.chunk_size,
        cfg.overlap,
        |frames, out| upscaler.infer(frames, out),
    )?;
    drop(sink);
    drop(source);
    encoder.wait("FFmpeg encoder")?;
    decoder.wait("FFmpeg decoder")?;
    tmp.persist_noclobber(&output)?;
    eprintln!("Saved {}", output.display());
    Ok(())
}

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let path = args.next().unwrap_or_else(|| "video_scale.toml".into());
    if path == "-h" || path == "--help" {
        println!(
            "Usage: video_scale [CONFIG.toml]\nDefault: video_scale.toml\nPaths are relative to CONFIG."
        );
        return Ok(());
    }
    if args.next().is_some() {
        bail!("usage: video_scale [CONFIG.toml]");
    }
    process(Path::new(&path))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("video_scale: {err:#}");
            ExitCode::FAILURE
        }
    }
}
