use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Child, Command, ExitCode, Stdio},
};
use video_scale::{temporal, upscaler::Upscaler};

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
