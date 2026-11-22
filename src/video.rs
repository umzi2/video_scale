//! FFmpeg plumbing: probing the source, streaming RGB24 frames, encoding the result.

use crate::config::Encode;
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};
use tempfile::NamedTempFile;

/// Source properties the pipeline needs.
pub struct Video {
    pub width: usize,
    pub height: usize,
    pub frame_rate: String,
}

impl Video {
    /// Asks `ffprobe` for the displayed geometry and frame rate.
    pub fn probe(input: &Path) -> Result<Self> {
        let output = Command::new("ffprobe")
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
        if !output.status.success() {
            bail!(
                "ffprobe failed for {}: {}",
                input.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let probe: Probe = serde_json::from_slice(&output.stdout).context("parsing ffprobe")?;
        let stream = probe
            .streams
            .into_iter()
            .next()
            .context("input has no video stream")?;
        let (mut width, mut height) = (stream.width as usize, stream.height as usize);
        ensure!(
            width > 0 && height > 0,
            "invalid input dimensions {width}x{height}"
        );
        validate_frame_rate(&stream.avg_frame_rate)?;
        // FFmpeg bakes the rotation into the frames, so the model sees the displayed geometry.
        if rotation_swaps(&stream.side_data_list)? {
            std::mem::swap(&mut width, &mut height);
        }
        Ok(Self {
            width,
            height,
            frame_rate: stream.avg_frame_rate,
        })
    }

    /// Model output size: twice every dimension.
    pub fn output_size(&self) -> (usize, usize) {
        (self.width * 2, self.height * 2)
    }
}

#[derive(Deserialize)]
struct Probe {
    streams: Vec<Stream>,
}

#[derive(Deserialize)]
struct Stream {
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

fn validate_frame_rate(rate: &str) -> Result<()> {
    let (numerator, denominator) = rate
        .split_once('/')
        .with_context(|| format!("missing frame rate in {rate:?}"))?;
    ensure!(
        numerator.parse::<u64>().is_ok_and(|n| n > 0)
            && denominator.parse::<u64>().is_ok_and(|d| d > 0),
        "invalid frame rate {rate:?}"
    );
    Ok(())
}

/// A quarter-turn rotation means width and height swap in display order.
fn rotation_swaps(side_data: &[SideData]) -> Result<bool> {
    let Some(angle) = side_data.iter().find_map(|data| data.rotation) else {
        return Ok(false);
    };
    ensure!(angle % 90 == 0, "unsupported video rotation: {angle}");
    Ok(angle.rem_euclid(180) == 90)
}

/// A running FFmpeg process, killed if the pipeline unwinds.
pub struct Stage(Child);

impl Stage {
    /// Takes the read end of a decoder.
    pub fn stdout(&mut self) -> Result<ChildStdout> {
        self.0.stdout.take().context("decoder has no stdout pipe")
    }

    /// Takes the write end of an encoder.
    pub fn stdin(&mut self) -> Result<ChildStdin> {
        self.0.stdin.take().context("encoder has no stdin pipe")
    }

    /// Waits for the process and fails on a non-zero exit code.
    pub fn wait(&mut self, name: &str) -> Result<()> {
        let status = self.0.wait()?;
        ensure!(status.success(), "{name} exited with {status}");
        Ok(())
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// FFmpeg with the flags shared by both ends of the pipe.
fn ffmpeg() -> Command {
    let mut command = Command::new("ffmpeg");
    command.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-xerror"]);
    command
}

/// Starts FFmpeg decoding `input` to RGB24 frames on stdout.
pub fn spawn_decoder(input: &Path) -> Result<Stage> {
    let mut command = ffmpeg();
    command
        .arg("-i")
        .arg(input)
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
        .stdout(Stdio::piped());
    Ok(Stage(
        command.spawn().context("starting the FFmpeg decoder")?,
    ))
}

/// Everything needed to start an encoder.
pub struct EncodeRequest<'a> {
    /// Input video, used for the audio track.
    pub source: &'a Path,
    /// Where FFmpeg writes; the caller publishes it afterwards.
    pub destination: &'a Path,
    pub size: (usize, usize),
    pub frame_rate: &'a str,
    pub settings: &'a Encode,
}

/// Starts FFmpeg reading RGB24 frames from stdin and encoding them.
pub fn spawn_encoder(request: EncodeRequest<'_>) -> Result<Stage> {
    let EncodeRequest {
        source,
        destination,
        size,
        frame_rate,
        settings,
    } = request;
    let with_audio = settings.audio != "none";

    let mut command = ffmpeg();
    command
        .args(["-y", "-stats", "-stats_period", "5"])
        .args(["-f", "rawvideo", "-pixel_format", "rgb24"])
        .arg("-video_size")
        .arg(format!("{}x{}", size.0, size.1))
        .arg("-framerate")
        .arg(frame_rate)
        .args(["-i", "pipe:0"]);
    if with_audio {
        command.arg("-i").arg(source);
    }
    command
        .args(["-map", "0:v:0", "-c:v"])
        .arg(&settings.codec)
        .arg("-pix_fmt")
        .arg(&settings.pixel_format);
    if with_audio {
        command
            .arg("-map")
            .arg("1:a?")
            .arg("-c:a")
            .arg(&settings.audio);
    }
    for (name, value) in &settings.options {
        ensure!(
            valid_option_name(name),
            "invalid FFmpeg option name: {name}"
        );
        command.arg(format!("-{name}")).arg(value);
    }
    command.arg(destination).stdin(Stdio::piped());
    Ok(Stage(
        command.spawn().context("starting the FFmpeg encoder")?,
    ))
}

/// FFmpeg option names are passed as `-name`, so they must be safe flag names.
fn valid_option_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('-')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_:-".contains(&byte))
}

/// The destination, written through a temporary sibling.
///
/// A failed or interrupted run leaves the real path untouched.
pub struct Output {
    file: NamedTempFile,
    target: PathBuf,
}

impl Output {
    pub fn create(target: &Path) -> Result<Self> {
        let parent = target.parent().context("output needs a parent directory")?;
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        let extension = target
            .extension()
            .and_then(|extension| extension.to_str())
            .context("output needs a container extension such as .mp4 or .mkv")?;
        let file = tempfile::Builder::new()
            .prefix(".video-scale-")
            .suffix(&format!(".{extension}"))
            .tempfile_in(parent)
            .context("creating a temporary output file")?;
        Ok(Self {
            file,
            target: target.to_owned(),
        })
    }

    /// Path FFmpeg should write to.
    pub fn staging_path(&self) -> &Path {
        self.file.path()
    }

    /// Moves the finished file into place, never replacing an existing one.
    pub fn publish(self) -> Result<PathBuf> {
        let Self { file, target } = self;
        file.persist_noclobber(&target)
            .map_err(|error| error.error)
            .with_context(|| format!("saving {}", target.display()))?;
        Ok(target)
    }
}
