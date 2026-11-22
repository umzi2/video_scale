//! Command line entry point: `video_scale [CONFIG.toml]`.

use anyhow::{Context, Result, bail, ensure};
use std::{
    env,
    ffi::OsStr,
    path::PathBuf,
    process::{Command, ExitCode},
};
use video_scale::{
    config::Settings,
    temporal,
    upscaler::Upscaler,
    video::{EncodeRequest, Output, Video, spawn_decoder, spawn_encoder},
};

const DEFAULT_CONFIG: &str = "video_scale.toml";

const USAGE: &str = "Usage: video_scale [CONFIG.toml]\n\
                     Default: video_scale.toml\n\
                     Paths are relative to CONFIG.";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("video_scale: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let Some(path) = config_path()? else {
        println!("{USAGE}");
        return Ok(());
    };
    let settings = Settings::load(&path)?;
    activate_libraries(&settings)?;
    process(&settings)
}

/// Reads the single optional argument; `None` means help was requested.
fn config_path() -> Result<Option<PathBuf>> {
    let mut args = env::args_os().skip(1);
    let first = args.next();
    if matches!(first.as_deref(), Some(flag) if flag == OsStr::new("--help") || flag == OsStr::new("-h"))
    {
        return Ok(None);
    }
    if args.next().is_some() {
        bail!("{USAGE}");
    }
    Ok(Some(first.map_or_else(
        || PathBuf::from(DEFAULT_CONFIG),
        PathBuf::from,
    )))
}

/// Prepends `lib_dir` to `LD_LIBRARY_PATH` and starts over, because the dynamic
/// loader only reads it at process start.
fn activate_libraries(settings: &Settings) -> Result<()> {
    let Some(dir) = &settings.config.lib_dir else {
        return Ok(());
    };
    let dir = settings.resolve_existing(dir, "lib_dir")?;
    let previous = env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    if env::split_paths(&previous).any(|path| path == dir) {
        return Ok(());
    }
    let mut paths = vec![dir];
    paths.extend(env::split_paths(&previous));
    let error = std::os::unix::process::CommandExt::exec(
        Command::new(env::current_exe()?)
            .args(env::args_os().skip(1))
            .env("LD_LIBRARY_PATH", env::join_paths(paths)?),
    );
    Err(error).context("re-executing with lib_dir")
}

fn process(settings: &Settings) -> Result<()> {
    let config = &settings.config;
    let input = settings.resolve_existing(&config.input, "input")?;
    let model = settings.resolve_existing(&config.model, "model")?;
    let destination = settings.resolve(&config.output);
    ensure!(
        !destination.try_exists()?,
        "output already exists: {}",
        destination.display()
    );
    let output = Output::create(&destination)?;
    let source_video = Video::probe(&input)?;
    let (width, height) = (source_video.width, source_video.height);
    let size = source_video.output_size();
    eprintln!(
        "{width}x{height} -> {}x{}, {} fps; T={}, overlap={}",
        size.0, size.1, source_video.frame_rate, config.chunk_size, config.overlap
    );

    let mut decoder = spawn_decoder(&input)?;
    let mut encoder = spawn_encoder(EncodeRequest {
        source: &input,
        destination: output.staging_path(),
        size,
        frame_rate: &source_video.frame_rate,
        settings: &config.encode,
    })?;
    let mut upscaler = Upscaler::new(
        &model,
        &settings.resolve(&config.cache_dir),
        width,
        height,
        config.chunk_size,
        config.tiling.as_ref().map(|tiling| tiling.max_tile),
    )?;

    let mut source = decoder.stdout()?;
    let mut sink = encoder.stdin()?;
    temporal::process(
        &mut source,
        &mut sink,
        width,
        height,
        config.chunk_size,
        config.overlap,
        |frames, planes| upscaler.infer(frames, planes),
    )?;
    // Close both pipes before waiting so FFmpeg sees end of stream and flushes.
    drop(sink);
    drop(source);
    encoder.wait("FFmpeg encoder")?;
    decoder.wait("FFmpeg decoder")?;

    let saved = output.publish()?;
    eprintln!("Saved {}", saved.display());
    Ok(())
}
