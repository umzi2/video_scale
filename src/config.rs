//! One `*.toml` file describes a run: what to read, what to write, how to encode.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// A parsed config plus the directory its relative paths resolve against.
pub struct Settings {
    pub config: Config,
    base: PathBuf,
}

impl Settings {
    /// Reads and validates `path`; relative paths are resolved against its directory.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let config: Config =
            toml::from_str(&text).with_context(|| format!("invalid {}", path.display()))?;
        config.validate()?;
        let base = path
            .canonicalize()
            .with_context(|| format!("resolving {}", path.display()))?
            .parent()
            .context("config path has no directory")?
            .to_owned();
        Ok(Self { config, base })
    }

    /// Resolves a path from the config, failing if it does not exist.
    pub fn resolve_existing(&self, path: &Path, what: &str) -> Result<PathBuf> {
        self.base
            .join(path)
            .canonicalize()
            .with_context(|| format!("{what} not found: {}", self.base.join(path).display()))
    }

    /// Resolves a path that may not exist yet (the output file, the cache).
    pub fn resolve(&self, path: &Path) -> PathBuf {
        self.base.join(path)
    }
}

/// The `*.toml` layout.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub input: PathBuf,
    pub output: PathBuf,
    pub model: PathBuf,
    pub cache_dir: PathBuf,
    /// Frames per model call.
    pub chunk_size: usize,
    /// Frames shared by neighbouring windows; 0 disables cross-fading.
    pub overlap: usize,
    /// Directory with CUDA/cuDNN/TensorRT libraries, if they are not installed system-wide.
    #[serde(default)]
    pub lib_dir: Option<PathBuf>,
    pub encode: Encode,
}

impl Config {
    fn validate(&self) -> Result<()> {
        if self.chunk_size == 0 {
            bail!("chunk_size must be at least 1");
        }
        ensure!(
            self.overlap < self.chunk_size,
            "require 0 <= overlap < chunk_size, got {} >= {}",
            self.overlap,
            self.chunk_size
        );
        if self.encode.codec.is_empty() {
            bail!("encode.codec must not be empty");
        }
        if self.encode.pixel_format.is_empty() {
            bail!("encode.pixel_format must not be empty");
        }
        if self.encode.audio.is_empty() {
            bail!("encode.audio must not be empty (use \"none\" to drop audio)");
        }
        Ok(())
    }
}

/// FFmpeg encoder settings.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Encode {
    /// FFmpeg video codec, for example `libx264`, `libx265` or `ffv1`.
    pub codec: String,
    /// Output pixel format, for example `yuv420p` or `bgr0`.
    pub pixel_format: String,
    /// `copy`, `none`, or an audio codec name.
    pub audio: String,
    /// Extra FFmpeg flags for this codec and container, without leading dashes.
    #[serde(default)]
    pub options: BTreeMap<String, String>,
}
