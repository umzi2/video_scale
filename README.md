# video_scale

CLI for 2x video upscaling with a recurrent VSR ONNX model (`2x_umzi_anime_smovsr_fp16.onnx`)
executed through ONNX Runtime's TensorRT EP (fp16).

```
video_scale [CONFIG.toml]      # default: video_scale.toml
```

Paths inside the config are resolved relative to the config file. Video is decoded and encoded by
FFmpeg over pipes (RGB24 both ways); audio is copied by default.

## Config

```toml
input = "test_video.mp4"
output = "test_video_2x.mp4"
model = "2x_umzi_anime_smovsr_fp16.onnx"
chunk_size = 5          # T frames per model call
overlap = 2             # 0 <= overlap < chunk_size
cache_dir = ".trt-cache" # TensorRT engine cache, keyed by GPU/model/shape/runtime
# lib_dir = "/path/to/cuda-cudnn-tensorrt/libs"  # else the runtime must be findable via LD_LIBRARY_PATH

[encode]
codec = "libx264"
pixel_format = "yuv420p"
audio = "copy"          # "none" drops audio
[encode.options]        # any extra FFmpeg flags
crf = "18"
```

## Chunking

Frames go through the model in windows of `chunk_size` frames with stride
`chunk_size - overlap`. Overlapping frames are **not dropped**: they are blended with a sin²
cross-fade (`sin²(π/2 · k/(overlap+1))` over the overlap), so every input frame contributes.
`overlap = 0` disables blending. Short inputs are zero-padded up to one window and trimmed back.

Blending happens on the model's fp16 output planes; only the finished frame is quantised to RGB24,
so no precision is lost while accumulating.

## Requirements

- FFmpeg/ffprobe on `PATH`.
- NVIDIA GPU + TensorRT 11 runtime. `ort` downloads its prebuilt CUDA/cuDNN/TensorRT set on first
  run; point `lib_dir` at it or put the libraries on `LD_LIBRARY_PATH`.
- The first run per shape compiles the engine (tens of seconds); later runs load it from `cache_dir`.

## Benchmarks

Stage-by-stage criterion benchmarks (640x360, T=5, overlap=2):

```
cargo bench --bench pipeline                          # CPU stages
BENCH_MODEL=2x_umzi_anime_smovsr_fp16.onnx \
    cargo bench --bench pipeline -- tensorrt           # + TensorRT inference (24-frame window)
BENCH_T=5 BENCH_MODEL=... cargo bench --bench pipeline -- tensorrt   # other window size
```
