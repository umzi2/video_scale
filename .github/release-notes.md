## video_scale {{version}}

CLI-апскейлер видео в 2 раза на рекуррентной VSR-модели через ONNX Runtime и TensorRT.

### Что в архиве

- `video_scale` — собранный бинарник (x86_64-linux, glibc 2.38+ и libstdc++ из GCC 13+);
- `README.md` — описание, конфиг, тайлинг, бенчмарки;
- `video_scale.toml` и `lossless.toml` — готовые примеры конфигов.

### Установка и запуск

```bash
tar xzf video_scale-{{version}}-x86_64-linux.tar.gz
cd video_scale-{{version}}-x86_64-linux
./video_scale video_scale.toml
```

Нужны `ffmpeg`/`ffprobe` в `PATH` и NVIDIA GPU с TensorRT 11. Если библиотеки
CUDA/cuDNN/TensorRT не установлены системно, укажите каталог с ними в `lib_dir` конфига.
Первый запуск для нового размера кадра и `T` компилирует движок TensorRT, дальше он
берётся из `cache_dir`.

### Целостность

```bash
sha256sum -c SHA256SUMS
```

Исходники релиза: https://github.com/{{repo}}/tree/{{tag}}
