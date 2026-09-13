# BIBIOCR

BIBIOCR is a Rust/C++ offline OCR desktop app for Windows, macOS and Linux. It
processes one image at a time, reads clipboard images, and exports structured
Markdown or Word documents.

## Required downloads

Models and third-party runtimes are intentionally not bundled. On first launch,
BIBIOCR opens **Downloads and dependency configuration**. Download everything
there, or select files already on disk. The resulting `bibiocr.toml` contains:

```toml
[dependencies]
vlm_model = '.../PaddleOCR-VL-1.6-GGUF.gguf'
mmproj = '.../PaddleOCR-VL-1.6-GGUF-mmproj.gguf'
layout_model = '.../inference.onnx'
ort_dll = '.../onnxruntime.dll'
llama_server = '.../llama-server.exe'

[tools]
pandoc = '.../pandoc.exe'
```

Model files are downloaded with the official
[`hf-hub`](https://github.com/huggingface/hf-hub) Rust client. Choose
`https://huggingface.co` or the China mirror `https://hf-mirror.com` in the UI
(equivalent to `HF_ENDPOINT=https://hf-mirror.com`). HTTP(S) proxy, retry count,
progress and resumable downloads are supported.

For GitHub release assets, leave acceleration blank for direct GitHub access or
use a template such as `https://gh-proxy.com/${giturl}`. `${giturl}` is replaced
with the original GitHub URL.

## UI Demo
![DEMO Image](docs/bibiocr_2.1_demo.png)



## Language support

- English
- Chinese
- French
- Russian
- Korean
- Japanese
