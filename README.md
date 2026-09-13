# BIBIOCR

BIBIOCR is a Rust/C++ offline OCR desktop app for Windows, macOS and Linux. It
processes images or folders, reads clipboard images, and exports structured
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
pdfium = '.../pdfium.dll'

[tools]
pandoc = '.../pandoc.exe'
```

## Batch conversion

Open **Batch Processing**, choose separate input and output folders, then start the batch. Supported office documents are converted with [AnyDoc](https://github.com/firecrawl/anydoc); images and scanned PDF pages are recognized by BIBIOCR. Every input produces `original-name.ext.md` and `original-name.ext.docx` in the output folder.

Model files are downloaded with the official
[`hf-hub`](https://github.com/huggingface/hf-hub) Rust client. Choose
`https://huggingface.co` or the China mirror `https://hf-mirror.com` in the UI
(equivalent to `HF_ENDPOINT=https://hf-mirror.com`). HTTP(S)/SOCKS proxy, retry count,
progress and resumable downloads are supported. Proxy, HF mirror, and GitHub
acceleration can each be enabled or disabled independently.

GitHub release assets are downloaded by the bundled `bibiget` 0.1.1 library
with eight parallel connections, resumable checkpoints, proxy support, and GUI progress events.

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
