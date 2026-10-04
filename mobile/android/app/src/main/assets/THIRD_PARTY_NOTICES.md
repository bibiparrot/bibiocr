# Third-party notices

BIBIOCR's application source in `desktop/` and `mobile/` is GPL-3.0-only.
Copies of the license are included at the root and in both applications.
This change does not replace licenses or attribution for third-party material.

- `desktop/assets/fonts/`: Noto Sans CJK, SIL Open Font License 1.1;
  see `LICENSE.txt` beside the font.
- `desktop/src/tools/pandoc/`: Pandoc Rust wrapper, MIT / Apache-2.0.
- `desktop/src/tools/bibiget/`: bibiget, MIT; see its LICENSE and UPSTREAM.md.
- `desktop/src/model_runtimes/third_party/onnxruntime/`: Microsoft ONNX Runtime,
  MIT; see its LICENSE.
- `desktop/vendor/sherpa-onnx-sys/` and `mobile/vendor/sherpa-onnx-sys/`:
  sherpa-onnx bindings and native runtime, Apache-2.0; see their LICENSE files.
- Other directories in `mobile/vendor/` retain their own upstream notices.
- Melo model: https://huggingface.co/csukuangfj/vits-melo-tts-zh_en
- Kokoro model and voice: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- OCR model assets are downloaded separately from PaddlePaddle model repositories.

Model weights and downloaded executables are separately distributed assets;
consult their upstream license files. Cargo dependencies retain the licenses
declared by their upstream crates. Release packaging preserves bundled native
runtime notices in `native-licenses/`.
