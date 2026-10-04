# BIBIOCR Mobile

Rust/egui Android prototype for offline document OCR. The five application states follow the references in `docs/`: model download, scan, recognition, result, and history.

## What works

- Resumable, pausable downloads for the three published PaddlePaddle model files, with exact-size validation.
- PP-DocLayoutV3 preprocessing, ONNX Runtime inference through `ort`, NMS, and reading-order output.
- Android llama.cpp/MTMD integration for PaddleOCR-VL GGUF + mmproj.
- Android system camera and photo picker feeding the real layout/VLM pipeline,
  with token-by-token OCR output and original-image comparison.
- Persistent OCR history (real results only), deletion, Markdown/DOCX save,
  and Android sharing.
- Runtime Android content insets for status-bar/cutout and navigation areas.
- System-locale selection with TOML catalogs for Chinese, English, Japanese,
  Korean, Russian, French, Spanish, German, Italian, and Portuguese.
- GameActivity-backed egui text input for Android IMEs; Gradle ABI-split APKs
  for arm64-v8a, armeabi-v7a, x86, and x86_64.
- Offline Chinese/English Melo TTS with sentence highlighting, adjustable
  pitch-preserved WSOLA playback speed, persistent per-scan 1x audio, and
  recovery when one speech slice fails. Speed changes never rerun TTS inference;
  the result page's regenerate button creates a fresh voice-cache revision.

DOCX is generated directly in Rust because official Pandoc Linux archives target
glibc Linux while Android uses Bionic.

## Build and test

```powershell
cargo test --lib
pwsh -File scripts/build-android.ps1
```

The build script uses Android SDK 35, NDK r30 for ARM, NDK r26b for x86,
Android Studio JBR, CMake 4.4.3, and Ninja. Override its path parameters on
other machines. It downloads the sherpa-onnx Android runtime archive once and
checks its SHA-256. Four APKs are written to `dist/`. This initial beta uses
the local Android debug signing key so it can update existing installations
without clearing models or history. The GitHub Actions copy of this key is
stored only in the encrypted `ANDROID_DEBUG_KEYSTORE_B64` repository secret;
never commit or publicly upload the key. Pushing `main` builds and tests one
APK per ABI. Pushing a `v*` tag in the unified repository publishes desktop packages and all four APKs after the
builds pass. The ignored real-model tests can be run when model files are cached.

## Settings and mirrors

On first launch the app writes `settings.toml` in its Android internal data
directory. See `settings.example.toml` for all options. `language = "system"`
follows the device locale; a supported locale code can override it.

Chinese automatically rewrites `https://huggingface.co` to
`https://hf-mirror.com`. Set `download.hf_endpoint` or the `HF_ENDPOINT`
environment variable to override it.

GitHub acceleration is configured as a template:

```toml
[download]
resume = true
github_proxy = "https://gh-proxy.com/{giturl}"
```

`{giturl}`, `${giturl}`, and `{$giturl}` are accepted. An empty template uses
GitHub directly. Partial downloads use `.part` files and HTTP `Range`; set
`resume = false` to restart from byte zero.

Completed model files are validated by their published byte sizes and reused on
later launches or APK upgrades; only missing or partial files are downloaded.

## License

Application source is GPL-3.0-only. See `LICENSE` and `../THIRD_PARTY_NOTICES.md`.
The app is maintained in `mobile/` of https://github.com/bibiparrot/bibiocr.
From the repository root use `cargo test --manifest-path mobile/Cargo.toml --lib --locked`
and `pwsh -File mobile/scripts/build-android.ps1`.
