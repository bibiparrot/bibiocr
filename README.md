# BIBIOCR

Offline OCR and text-to-speech for desktop and Android, licensed under GPL-3.0-only.

| Directory | Application | Build |
| --- | --- | --- |
| [`desktop/`](desktop/) | Windows, macOS, Linux desktop | `cargo build --manifest-path desktop/Cargo.toml --release --locked` |
| [`mobile/`](mobile/) | Android, four ABI APKs | `pwsh -File mobile/scripts/build-android.ps1` |

The applications keep separate manifests, lockfiles, assets, native integrations,
and build outputs. Run tests separately:

```sh
cargo test --manifest-path desktop/Cargo.toml --locked
cargo test --manifest-path mobile/Cargo.toml --lib --locked
```

Desktop includes offline Melo (Chinese/English) and Kokoro (English) speech,
sentence highlighting, pause/resume/stop, volume, pitch-preserving speed control,
and persistent sentence audio. Open **Read aloud** to download voice models or
select an existing model directory. The existing download proxy/mirror settings
also apply to voice downloads. Once models are available, synthesis is offline.

See each application's README for OCR setup and platform prerequisites.
Tagged releases build desktop packages and Android APKs in one workflow, then
publish them together with GPL license text, source archives, and SHA-256 hashes.
Android remains `com.bibiocr.mobile` and keeps the existing beta signing key;
the destination repository needs `ANDROID_DEBUG_KEYSTORE_B64` to build releases.

Third-party code, fonts, runtimes, and model weights retain their upstream
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Mobile was imported from `bibiparrot/bibiocr-mobile` commit
`63f61fc11633feb6f4ac725c57a57894d188344e`. See
[mobile/UPSTREAM.md](mobile/UPSTREAM.md) for provenance.
