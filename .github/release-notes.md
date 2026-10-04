BIBIOCR now contains separate desktop/ and mobile/ applications under GPL-3.0-only.

- Desktop 2.3.0: Windows, macOS (Intel / Apple Silicon), Linux (x86_64 / ARM64).
- Android 0.1.6: arm64-v8a, armeabi-v7a, x86 and x86_64 APKs.
- Desktop offline Melo Chinese/English and Kokoro English speech, pause/resume,
  stop, sentence highlighting, persistent audio and pitch-preserving live speed.
- Speech models download on demand and use the existing proxy/mirror settings.
- Windows ZIPs and macOS/Linux packages include the required speech runtime.
- Android retains its existing package identity and beta signing certificate.

Extract the complete Windows ZIP before running bibiocr.exe. macOS bundles use
ad-hoc signing and are not notarized. Linux portable archives launch through
the top-level bibiocr script; GTK, ALSA and OpenGL system libraries are required.
OCR and speech models download separately on first use. Third-party code,
fonts, model assets and runtimes retain their upstream licenses.

The corresponding application source is included in the source archive.
SHA256SUMS.txt covers every attached binary, source archive and license notice.
