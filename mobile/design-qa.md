# Design QA

- Source visuals: `docs/download.png`, `docs/photo.png`, `docs/ocr.png`,
  `docs/text.png`, `docs/history.png`, plus the five attached 1280×2772 phone
  screenshots supplied during implementation.
- Target viewport/state: portrait Android; download, empty/filled history,
  system camera/photo picker, streaming recognition, and result tabs.
- Focused findings fixed in code: runtime top/bottom system insets; fake camera,
  history, OCR text, result text, and original-image placeholders removed;
  save/share actions connected to native Android flows.
- Implementation evidence: `cargo test --lib` passes 32 tests, with three
  cached-model/integration smoke tests intentionally ignored by default.
  All four ABI-split APKs build with their expected native libraries and
  matching update signatures.
- Device: a phone is connected; the latest APK install and visual comparison
  remain pending the phone's USB installation prompt.

final result: pending device verification
