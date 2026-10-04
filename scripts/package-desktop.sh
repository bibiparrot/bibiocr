#!/usr/bin/env bash
set -euo pipefail
version=${1:-2.3.0}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
release=desktop/target/release
os=$(uname -s)
arch=$(uname -m)
if [[ "$os" == Darwin ]]; then
  app=dist/BIBIOCR.app
  stage="$app/Contents/MacOS"
else
  app="dist/bibiocr-$version-linux-$arch"
  stage="$app/lib"
fi
mkdir -p "$stage/native-licenses"
cp "$release/bibiocr" "$stage/bibiocr"
cp LICENSE THIRD_PARTY_NOTICES.md "$stage/"
cp desktop/assets/fonts/LICENSE.txt "$stage/NotoSansCJK-LICENSE.txt"
cp desktop/vendor/sherpa-onnx-sys/LICENSE "$stage/native-licenses/sherpa-onnx.txt"
cp desktop/src/model_runtimes/third_party/onnxruntime/LICENSE "$stage/native-licenses/onnxruntime.txt"
if [[ "$os" == Darwin ]]; then
  find "$release" -maxdepth 1 -name '*.dylib' -exec cp -L {} "$stage/" \;
  test -s "$stage/libsherpa-onnx-c-api.dylib"
  test -s "$stage/libonnxruntime.dylib"
  # Remove build-machine library paths from both executable and dylibs.
  for binary in "$stage/bibiocr" "$stage/"*.dylib; do
    while IFS= read -r dep; do
      name=$(basename "$dep")
      if [[ -f "$stage/$name" && "$dep" != "@executable_path/$name" ]]; then
        install_name_tool -change "$dep" "@executable_path/$name" "$binary"
      fi
    done < <(otool -L "$binary" | tail -n +2 | awk '{print $1}')
    if [[ "$binary" == *.dylib ]]; then install_name_tool -id "@rpath/$(basename "$binary")" "$binary"; fi
  done
  mkdir -p "$app/Contents/Resources" dist/bibiocr.iconset
  for size in 16 32 128 256 512; do
    sips -z "$size" "$size" desktop/assets/bibiocr-icon.png --out "dist/bibiocr.iconset/icon_${size}x${size}.png" >/dev/null
    retina=$((size * 2))
    sips -z "$retina" "$retina" desktop/assets/bibiocr-icon.png --out "dist/bibiocr.iconset/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns dist/bibiocr.iconset -o "$app/Contents/Resources/bibiocr.icns"
  cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>bibiocr</string>
<key>CFBundleIdentifier</key><string>com.bibiocr.desktop</string>
<key>CFBundleName</key><string>BIBIOCR</string>
<key>CFBundleIconFile</key><string>bibiocr.icns</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$version</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
  plutil -lint "$app/Contents/Info.plist"
  for library in "$stage/"*.dylib; do codesign --force --sign - "$library"; done
  codesign --force --sign - "$app"
  codesign --verify --deep --strict "$app"
  ditto -c -k --keepParent "$app" "dist/bibiocr-$version-$arch-macOS.zip"
  pkgbuild --component "$app" --install-location /Applications "dist/bibiocr-$version-$arch-macOS.pkg"
  hdiutil create -volname BIBIOCR -srcfolder "$app" -ov -format UDZO "dist/bibiocr-$version-$arch-macOS.dmg"
else
  find "$release" -maxdepth 1 -name '*.so*' -exec cp -L {} "$stage/" \;
  test -s "$stage/libsherpa-onnx-c-api.so"
  test -s "$stage/libonnxruntime.so"
  cat > "$app/bibiocr" <<'EOF'
#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/lib/bibiocr" "$@"
EOF
  chmod +x "$app/bibiocr"
  LD_LIBRARY_PATH="$stage" ldd "$stage/bibiocr" > "$app/runtime-linkage.txt"
  if grep -q 'not found' "$app/runtime-linkage.txt"; then cat "$app/runtime-linkage.txt"; exit 1; fi
  tar -czf "dist/bibiocr-$version-linux-$arch.tar.gz" -C dist "$(basename "$app")"
  if [[ "$arch" == aarch64 ]]; then deb_arch=arm64; else deb_arch=amd64; fi
  mkdir -p dist/deb/DEBIAN dist/deb/usr/lib/bibiocr dist/deb/usr/bin
  cp -R "$stage/". dist/deb/usr/lib/bibiocr/
  printf '#!/bin/sh\nexport LD_LIBRARY_PATH="/usr/lib/bibiocr${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec /usr/lib/bibiocr/bibiocr "$@"\n' > dist/deb/usr/bin/bibiocr
  chmod +x dist/deb/usr/bin/bibiocr
  printf 'Package: bibiocr\nVersion: %s\nArchitecture: %s\nMaintainer: BIBIOCR contributors\nDepends: libasound2, libgtk-3-0, libxkbcommon0, libwayland-client0, libgl1\nDescription: Offline OCR and TTS desktop application\n' "$version" "$deb_arch" > dist/deb/DEBIAN/control
  dpkg-deb --build --root-owner-group dist/deb "dist/bibiocr_${version}_${deb_arch}.deb"
fi
