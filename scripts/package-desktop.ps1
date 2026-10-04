param([string]$Version = '2.3.0')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$release = Join-Path $repo 'desktop/target/release'
$stage = Join-Path $repo 'dist/windows'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -LiteralPath (Join-Path $release 'bibiocr.exe') -Destination $stage
$libraries = @(Get-ChildItem -LiteralPath $release -Filter '*.dll')
if (-not ($libraries.Name -contains 'sherpa-onnx-c-api.dll') -or -not ($libraries.Name -contains 'onnxruntime.dll')) {
    throw 'Required TTS runtime DLLs are missing'
}
$libraries | Copy-Item -Destination $stage
Copy-Item -LiteralPath (Join-Path $repo 'LICENSE'),(Join-Path $repo 'THIRD_PARTY_NOTICES.md') -Destination $stage
Copy-Item -LiteralPath (Join-Path $repo 'desktop/assets/fonts/LICENSE.txt') -Destination (Join-Path $stage 'NotoSansCJK-LICENSE.txt')
$notices = Join-Path $stage 'native-licenses'
New-Item -ItemType Directory -Force -Path $notices | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'desktop/vendor/sherpa-onnx-sys/LICENSE') -Destination (Join-Path $notices 'sherpa-onnx.txt')
Copy-Item -LiteralPath (Join-Path $repo 'desktop/src/model_runtimes/third_party/onnxruntime/LICENSE') -Destination (Join-Path $notices 'onnxruntime.txt')
Compress-Archive -Path "$stage/*" -DestinationPath (Join-Path $repo "dist/bibiocr-$Version-windows-x86_64.zip") -Force
