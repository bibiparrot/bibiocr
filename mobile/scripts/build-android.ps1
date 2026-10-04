param(
    [ValidateSet('arm64-v8a', 'armeabi-v7a', 'x86', 'x86_64')]
    [string[]]$Abi = @('arm64-v8a', 'armeabi-v7a', 'x86', 'x86_64'),
    [string]$SdkRoot = 'C:\Users\shi\AppData\Local\Android\Sdk',
    [string]$Ndk30 = 'D:\Programs\android-ndk-r30',
    [string]$Ndk26 = 'D:\Programs\android-ndk-r26b',
    [string]$JdkRoot = 'D:\Program Files\Android\Android Studio\jbr',
    [string]$CmakeBin = '',
    [string]$NinjaBin = 'D:\Python\conda_python_3.12\envs\conda_python\Scripts'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$cache = Join-Path $repo 'target\sherpa-onnx-prebuilt'
$archive = Join-Path $cache 'sherpa-onnx-v1.13.8-android.tar.bz2'
if (-not $CmakeBin) { $CmakeBin = Join-Path $repo 'target\tools\cmake-4.4.3-windows-x86_64\bin' }

function Assert-Success([string]$step) {
    if ($LASTEXITCODE -ne 0) { throw "$step failed (exit $LASTEXITCODE)" }
}

foreach ($path in @($SdkRoot, $JdkRoot, $CmakeBin, $NinjaBin)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Build prerequisite missing: $path" }
}

if (-not (Test-Path -LiteralPath $archive)) {
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    Invoke-WebRequest -Uri 'https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-v1.13.8-android.tar.bz2' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne '2ff63469a71cb6009aa2e3ed5f4a670f8abdcbe4bb9ffd23776afc792a6b4f44') {
    throw 'Cached sherpa-onnx Android archive has the wrong SHA-256'
}
if (-not (Test-Path -LiteralPath (Join-Path $cache 'jniLibs\x86_64\libonnxruntime.so'))) {
    tar -xjf $archive -C $cache
    Assert-Success 'Extract sherpa-onnx runtime'
}

$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_JAR = Join-Path $SdkRoot 'platforms\android-35\android.jar'
$env:ANDROID_API_LEVEL = '27'
$env:JAVA_HOME = $JdkRoot
$env:JAVA_SOURCE_VERSION = '17'
$env:JAVA_TARGET_VERSION = '17'
$env:CMAKE_GENERATOR = 'Ninja'
$env:ORT_PREFER_DYNAMIC_LINK = '1'
$env:PATH = "$CmakeBin;$NinjaBin;$env:PATH"

$targets = @{
    'arm64-v8a' = @{ Rust = 'aarch64-linux-android'; Clang = 'aarch64-linux-android'; Runtime = 'aarch64-linux-android'; Ndk = $Ndk30 }
    'armeabi-v7a' = @{ Rust = 'armv7-linux-androideabi'; Clang = 'armv7a-linux-androideabi'; Runtime = 'arm-linux-androideabi'; Ndk = $Ndk30 }
    'x86' = @{ Rust = 'i686-linux-android'; Clang = 'i686-linux-android'; Runtime = 'i686-linux-android'; Ndk = $Ndk26 }
    'x86_64' = @{ Rust = 'x86_64-linux-android'; Clang = 'x86_64-linux-android'; Runtime = 'x86_64-linux-android'; Ndk = $Ndk26 }
}

Push-Location $repo
try {
    foreach ($abiName in $Abi) {
        $config = $targets[$abiName]
        $target = $config.Rust
        $ndk = $config.Ndk
        if (-not (Test-Path -LiteralPath $ndk)) { throw "Build prerequisite missing: $ndk" }
        $toolBin = Join-Path $ndk 'toolchains\llvm\prebuilt\windows-x86_64\bin'
        $cc = Join-Path $toolBin "$($config.Clang)27-clang.cmd"
        $cxx = Join-Path $toolBin "$($config.Clang)27-clang++.cmd"
        $runtime = Join-Path $cache "jniLibs\$abiName"
        if (-not (Test-Path -LiteralPath $cc)) { throw "Missing C compiler: $cc" }
        $env:ANDROID_NDK_ROOT = $ndk
        $env:ANDROID_NDK_HOME = $ndk
        $env:ORT_LIB_LOCATION = $runtime
        $env:SHERPA_ONNX_LIB_DIR = $runtime
        $env:PATH = "$toolBin;$CmakeBin;$NinjaBin;$env:PATH"
        $suffix = $target.Replace('-', '_')
        [Environment]::SetEnvironmentVariable("CC_$suffix", $cc, 'Process')
        [Environment]::SetEnvironmentVariable("CXX_$suffix", $cxx, 'Process')
        [Environment]::SetEnvironmentVariable("CARGO_TARGET_$($suffix.ToUpperInvariant())_LINKER", $cc, 'Process')
        [Environment]::SetEnvironmentVariable("BINDGEN_EXTRA_CLANG_ARGS_$suffix", "--target=$($config.Clang)27", 'Process')
        if ($target -eq 'armv7-linux-androideabi') {
            # llama-cpp-sys expects the variable even when Rust reports no default features.
            $env:CARGO_CFG_TARGET_FEATURE = 'none'
        } else {
            Remove-Item Env:CARGO_CFG_TARGET_FEATURE -ErrorAction SilentlyContinue
        }
        rustup target add $target
        Assert-Success "Install Rust target $target"
        cargo build --locked --quiet --release --lib --target $target --jobs 1
        Assert-Success "Build Rust $abiName"

        $destination = Join-Path $repo "android\app\src\main\jniLibs\$abiName"
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Copy-Item -LiteralPath (Join-Path $repo "target\$target\release\libbibiocr_mobile.so") -Destination $destination -Force
        Copy-Item -LiteralPath (Join-Path $runtime 'libonnxruntime.so') -Destination $destination -Force
        Copy-Item -LiteralPath (Join-Path $runtime 'libsherpa-onnx-c-api.so') -Destination $destination -Force
        Copy-Item -LiteralPath (Join-Path $ndk "toolchains\llvm\prebuilt\windows-x86_64\sysroot\usr\lib\$($config.Runtime)\libc++_shared.so") -Destination $destination -Force
    }

    Push-Location (Join-Path $repo 'android')
    try {
        $localGradle = Join-Path $repo 'target\tools\gradle-9.3.1\bin\gradle.bat'
        if (Test-Path -LiteralPath $localGradle) {
            & $localGradle :app:assembleRelease
        } else {
            .\gradlew.bat :app:assembleRelease
        }
        Assert-Success 'Build signed Android APKs'
    } finally { Pop-Location }

    $dist = Join-Path $repo 'dist'
    New-Item -ItemType Directory -Force -Path $dist | Out-Null
    foreach ($abiName in $Abi) {
        $source = Join-Path $repo "android\app\build\outputs\apk\release\app-$abiName-release.apk"
        if (-not (Test-Path -LiteralPath $source)) { throw "Missing APK: $source" }
        Copy-Item -LiteralPath $source -Destination (Join-Path $dist "bibiocr-mobile-v0.1.6-$abiName.apk") -Force
    }
    Get-ChildItem -LiteralPath $dist -Filter '*.apk' | Select-Object Name, Length
} finally { Pop-Location }
