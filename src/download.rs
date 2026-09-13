use crate::{
    dependencies::{self, DependencyKey},
    settings,
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct DownloadOptions {
    pub proxy: String,
    pub no_proxy: bool,
    pub hf_endpoint: String,
    pub github_proxy: String,
    pub resume: bool,
    pub retries: u32,
}

#[derive(Debug)]
pub enum DownloadEvent {
    Started(DependencyKey),
    Progress(DependencyKey, u64, Option<u64>),
    Complete(DependencyKey, PathBuf),
    Failed(DependencyKey, String),
    Finished,
}

#[derive(Clone, Copy)]
enum PackageKind {
    HuggingFace,
    Archive { executable: &'static str },
}

#[derive(Clone)]
struct Package {
    key: DependencyKey,
    url: String,
    destination: PathBuf,
    kind: PackageKind,
}

pub fn start(keys: Vec<DependencyKey>, options: DownloadOptions) -> Receiver<DownloadEvent> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let packages = match packages(&options) {
            Ok(packages) => packages,
            Err(error) => {
                let key = keys.first().copied().unwrap_or(DependencyKey::VlmModel);
                let _ = sender.send(DownloadEvent::Failed(key, error));
                return;
            }
        };
        for package in packages
            .into_iter()
            .filter(|package| keys.contains(&package.key))
        {
            let _ = sender.send(DownloadEvent::Started(package.key));
            match install(&package, &options, &sender) {
                Ok(path) => {
                    let _ = sender.send(DownloadEvent::Complete(package.key, path));
                }
                Err(error) => {
                    let _ = sender.send(DownloadEvent::Failed(package.key, error));
                    return;
                }
            }
        }
        let _ = sender.send(DownloadEvent::Finished);
    });
    receiver
}

fn packages(options: &DownloadOptions) -> Result<Vec<Package>, String> {
    let defaults = dependencies::RuntimeConfig::default_for_platform();
    let mut result = vec![
        Package {
            key: DependencyKey::VlmModel,
            url: hugging_face_url(
                &options.hf_endpoint,
                "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
                "PaddleOCR-VL-1.6-GGUF.gguf",
            ),
            destination: defaults.path(DependencyKey::VlmModel).to_path_buf(),
            kind: PackageKind::HuggingFace,
        },
        Package {
            key: DependencyKey::Mmproj,
            url: hugging_face_url(
                &options.hf_endpoint,
                "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
                "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
            ),
            destination: defaults.path(DependencyKey::Mmproj).to_path_buf(),
            kind: PackageKind::HuggingFace,
        },
        Package {
            key: DependencyKey::LayoutModel,
            url: hugging_face_url(
                &options.hf_endpoint,
                "PaddlePaddle/PP-DocLayoutV3_onnx",
                "inference.onnx",
            ),
            destination: defaults.path(DependencyKey::LayoutModel).to_path_buf(),
            kind: PackageKind::HuggingFace,
        },
    ];

    let (ort, llama, pandoc, pdfium) = platform_archives()?;
    for (key, url, folder, expected) in [
        (
            DependencyKey::OrtDll,
            ort,
            "onnxruntime",
            dependencies::ort_library_name(),
        ),
        (
            DependencyKey::LlamaServer,
            llama,
            "llama",
            if cfg!(windows) {
                "llama-server.exe"
            } else {
                "llama-server"
            },
        ),
        (
            DependencyKey::Pandoc,
            pandoc,
            "pandoc",
            if cfg!(windows) {
                "pandoc.exe"
            } else {
                "pandoc"
            },
        ),
        (
            DependencyKey::Pdfium,
            pdfium,
            "pdfium",
            dependencies::pdfium_library_name(),
        ),
    ] {
        result.push(Package {
            key,
            url: github_url(&options.github_proxy, url),
            destination: settings::runtime_directory().join(folder),
            kind: PackageKind::Archive {
                executable: expected,
            },
        });
    }
    Ok(result)
}

fn platform_archives() -> Result<(&'static str, &'static str, &'static str, &'static str), String> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-win-x64-1.28.1.zip",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-win-cpu-x64.zip",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-windows-x86_64.zip",
            "https://github.com/firecrawl/pdfium-rs/releases/download/native-v7988/firecrawl-pdfium-win-x64.tgz",
        )),
        ("macos", "x86_64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-osx-x86_64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-macos-x64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-x86_64-macOS.zip",
            "https://github.com/firecrawl/pdfium-rs/releases/download/native-v7988/firecrawl-pdfium-mac-x64.tgz",
        )),
        ("macos", "aarch64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-osx-arm64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-macos-arm64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-arm64-macOS.zip",
            "https://github.com/firecrawl/pdfium-rs/releases/download/native-v7988/firecrawl-pdfium-mac-arm64.tgz",
        )),
        ("linux", "x86_64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-linux-x64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-ubuntu-x64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-linux-amd64.tar.gz",
            "https://github.com/firecrawl/pdfium-rs/releases/download/native-v7988/firecrawl-pdfium-linux-x64.tgz",
        )),
        ("linux", "aarch64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-linux-aarch64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-ubuntu-arm64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-linux-arm64.tar.gz",
            "https://github.com/firecrawl/pdfium-rs/releases/download/native-v7988/firecrawl-pdfium-linux-arm64.tgz",
        )),
        (os, arch) => Err(format!("No dependency package is defined for {os}/{arch}")),
    }
}

fn install(
    package: &Package,
    options: &DownloadOptions,
    sender: &Sender<DownloadEvent>,
) -> Result<PathBuf, String> {
    match package.kind {
        PackageKind::HuggingFace => {
            download_file(package, &package.destination, options, sender)?;
            Ok(package.destination.clone())
        }
        PackageKind::Archive { executable } => {
            fs::create_dir_all(&package.destination).map_err(|error| {
                format!("Cannot create {}: {error}", package.destination.display())
            })?;
            let filename = package.url.rsplit('/').next().unwrap_or("package.archive");
            let archive = settings::runtime_directory()
                .join("downloads")
                .join(filename);
            download_file(package, &archive, options, sender)?;
            let status = Command::new("tar")
                .args(["-xf"])
                .arg(&archive)
                .arg("-C")
                .arg(&package.destination)
                .status()
                .map_err(|error| format!("Cannot start archive extractor: {error}"))?;
            if !status.success() {
                return Err(format!("Cannot extract {}", archive.display()));
            }
            let installed = find_file(&package.destination, executable).ok_or_else(|| {
                format!(
                    "{executable} was not found after extracting {}",
                    archive.display()
                )
            })?;
            make_executable(&installed)?;
            Ok(installed)
        }
    }
}

fn download_file(
    package: &Package,
    destination: &Path,
    options: &DownloadOptions,
    sender: &Sender<DownloadEvent>,
) -> Result<(), String> {
    if destination.is_file()
        && destination
            .metadata()
            .is_ok_and(|metadata| metadata.len() > 0)
    {
        let size = destination.metadata().map(|value| value.len()).unwrap_or(0);
        let _ = sender.send(DownloadEvent::Progress(package.key, size, Some(size)));
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Cannot create {}: {error}", parent.display()))?;
    }
    if destination.is_file() {
        fs::remove_file(destination).map_err(|error| error.to_string())?;
    }
    let (progress_sender, progress_receiver) = mpsc::channel();
    let events = sender.clone();
    let key = package.key;
    let progress = thread::spawn(move || {
        while let Ok((downloaded, total)) = progress_receiver.recv() {
            let _ = events.send(DownloadEvent::Progress(key, downloaded, Some(total)));
        }
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    let result = runtime.block_on(bibiget::download(bibiget::DownloadOptions {
        urls: vec![reqwest::Url::parse(&package.url).map_err(|error| error.to_string())?],
        output: Some(destination.to_path_buf()),
        max_speed: 0,
        num_connections: 8,
        headers: reqwest::header::HeaderMap::new(),
        user_agent: concat!("bibiocr/", env!("CARGO_PKG_VERSION")).to_owned(),
        no_proxy: options.no_proxy,
        quiet: true,
        verbose: 0,
        alternate: false,
        timeout: Duration::from_secs(60),
        chunk_size: 4 * 1024 * 1024,
        request_size: None,
        retries: options.retries.max(1) as usize,
        resume: options.resume,
        proxy: (!options.proxy.trim().is_empty()).then(|| options.proxy.trim().to_owned()),
        progress_events: Some(progress_sender),
    }));
    let _ = progress.join();
    result.map(|_| ()).map_err(|error| error.to_string())
}

fn github_url(template: &str, url: &str) -> String {
    let template = template.trim();
    if template.is_empty() {
        return url.to_owned();
    }
    if template.contains("${giturl}") {
        return template.replace("${giturl}", url);
    }
    if template.contains("{giturl}") {
        return template.replace("{giturl}", url);
    }
    format!("{}/{url}", template.trim_end_matches('/'))
}

fn hugging_face_url(endpoint: &str, repository: &str, filename: &str) -> String {
    format!(
        "{}/{repository}/resolve/main/{filename}",
        endpoint.trim_end_matches('/')
    )
}

fn find_file(root: &Path, filename: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().is_some_and(|name| name == filename) {
            return Some(path);
        }
        if path.is_dir()
            && let Some(found) = find_file(&path, filename)
        {
            return Some(found);
        }
    }
    None
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = path
        .metadata()
        .map_err(|error| error.to_string())?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    fs::set_permissions(path, permissions).map_err(|error| error.to_string())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{github_url, hugging_face_url};

    #[test]
    fn github_proxy_supports_documented_template_and_prefix_forms() {
        let url = "https://github.com/example/file.zip";
        assert_eq!(
            github_url("https://gh-proxy.com/${giturl}", url),
            format!("https://gh-proxy.com/{url}")
        );
        assert_eq!(
            github_url("https://gh-proxy.com/", url),
            format!("https://gh-proxy.com/{url}")
        );
        assert_eq!(github_url("", url), url);
    }

    #[test]
    fn hugging_face_url_supports_mirror_endpoint() {
        assert_eq!(
            hugging_face_url(
                "https://hf-mirror.com/",
                "PaddlePaddle/PP-DocLayoutV3_onnx",
                "inference.onnx"
            ),
            "https://hf-mirror.com/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx"
        );
    }
}
