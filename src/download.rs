use crate::{
    dependencies::{self, DependencyKey},
    settings,
};
use futures::StreamExt;
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct DownloadOptions {
    pub proxy: String,
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
    HuggingFace {
        owner: &'static str,
        repository: &'static str,
        filename: &'static str,
    },
    Archive {
        executable: &'static str,
    },
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
            url: String::new(),
            destination: defaults.path(DependencyKey::VlmModel).to_path_buf(),
            kind: PackageKind::HuggingFace {
                owner: "PaddlePaddle",
                repository: "PaddleOCR-VL-1.6-GGUF",
                filename: "PaddleOCR-VL-1.6-GGUF.gguf",
            },
        },
        Package {
            key: DependencyKey::Mmproj,
            url: String::new(),
            destination: defaults.path(DependencyKey::Mmproj).to_path_buf(),
            kind: PackageKind::HuggingFace {
                owner: "PaddlePaddle",
                repository: "PaddleOCR-VL-1.6-GGUF",
                filename: "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
            },
        },
        Package {
            key: DependencyKey::LayoutModel,
            url: String::new(),
            destination: defaults.path(DependencyKey::LayoutModel).to_path_buf(),
            kind: PackageKind::HuggingFace {
                owner: "PaddlePaddle",
                repository: "PP-DocLayoutV3_onnx",
                filename: "inference.onnx",
            },
        },
    ];

    let (ort, llama, pandoc) = platform_archives()?;
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

fn platform_archives() -> Result<(&'static str, &'static str, &'static str), String> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-win-x64-1.28.1.zip",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-win-cpu-x64.zip",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-windows-x86_64.zip",
        )),
        ("macos", "x86_64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-osx-x86_64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-macos-x64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-x86_64-macOS.zip",
        )),
        ("macos", "aarch64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-osx-arm64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-macos-arm64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-arm64-macOS.zip",
        )),
        ("linux", "x86_64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-linux-x64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-ubuntu-x64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-linux-amd64.tar.gz",
        )),
        ("linux", "aarch64") => Ok((
            "https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-linux-aarch64-1.28.1.tgz",
            "https://github.com/ggml-org/llama.cpp/releases/download/b10603/llama-b10603-bin-ubuntu-arm64.tar.gz",
            "https://github.com/jgm/pandoc/releases/download/3.10.2/pandoc-3.10.2-linux-arm64.tar.gz",
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
        PackageKind::HuggingFace {
            owner,
            repository,
            filename,
        } => {
            download_hugging_face(package, owner, repository, filename, options, sender)?;
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

fn download_hugging_face(
    package: &Package,
    owner: &str,
    repository: &str,
    filename: &str,
    options: &DownloadOptions,
    sender: &Sender<DownloadEvent>,
) -> Result<(), String> {
    if package.destination.is_file()
        && package
            .destination
            .metadata()
            .is_ok_and(|metadata| metadata.len() > 0)
    {
        let size = package
            .destination
            .metadata()
            .map(|value| value.len())
            .unwrap_or(0);
        let _ = sender.send(DownloadEvent::Progress(package.key, size, Some(size)));
        return Ok(());
    }
    if let Some(parent) = package.destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Cannot create {}: {error}", parent.display()))?;
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async {
        let mut http = reqwest::Client::builder();
        http = http.user_agent(concat!("bibiocr/", env!("CARGO_PKG_VERSION")));
        if !options.proxy.trim().is_empty() {
            http = http.proxy(
                reqwest::Proxy::all(options.proxy.trim())
                    .map_err(|error| format!("Invalid proxy: {error}"))?,
            );
        }
        let client = hf_hub::HFClient::builder()
            .endpoint(options.hf_endpoint.trim_end_matches('/'))
            .cache_dir(settings::runtime_directory().join("hf-cache"))
            .retry_max_attempts(options.retries as usize)
            .client(http.build().map_err(|error| error.to_string())?)
            .build()
            .map_err(|error| error.to_string())?;
        let repo = client.model(owner, repository);
        let metadata = repo
            .get_file_metadata()
            .filepath(filename)
            .send()
            .await
            .map_err(|error| error.to_string())?;
        let partial = package.destination.with_extension(format!(
            "{}part",
            package
                .destination
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| format!("{value}."))
                .unwrap_or_default()
        ));
        let mut offset = if options.resume {
            partial.metadata().map(|value| value.len()).unwrap_or(0)
        } else {
            0
        };
        if offset > metadata.file_size {
            fs::remove_file(&partial).map_err(|error| error.to_string())?;
            offset = 0;
        }
        if offset == metadata.file_size && offset > 0 {
            fs::rename(&partial, &package.destination).map_err(|error| error.to_string())?;
            let _ = sender.send(DownloadEvent::Progress(
                package.key,
                offset,
                Some(metadata.file_size),
            ));
            return Ok(());
        }
        let builder = repo.download_file_stream().filename(filename);
        let (_, mut stream) = if offset > 0 {
            builder
                .range(offset..metadata.file_size)
                .send()
                .await
                .map_err(|error| error.to_string())?
        } else {
            builder.send().await.map_err(|error| error.to_string())?
        };
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(offset > 0)
            .truncate(offset == 0)
            .open(&partial)
            .map_err(|error| error.to_string())?;
        let mut downloaded = offset;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| error.to_string())?;
            file.write_all(&chunk).map_err(|error| error.to_string())?;
            downloaded += chunk.len() as u64;
            let _ = sender.send(DownloadEvent::Progress(
                package.key,
                downloaded,
                Some(metadata.file_size),
            ));
        }
        file.flush().map_err(|error| error.to_string())?;
        if downloaded != metadata.file_size {
            return Err(format!(
                "Incomplete response: received {downloaded} of {} bytes",
                metadata.file_size
            ));
        }
        fs::rename(&partial, &package.destination).map_err(|error| error.to_string())
    })
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
    let partial = destination.with_extension(format!(
        "{}part",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| format!("{value}."))
            .unwrap_or_default()
    ));
    let mut last_error = String::new();
    for attempt in 0..=options.retries {
        match download_attempt(package, destination, &partial, options, sender) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = error,
        }
        if attempt < options.retries {
            thread::sleep(Duration::from_secs(1_u64 << attempt.min(5)));
        }
    }
    Err(format!(
        "Download failed after {} attempts: {last_error}",
        options.retries + 1
    ))
}

fn download_attempt(
    package: &Package,
    destination: &Path,
    partial: &Path,
    options: &DownloadOptions,
    sender: &Sender<DownloadEvent>,
) -> Result<(), String> {
    let offset = if options.resume {
        partial.metadata().map(|value| value.len()).unwrap_or(0)
    } else {
        0
    };
    let mut builder = ureq::Agent::config_builder().timeout_global(None);
    if !options.proxy.trim().is_empty() {
        let proxy = ureq::Proxy::new(options.proxy.trim())
            .map_err(|error| format!("Invalid proxy: {error}"))?;
        builder = builder.proxy(Some(proxy));
    }
    let agent: ureq::Agent = builder.build().into();
    let mut request = agent.get(&package.url);
    if offset > 0 {
        request = request.header("Range", &format!("bytes={offset}-"));
    }
    let mut response = request.call().map_err(|error| error.to_string())?;
    let resumed = offset > 0 && response.status().as_u16() == 206;
    let downloaded = if resumed { offset } else { 0 };
    let total = response
        .body()
        .content_length()
        .map(|length| length + downloaded);
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(partial)
        .map_err(|error| format!("Cannot open {}: {error}", partial.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut current = downloaded;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        file.write_all(&buffer[..count])
            .map_err(|error| error.to_string())?;
        current += count as u64;
        let _ = sender.send(DownloadEvent::Progress(package.key, current, total));
    }
    file.flush().map_err(|error| error.to_string())?;
    if let Some(total) = total
        && current != total
    {
        return Err(format!(
            "Incomplete response: received {current} of {total} bytes"
        ));
    }
    fs::rename(partial, destination)
        .map_err(|error| format!("Cannot finish {}: {error}", destination.display()))
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
    use super::github_url;

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
}
