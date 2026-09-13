use crate::ffi::bridge;
use serde::Deserialize;
use std::{
    env, fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
pub struct PipelineOutput {
    pub markdown: String,
    pub overlay_path: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Debug)]
pub enum BackendEvent {
    Progress(f32, String),
    Complete(PipelineOutput),
    Failed(String),
}

pub fn start_pipeline(image_path: PathBuf) -> Receiver<BackendEvent> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let panic_sender = sender.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = sender.send(BackendEvent::Progress(
                0.08,
                rust_i18n::t!("loading_models").into_owned(),
            ));
            let config_path = match config_path() {
                Ok(path) => path,
                Err(error) => {
                    let _ = sender.send(BackendEvent::Failed(error));
                    return;
                }
            };
            let output_dir = output_directory();
            if let Err(error) = fs::create_dir_all(&output_dir) {
                let _ = sender.send(BackendEvent::Failed(format!(
                    "Cannot create output directory / 无法创建输出目录: {error}"
                )));
                return;
            }

            let _ = sender.send(BackendEvent::Progress(
                0.20,
                rust_i18n::t!("detecting_layout").into_owned(),
            ));
            let image = image_path.to_string_lossy();
            let output = output_dir.to_string_lossy();
            let config = config_path.to_string_lossy();
            cxx::let_cxx_string!(image_cxx = image.as_ref());
            cxx::let_cxx_string!(output_cxx = output.as_ref());
            cxx::let_cxx_string!(config_cxx = config.as_ref());

            match bridge::run_pipeline(&image_cxx, &output_cxx, &config_cxx) {
                Ok(result) => {
                    let _ = sender.send(BackendEvent::Progress(
                        0.94,
                        rust_i18n::t!("loading_artifacts").into_owned(),
                    ));
                    let _ = sender.send(BackendEvent::Complete(PipelineOutput {
                        markdown: result.markdown,
                        overlay_path: PathBuf::from(result.layout_path),
                        output_dir: PathBuf::from(result.output_dir),
                    }));
                }
                Err(error) => {
                    let _ = sender.send(BackendEvent::Failed(format!(
                        "C++ pipeline failed / C++ 管线失败: {error}"
                    )));
                }
            }
        }));
        if let Err(payload) = result {
            let _ = panic_sender.send(BackendEvent::Failed(
                rust_i18n::t!("backend_panic", error = panic_message(payload)).into_owned(),
            ));
        }
    });
    receiver
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "unknown panic".to_owned()
    }
}

pub fn availability() -> Result<PathBuf, String> {
    if !cfg!(target_os = "windows") {
        return Err(
            "Linux UI is available; the copied backend still requires Windows WIC/WinHTTP APIs. / Linux 界面可用，但 C++ 后端仍依赖 Windows API。"
                .to_owned(),
        );
    }
    config_path()
}

#[derive(Deserialize)]
struct RuntimeConfig {
    tools: Option<RuntimeTools>,
}

#[derive(Deserialize)]
struct RuntimeTools {
    pandoc: Option<PathBuf>,
}

pub fn pandoc_path() -> Result<PathBuf, String> {
    let config = config_path()?;
    let contents = fs::read_to_string(&config).map_err(|error| {
        rust_i18n::t!(
            "pandoc_config_error",
            path = config.display(),
            error = error
        )
        .into_owned()
    })?;
    let parsed: RuntimeConfig = toml::from_str(&contents).map_err(|error| {
        rust_i18n::t!(
            "pandoc_config_error",
            path = config.display(),
            error = error
        )
        .into_owned()
    })?;
    let configured = parsed
        .tools
        .and_then(|tools| tools.pandoc)
        .ok_or_else(|| rust_i18n::t!("pandoc_not_configured").into_owned())?;
    let runtime_directory = executable_directory()?;
    let resolved = resolve_from_executable_directory(&configured, &runtime_directory);
    if resolved.is_file() {
        Ok(resolved.canonicalize().unwrap_or(resolved))
    } else {
        Err(rust_i18n::t!("pandoc_missing", path = resolved.display()).into_owned())
    }
}

fn config_path() -> Result<PathBuf, String> {
    if let Some(explicit) = env::var_os("BIBIOCR_CONFIG") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Ok(directory) = executable_directory() {
        let beside_executable = directory.join("bibiocr.toml");
        if beside_executable.is_file() {
            return Ok(beside_executable);
        }
    }
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("backend")
        .join("bibiocr.toml");
    if manifest_path.is_file() {
        return Ok(manifest_path);
    }
    Err(
        "Cannot find bibiocr.toml. Set BIBIOCR_CONFIG or copy it beside the executable. / 找不到 bibiocr.toml。"
            .to_owned(),
    )
}

fn executable_directory() -> Result<PathBuf, String> {
    env::current_exe()
        .map_err(|error| format!("Cannot determine executable path: {error}"))?
        .parent()
        .map(std::path::Path::to_path_buf)
        .ok_or_else(|| "Cannot determine executable directory".to_owned())
}

fn resolve_from_executable_directory(
    configured: &std::path::Path,
    directory: &std::path::Path,
) -> PathBuf {
    let combined = if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        directory.join(configured)
    };
    normalize_lexically(&combined)
}

fn normalize_lexically(path: &std::path::Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn output_directory() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    env::temp_dir()
        .join("bibiocr")
        .join(format!("pipeline-{stamp}"))
}

#[cfg(test)]
mod tests {
    use super::{BackendEvent, RuntimeConfig, resolve_from_executable_directory, start_pipeline};
    use std::{path::PathBuf, time::Duration};

    #[test]
    fn relative_runtime_paths_are_based_on_the_executable_directory() {
        let executable_directory = PathBuf::from("bundle").join("bin");
        let relative = PathBuf::from("..").join("tools").join("pandoc.exe");
        assert_eq!(
            resolve_from_executable_directory(&relative, &executable_directory),
            PathBuf::from("bundle").join("tools").join("pandoc.exe")
        );
    }

    #[test]
    fn absolute_runtime_paths_are_unchanged() {
        let absolute = if cfg!(windows) {
            PathBuf::from(r"D:\tools\pandoc.exe")
        } else {
            PathBuf::from("/opt/tools/pandoc")
        };
        assert_eq!(
            resolve_from_executable_directory(&absolute, std::path::Path::new("ignored")),
            absolute
        );
    }

    #[cfg(windows)]
    #[test]
    fn checked_in_pandoc_path_resolves_for_the_release_bundle() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let contents = std::fs::read_to_string(manifest.join("backend/bibiocr.toml")).unwrap();
        let configured = toml::from_str::<RuntimeConfig>(&contents)
            .unwrap()
            .tools
            .and_then(|tools| tools.pandoc)
            .expect("[tools].pandoc must be configured");
        let release_directory = manifest.join("target/release");
        let resolved = resolve_from_executable_directory(&configured, &release_directory);
        let expected = manifest
            .parent()
            .unwrap()
            .join("pandoc-3.10.2-windows-x86_64/pandoc-3.10.2/pandoc.exe");
        assert_eq!(resolved, expected);
    }

    #[test]
    #[ignore = "loads the real ONNX and GGUF models"]
    fn bridge_pipeline_smoke() {
        let image = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("backend")
            .join("testdata")
            .join("demo1.png");
        let receiver = start_pipeline(image);
        loop {
            match receiver.recv_timeout(Duration::from_secs(300)) {
                Ok(BackendEvent::Complete(output)) => {
                    assert!(!output.markdown.trim().is_empty());
                    assert!(output.overlay_path.is_file());
                    break;
                }
                Ok(BackendEvent::Failed(error)) => panic!("{error}"),
                Ok(BackendEvent::Progress(_, _)) => {}
                Err(error) => panic!("backend timed out or disconnected: {error}"),
            }
        }
    }
}
