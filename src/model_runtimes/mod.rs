use crate::{
    dependencies::{DependencyKey, RuntimeConfig},
    ffi::bridge,
};
use std::{
    env, fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
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
            let _ = sender.send(BackendEvent::Progress(
                0.20,
                rust_i18n::t!("detecting_layout").into_owned(),
            ));
            match process_image(&image_path) {
                Ok(output) => {
                    let _ = sender.send(BackendEvent::Progress(
                        0.94,
                        rust_i18n::t!("loading_artifacts").into_owned(),
                    ));
                    let _ = sender.send(BackendEvent::Complete(output));
                }
                Err(error) => {
                    let _ = sender.send(BackendEvent::Failed(error));
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

pub fn process_image(image_path: &Path) -> Result<PipelineOutput, String> {
    let config_path = config_path()?;
    let output_dir = output_directory();
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("Cannot create output directory / 无法创建输出目录: {error}"))?;
    let image = image_path.to_string_lossy();
    let output = output_dir.to_string_lossy();
    let config = config_path.to_string_lossy();
    cxx::let_cxx_string!(image_cxx = image.as_ref());
    cxx::let_cxx_string!(output_cxx = output.as_ref());
    cxx::let_cxx_string!(config_cxx = config.as_ref());
    bridge::run_pipeline(&image_cxx, &output_cxx, &config_cxx)
        .map(|result| PipelineOutput {
            markdown: result.markdown,
            overlay_path: PathBuf::from(result.layout_path),
            output_dir: PathBuf::from(result.output_dir),
        })
        .map_err(|error| format!("C++ pipeline failed / C++ 管线失败: {error}"))
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
    let config = RuntimeConfig::load();
    let missing = config.missing();
    if !missing.is_empty() {
        return Err(format!(
            "Missing dependencies: {}",
            missing
                .into_iter()
                .map(DependencyKey::name)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    config_path()
}

pub fn pandoc_path() -> Result<PathBuf, String> {
    let resolved = RuntimeConfig::load().tools.pandoc;
    if resolved.is_file() {
        Ok(resolved.canonicalize().unwrap_or(resolved))
    } else {
        Err(rust_i18n::t!("pandoc_missing", path = resolved.display()).into_owned())
    }
}

fn config_path() -> Result<PathBuf, String> {
    let path = crate::dependencies::active_config_path();
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!("Cannot find bibiocr.toml: {}", path.display()))
    }
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
    use super::{BackendEvent, start_pipeline};
    use std::{path::PathBuf, time::Duration};

    #[test]
    #[ignore = "loads the real ONNX and GGUF models"]
    fn bridge_pipeline_smoke() {
        let image = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("model_runtimes")
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
