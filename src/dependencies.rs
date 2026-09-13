use crate::settings;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyKey {
    VlmModel,
    Mmproj,
    LayoutModel,
    OrtDll,
    LlamaServer,
    Pandoc,
}

impl DependencyKey {
    pub const ALL: [Self; 6] = [
        Self::VlmModel,
        Self::Mmproj,
        Self::LayoutModel,
        Self::OrtDll,
        Self::LlamaServer,
        Self::Pandoc,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::VlmModel => "vlm_model",
            Self::Mmproj => "mmproj",
            Self::LayoutModel => "layout_model",
            Self::OrtDll => "ort_dll",
            Self::LlamaServer => "llama_server",
            Self::Pandoc => "pandoc",
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub dependencies: DependencyPaths,
    pub tools: ToolPaths,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct DependencyPaths {
    pub vlm_model: PathBuf,
    pub mmproj: PathBuf,
    pub layout_model: PathBuf,
    pub ort_dll: PathBuf,
    pub llama_server: PathBuf,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct ToolPaths {
    pub pandoc: PathBuf,
}

impl RuntimeConfig {
    pub fn load() -> Self {
        let path = active_config_path();
        let Some(mut config) = fs::read_to_string(&path)
            .ok()
            .and_then(|contents| toml::from_str::<Self>(&contents).ok())
        else {
            return Self::default_for_platform();
        };
        let base = path.parent().unwrap_or(Path::new("."));
        for key in DependencyKey::ALL {
            let value = config.path_mut(key);
            if value.is_relative() {
                *value = normalize(&base.join(&*value));
            }
        }
        config
    }

    pub fn save(&self) -> Result<(), String> {
        let path = settings::runtime_config_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Cannot create configuration directory: {error}"))?;
        }
        let contents = toml::to_string_pretty(self)
            .map_err(|error| format!("Cannot serialize bibiocr.toml: {error}"))?;
        fs::write(&path, contents)
            .map_err(|error| format!("Cannot save {}: {error}", path.display()))
    }

    pub fn default_for_platform() -> Self {
        let root = settings::runtime_directory();
        Self {
            dependencies: DependencyPaths {
                vlm_model: root.join("models/PaddleOCR-VL-1.6-GGUF.gguf"),
                mmproj: root.join("models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"),
                layout_model: root.join("layout/inference.onnx"),
                ort_dll: root.join("onnxruntime").join(ort_library_name()),
                llama_server: root.join("llama").join(executable_name("llama-server")),
            },
            tools: ToolPaths {
                pandoc: root.join("pandoc").join(executable_name("pandoc")),
            },
        }
    }

    pub fn path(&self, key: DependencyKey) -> &Path {
        match key {
            DependencyKey::VlmModel => &self.dependencies.vlm_model,
            DependencyKey::Mmproj => &self.dependencies.mmproj,
            DependencyKey::LayoutModel => &self.dependencies.layout_model,
            DependencyKey::OrtDll => &self.dependencies.ort_dll,
            DependencyKey::LlamaServer => &self.dependencies.llama_server,
            DependencyKey::Pandoc => &self.tools.pandoc,
        }
    }

    pub fn set_path(&mut self, key: DependencyKey, path: PathBuf) {
        *self.path_mut(key) = path;
    }

    pub fn missing(&self) -> Vec<DependencyKey> {
        DependencyKey::ALL
            .into_iter()
            .filter(|key| !self.path(*key).is_file())
            .collect()
    }

    fn path_mut(&mut self, key: DependencyKey) -> &mut PathBuf {
        match key {
            DependencyKey::VlmModel => &mut self.dependencies.vlm_model,
            DependencyKey::Mmproj => &mut self.dependencies.mmproj,
            DependencyKey::LayoutModel => &mut self.dependencies.layout_model,
            DependencyKey::OrtDll => &mut self.dependencies.ort_dll,
            DependencyKey::LlamaServer => &mut self.dependencies.llama_server,
            DependencyKey::Pandoc => &mut self.tools.pandoc,
        }
    }
}

pub fn active_config_path() -> PathBuf {
    let user = settings::runtime_config_path();
    if user.is_file() {
        return user;
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(directory) = executable.parent()
    {
        let packaged = directory.join("bibiocr.toml");
        if packaged.is_file() {
            return packaged;
        }
    }
    user
}

pub fn executable_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

pub fn ort_library_name() -> &'static str {
    if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

fn normalize(path: &Path) -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::{DependencyKey, RuntimeConfig, executable_name, ort_library_name};

    #[test]
    fn platform_defaults_cover_every_dependency() {
        let config = RuntimeConfig::default_for_platform();
        for key in DependencyKey::ALL {
            assert!(!config.path(key).as_os_str().is_empty());
        }
        assert!(config.dependencies.ort_dll.ends_with(ort_library_name()));
        assert!(
            config
                .dependencies
                .llama_server
                .ends_with(executable_name("llama-server"))
        );
    }
}
