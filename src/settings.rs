use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProxyMode {
    None,
    System,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// `system` follows the operating-system locale; any other value is a
    /// supported rust-i18n locale code.
    pub locale: String,
    pub last_input_dir: Option<PathBuf>,
    pub use_proxy: bool,
    pub use_system_proxy: bool,
    pub proxy: String,
    pub use_hf_mirror: bool,
    pub hf_endpoint: String,
    pub use_github_proxy: bool,
    pub github_proxy: String,
    pub resume_downloads: bool,
    pub download_retries: u32,
}

impl Default for AppSettings {
    fn default() -> Self {
        let chinese = sys_locale::get_locale().is_some_and(|value| value.starts_with("zh"));
        let hf_endpoint = std::env::var("HF_ENDPOINT").ok();
        Self {
            locale: "system".to_owned(),
            last_input_dir: None,
            use_proxy: false,
            use_system_proxy: true,
            proxy: String::new(),
            use_hf_mirror: hf_endpoint.is_some() || chinese,
            hf_endpoint: hf_endpoint.unwrap_or_else(|| {
                if chinese {
                    "https://hf-mirror.com"
                } else {
                    "https://huggingface.co"
                }
                .to_owned()
            }),
            use_github_proxy: chinese,
            github_proxy: if chinese {
                "https://gh-proxy.com/${giturl}".to_owned()
            } else {
                String::new()
            },
            resume_downloads: true,
            download_retries: 3,
        }
    }
}

impl AppSettings {
    pub fn proxy_mode(&self) -> ProxyMode {
        if self.use_proxy {
            ProxyMode::Manual
        } else if self.use_system_proxy {
            ProxyMode::System
        } else {
            ProxyMode::None
        }
    }

    pub fn set_proxy_mode(&mut self, mode: ProxyMode) {
        self.use_proxy = mode == ProxyMode::Manual;
        self.use_system_proxy = mode == ProxyMode::System;
    }

    pub fn load() -> Self {
        let path = settings_path();
        fs::read_to_string(path)
            .ok()
            .and_then(|contents| toml::from_str(&contents).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        let path = settings_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Cannot create settings directory: {error}"))?;
        }
        let contents = toml::to_string_pretty(self)
            .map_err(|error| format!("Cannot serialize settings: {error}"))?;
        fs::write(path, contents).map_err(|error| format!("Cannot save settings: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{AppSettings, ProxyMode};

    #[test]
    fn proxy_modes_are_mutually_exclusive() {
        let mut settings = AppSettings::default();
        for mode in [ProxyMode::None, ProxyMode::System, ProxyMode::Manual] {
            settings.set_proxy_mode(mode);
            assert_eq!(settings.proxy_mode(), mode);
        }
    }
}

pub fn settings_path() -> PathBuf {
    std::env::var_os("BIBIOCR_SETTINGS")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_settings_dir().join("settings.toml"))
}

pub fn runtime_config_path() -> PathBuf {
    std::env::var_os("BIBIOCR_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_settings_dir().join("bibiocr.toml"))
}

pub fn runtime_directory() -> PathBuf {
    default_settings_dir().join("runtime")
}

fn default_settings_dir() -> PathBuf {
    if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("BIBIOCR")
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("Library/Application Support/BIBIOCR")
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
            .unwrap_or_else(std::env::temp_dir)
            .join("bibiocr")
    }
}
