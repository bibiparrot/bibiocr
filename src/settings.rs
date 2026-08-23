use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// `system` follows the operating-system locale; any other value is a
    /// supported rust-i18n locale code.
    pub locale: String,
    pub last_input_dir: Option<PathBuf>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            locale: "system".to_owned(),
            last_input_dir: None,
        }
    }
}

impl AppSettings {
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

pub fn settings_path() -> PathBuf {
    std::env::var_os("BIBIOCR_SETTINGS")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_settings_dir().join("settings.toml"))
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
