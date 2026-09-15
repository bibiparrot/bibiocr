use crate::{
    dependencies::{self, RuntimeConfig},
    settings,
};
use eframe::egui::{self, Color32, RichText};
use std::{fs, path::Path};

pub struct ConfigPanel {
    contents: String,
    dirty: bool,
    status: String,
    failed: bool,
}

impl Default for ConfigPanel {
    fn default() -> Self {
        Self::new(&RuntimeConfig::load())
    }
}

impl ConfigPanel {
    fn new(config: &RuntimeConfig) -> Self {
        let source = dependencies::active_config_path();
        let target = settings::runtime_config_path();
        let contents = if source == target {
            fs::read_to_string(source).ok()
        } else {
            None
        }
        .unwrap_or_else(|| toml::to_string_pretty(config).unwrap_or_default());
        Self {
            contents,
            dirty: false,
            status: String::new(),
            failed: false,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &mut RuntimeConfig) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.dirty,
                    egui::Button::new(rust_i18n::t!("config_reload")),
                )
                .clicked()
            {
                *self = Self::new(config);
            }
            if ui.button(rust_i18n::t!("config_save")).clicked() {
                self.save_to(&settings::runtime_config_path(), config);
            }
            let path = settings::runtime_config_path().display().to_string();
            ui.label(if self.dirty {
                format!("{path} *")
            } else {
                path
            });
        });
        if !self.status.is_empty() {
            ui.label(RichText::new(&self.status).color(if self.failed {
                Color32::DARK_RED
            } else {
                Color32::DARK_GREEN
            }));
        }
        ui.separator();
        if ui
            .add_sized(
                ui.available_size(),
                egui::TextEdit::multiline(&mut self.contents)
                    .font(egui::TextStyle::Monospace)
                    .code_editor()
                    .desired_rows(24),
            )
            .changed()
        {
            self.dirty = true;
        }
    }

    fn save_to(&mut self, path: &Path, config: &mut RuntimeConfig) {
        let parsed = match RuntimeConfig::from_toml(&self.contents, path) {
            Ok(config) => config,
            Err(error) => {
                self.status = rust_i18n::t!("config_parse_error", error = error).into_owned();
                self.failed = true;
                return;
            }
        };
        let result = path
            .parent()
            .map(fs::create_dir_all)
            .transpose()
            .and_then(|_| fs::write(path, &self.contents));
        match result {
            Ok(()) => {
                *config = parsed;
                self.dirty = false;
                self.status = rust_i18n::t!("config_saved", path = path.display()).into_owned();
                self.failed = false;
            }
            Err(error) => {
                self.status = rust_i18n::t!("config_write_error", error = error).into_owned();
                self.failed = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ConfigPanel;
    use crate::dependencies::RuntimeConfig;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn saves_valid_bibiocr_config_and_rejects_invalid_toml() {
        let path = std::env::temp_dir().join(format!(
            "bibiocr-config-{}-{}.toml",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut config = RuntimeConfig::default_for_platform();
        let mut panel = ConfigPanel::new(&config);
        panel.contents = toml::to_string_pretty(&config).unwrap();
        panel.save_to(&path, &mut config);
        assert!(path.is_file());
        let saved = std::fs::read_to_string(&path).unwrap();
        panel.contents = "[dependencies\n".to_owned();
        panel.save_to(&path, &mut config);
        assert!(panel.failed);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), saved);
        let _ = std::fs::remove_file(path);
    }
}
