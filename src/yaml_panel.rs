use eframe::egui::{self, Color32, RichText};
use std::{fs, path::PathBuf};

#[derive(Default)]
pub struct YamlPanel {
    path: Option<PathBuf>,
    contents: String,
    dirty: bool,
    status: String,
    failed: bool,
}

impl YamlPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.dirty, egui::Button::new(rust_i18n::t!("yaml_open")))
                .clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("YAML", &["yaml", "yml"])
                    .pick_file()
            {
                self.load(path);
            }
            if ui
                .add_enabled(
                    self.path.is_some(),
                    egui::Button::new(rust_i18n::t!("yaml_save")),
                )
                .clicked()
            {
                self.save();
            }
            if ui.button(rust_i18n::t!("yaml_save_as")).clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("YAML", &["yaml", "yml"])
                    .set_file_name("config.yaml")
                    .save_file()
            {
                self.path = Some(path);
                self.save();
            }
            let path = self
                .path
                .as_deref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| rust_i18n::t!("yaml_empty").into_owned());
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

    fn load(&mut self, path: PathBuf) {
        match fs::read_to_string(&path) {
            Ok(contents) => {
                self.path = Some(path);
                self.contents = contents;
                self.dirty = false;
                self.status.clear();
                self.failed = false;
            }
            Err(error) => {
                self.status = rust_i18n::t!("yaml_read_error", error = error).into_owned();
                self.failed = true;
            }
        }
    }

    fn save(&mut self) {
        let Some(path) = &self.path else {
            return;
        };
        match fs::write(path, &self.contents) {
            Ok(()) => {
                self.dirty = false;
                self.status = rust_i18n::t!("yaml_saved", path = path.display()).into_owned();
                self.failed = false;
            }
            Err(error) => {
                self.status = rust_i18n::t!("yaml_write_error", error = error).into_owned();
                self.failed = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::YamlPanel;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn yaml_file_round_trip() {
        let path = std::env::temp_dir().join(format!(
            "bibiocr-yaml-{}-{}.yaml",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut panel = YamlPanel {
            path: Some(path.clone()),
            contents: "language: zh-CN\n".to_owned(),
            dirty: true,
            ..Default::default()
        };
        panel.save();
        panel.contents.clear();
        panel.load(path.clone());
        assert_eq!(panel.contents, "language: zh-CN\n");
        assert!(!panel.dirty);
        let _ = std::fs::remove_file(path);
    }
}
