use crate::settings::AppSettings;

#[derive(Clone, Copy)]
pub struct SupportedLocale {
    pub code: &'static str,
    pub native_name: &'static str,
}

pub const SUPPORTED_LOCALES: &[SupportedLocale] = &[
    SupportedLocale {
        code: "en",
        native_name: "English",
    },
    SupportedLocale {
        code: "zh-CN",
        native_name: "中文",
    },
    SupportedLocale {
        code: "ja",
        native_name: "日本語",
    },
    SupportedLocale {
        code: "la",
        native_name: "Latina",
    },
    SupportedLocale {
        code: "ko",
        native_name: "한국어",
    },
    SupportedLocale {
        code: "ru",
        native_name: "Русский",
    },
    SupportedLocale {
        code: "fr",
        native_name: "Français",
    },
    SupportedLocale {
        code: "es",
        native_name: "Español",
    },
];

pub struct LocaleManager {
    selected: String,
    active: &'static str,
}

impl LocaleManager {
    pub fn from_settings(settings: &AppSettings) -> Self {
        let selected = settings.locale.clone();
        let active = if selected == "system" {
            system_locale()
        } else {
            normalize(&selected)
        };
        rust_i18n::set_locale(active);
        Self { selected, active }
    }

    pub fn selected(&self) -> &str {
        &self.selected
    }

    pub fn active(&self) -> &'static str {
        self.active
    }

    pub fn set(&mut self, selected: &str, settings: &mut AppSettings) {
        self.selected = selected.to_owned();
        self.active = if selected == "system" {
            system_locale()
        } else {
            normalize(selected)
        };
        settings.locale.clone_from(&self.selected);
        rust_i18n::set_locale(self.active);
        let _ = settings.save();
    }
}

fn system_locale() -> &'static str {
    sys_locale::get_locale()
        .as_deref()
        .map(normalize)
        .unwrap_or("en")
}

fn normalize(locale: &str) -> &'static str {
    let lower = locale.to_ascii_lowercase().replace('_', "-");
    if lower.starts_with("zh") {
        "zh-CN"
    } else if lower.starts_with("ja") {
        "ja"
    } else if lower.starts_with("la") {
        "la"
    } else if lower.starts_with("ko") {
        "ko"
    } else if lower.starts_with("ru") {
        "ru"
    } else if lower.starts_with("fr") {
        "fr"
    } else if lower.starts_with("es") {
        "es"
    } else {
        "en"
    }
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn normalizes_bcp47_locales() {
        assert_eq!(normalize("zh_TW"), "zh-CN");
        assert_eq!(normalize("fr-CA"), "fr");
        assert_eq!(normalize("ko-KR"), "ko");
        assert_eq!(normalize("unknown"), "en");
    }
}
