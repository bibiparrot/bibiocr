use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub struct LocaleManager;

impl LocaleManager {
    pub fn choices() -> &'static [(&'static str, &'static str)] {
        &[
            ("system", ""),
            ("zh-CN", "简体中文"),
            ("en", "English"),
            ("ja", "日本語"),
            ("ko", "한국어"),
            ("ru", "Русский"),
            ("fr", "Français"),
            ("es", "Español"),
            ("de", "Deutsch"),
            ("it", "Italiano"),
            ("pt", "Português"),
        ]
    }

    #[must_use]
    pub fn resolve(locale: &str) -> &'static str {
        match locale
            .split(['-', '_'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "zh" => "zh-CN",
            "ja" => "ja",
            "ko" => "ko",
            "ru" => "ru",
            "fr" => "fr",
            "es" => "es",
            "de" => "de",
            "it" => "it",
            "pt" => "pt",
            _ => "en",
        }
    }

    pub fn apply(language: &str) -> &'static str {
        let requested = if language.eq_ignore_ascii_case("system") {
            sys_locale::get_locale().unwrap_or_else(|| "en".to_owned())
        } else {
            language.to_owned()
        };
        let locale = Self::resolve(&requested);
        rust_i18n::set_locale(locale);
        locale
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OcrEngine {
    #[default]
    PaddleV6,
    PaddleVl16,
}

impl OcrEngine {
    pub const fn label(self) -> &'static str {
        match self {
            Self::PaddleV6 => "PP-OCRv6 · ONNX",
            Self::PaddleVl16 => "PaddleOCR-VL 1.6",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TtsEngine {
    #[default]
    Melo,
    Kokoro,
}

impl TtsEngine {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Melo => "Melo · ZH/EN",
            Self::Kokoro => "Kokoro · EN",
        }
    }

    pub const fn cache_name(self) -> &'static str {
        match self {
            Self::Melo => "melo",
            Self::Kokoro => "kokoro",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Settings {
    pub locale: LocaleSettings,
    pub download: DownloadSettings,
    pub ocr_engine: OcrEngine,
    pub tts_engine: TtsEngine,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            locale: LocaleSettings::default(),
            download: DownloadSettings::default(),
            ocr_engine: OcrEngine::default(),
            tts_engine: TtsEngine::default(),
        }
    }
}

impl Settings {
    #[must_use]
    pub fn load_or_default(path: &Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|source| toml::from_str(&source).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let source = toml::to_string_pretty(self).map_err(|error| error.to_string())?;
        fs::write(path, source).map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct LocaleSettings {
    pub language: String,
}

impl Default for LocaleSettings {
    fn default() -> Self {
        Self {
            language: "system".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct DownloadSettings {
    pub resume: bool,
    pub wifi_only: bool,
    pub hf_endpoint: String,
    pub github_proxy: String,
}

impl Default for DownloadSettings {
    fn default() -> Self {
        Self {
            resume: true,
            wifi_only: false,
            hf_endpoint: String::new(),
            github_proxy: String::new(),
        }
    }
}

impl DownloadSettings {
    #[must_use]
    pub fn resolve_url(&self, source: &str, locale: &str, env_endpoint: Option<&str>) -> String {
        if source.starts_with("https://huggingface.co/") {
            let endpoint = env_endpoint
                .filter(|value| !value.trim().is_empty())
                .or_else(|| {
                    (!self.hf_endpoint.trim().is_empty()).then_some(self.hf_endpoint.as_str())
                })
                .unwrap_or(if locale == "zh-CN" {
                    "https://hf-mirror.com"
                } else {
                    "https://huggingface.co"
                })
                .trim_end_matches('/');
            return source.replacen("https://huggingface.co", endpoint, 1);
        }
        if (source.starts_with("https://github.com/")
            || source.starts_with("https://raw.githubusercontent.com/"))
            && !self.github_proxy.trim().is_empty()
        {
            for placeholder in ["${giturl}", "{$giturl}", "{giturl}"] {
                if self.github_proxy.contains(placeholder) {
                    return self.github_proxy.replace(placeholder, source);
                }
            }
        }
        source.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{DownloadSettings, LocaleManager, OcrEngine, Settings, TtsEngine};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn locale_manager_maps_system_tags_to_supported_translations() {
        assert_eq!(LocaleManager::resolve("zh-Hans-CN"), "zh-CN");
        assert_eq!(LocaleManager::resolve("ja-JP"), "ja");
        assert_eq!(LocaleManager::resolve("ko-KR"), "ko");
        assert_eq!(LocaleManager::resolve("ru-RU"), "ru");
        assert_eq!(LocaleManager::resolve("fr-FR"), "fr");
        assert_eq!(LocaleManager::resolve("es-MX"), "es");
        assert_eq!(LocaleManager::resolve("pt-BR"), "pt");
        assert_eq!(LocaleManager::resolve("unsupported"), "en");
    }

    #[test]
    fn language_picker_starts_with_system_and_lists_every_translation() {
        assert_eq!(Settings::default().locale.language, "system");
        let choices = LocaleManager::choices();
        assert_eq!(choices[0].0, "system");
        let mut selectable: Vec<_> = choices.iter().skip(1).map(|choice| choice.0).collect();
        selectable.sort_unstable();
        let mut embedded = rust_i18n::available_locales!();
        embedded.sort_unstable();
        assert_eq!(selectable, embedded);
    }

    #[test]
    fn chinese_uses_hf_mirror_unless_endpoint_is_configured() {
        let settings = DownloadSettings::default();
        let source = "https://huggingface.co/PaddlePaddle/model/resolve/main/model.gguf";

        assert_eq!(
            settings.resolve_url(source, "zh-CN", None),
            "https://hf-mirror.com/PaddlePaddle/model/resolve/main/model.gguf"
        );
        assert_eq!(settings.resolve_url(source, "en", None), source);
        assert_eq!(
            settings.resolve_url(source, "zh-CN", Some("https://hf.example")),
            "https://hf.example/PaddlePaddle/model/resolve/main/model.gguf"
        );
    }

    #[test]
    fn github_proxy_replaces_supported_giturl_placeholders() {
        let source = "https://github.com/microsoft/onnxruntime/releases/download/v1/file.zip";
        for placeholder in ["{giturl}", "${giturl}", "{$giturl}"] {
            let settings = DownloadSettings {
                github_proxy: format!("https://gh-proxy.com/{placeholder}"),
                ..DownloadSettings::default()
            };
            assert_eq!(
                settings.resolve_url(source, "en", None),
                format!("https://gh-proxy.com/{source}")
            );
        }
        let raw = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.7.0/ppocr/utils/dict/ppocrv6_dict.txt";
        let settings = DownloadSettings {
            github_proxy: "https://gh-proxy.com/{giturl}".to_owned(),
            ..DownloadSettings::default()
        };
        assert_eq!(
            settings.resolve_url(raw, "zh-CN", None),
            format!("https://gh-proxy.com/{raw}")
        );
    }

    #[test]
    fn requested_locale_catalogs_are_embedded() {
        let mut locales = rust_i18n::available_locales!();
        locales.sort_unstable();
        assert_eq!(
            locales,
            [
                "de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh-CN"
            ]
        );
    }

    #[test]
    fn settings_toml_round_trip_preserves_user_choices() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-settings-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = root.join("settings.toml");
        let mut settings = Settings::default();
        settings.locale.language = "ja".to_owned();
        settings.download.resume = false;
        settings.download.wifi_only = true;
        settings.download.github_proxy = "https://gh-proxy.org/{giturl}".to_owned();
        settings.save(&path).unwrap();

        let loaded = Settings::load_or_default(&path);
        assert_eq!(loaded.locale.language, "ja");
        assert!(!loaded.download.resume);
        assert!(loaded.download.wifi_only);
        assert_eq!(
            loaded.download.github_proxy,
            "https://gh-proxy.org/{giturl}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ocr_engine_defaults_to_v6_and_survives_settings_round_trip() {
        assert_eq!(Settings::default().ocr_engine, OcrEngine::PaddleV6);
        let legacy: Settings = toml::from_str("[locale]\nlanguage = 'en'\n").unwrap();
        assert_eq!(legacy.ocr_engine, OcrEngine::PaddleV6);
        let mut selected = Settings::default();
        selected.ocr_engine = OcrEngine::PaddleVl16;
        let restored: Settings = toml::from_str(&toml::to_string(&selected).unwrap()).unwrap();
        assert_eq!(restored.ocr_engine, OcrEngine::PaddleVl16);
    }

    #[test]
    fn tts_engine_defaults_to_melo_and_survives_settings_round_trip() {
        assert_eq!(Settings::default().tts_engine, TtsEngine::Melo);
        let legacy: Settings = toml::from_str("ocr_engine = 'paddle_v6'").unwrap();
        assert_eq!(legacy.tts_engine, TtsEngine::Melo);
        let selected: Settings = toml::from_str("tts_engine = 'kokoro'").unwrap();
        assert_eq!(selected.tts_engine, TtsEngine::Kokoro);
        assert!(
            toml::to_string(&selected)
                .unwrap()
                .contains("tts_engine = \"kokoro\"")
        );
    }
}
