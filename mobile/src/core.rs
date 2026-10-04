use crate::settings::{OcrEngine, TtsEngine};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelGroup {
    Vl,
    V6,
    Layout,
    Melo,
    Kokoro,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    pub group: ModelGroup,
    pub name: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
    pub expected_bytes: u64,
}

pub const MODELS: [ModelSpec; 13] = [
    ModelSpec {
        group: ModelGroup::Vl,
        name: "VLM",
        file_name: "PaddleOCR-VL-1.6-GGUF.gguf",
        url: "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF.gguf",
        expected_bytes: 935_769_056,
    },
    ModelSpec {
        group: ModelGroup::Vl,
        name: "MMProj",
        file_name: "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        url: "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        expected_bytes: 881_770_560,
    },
    ModelSpec {
        group: ModelGroup::Layout,
        name: "Layout",
        file_name: "pp-doclayoutv3_onnx.onnx",
        url: "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx",
        expected_bytes: 130_502_049,
    },
    ModelSpec {
        group: ModelGroup::Melo,
        name: "Melo TTS (Chinese + English)",
        file_name: "melo-model.onnx",
        url: "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/model.onnx",
        expected_bytes: 170_429_550,
    },
    ModelSpec {
        group: ModelGroup::Melo,
        name: "Melo TTS lexicon",
        file_name: "melo-lexicon.txt",
        url: "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/lexicon.txt",
        expected_bytes: 6_837_671,
    },
    ModelSpec {
        group: ModelGroup::Melo,
        name: "Melo TTS tokens",
        file_name: "melo-tokens.txt",
        url: "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/tokens.txt",
        expected_bytes: 655,
    },
    ModelSpec {
        group: ModelGroup::V6,
        name: "PP-OCRv6 Small detector",
        file_name: "pp-ocrv6_small_det.onnx",
        url: "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/28fe5895c24fd108c19eb3e8479f4ab385fbfc62/inference.onnx",
        expected_bytes: 9_880_512,
    },
    ModelSpec {
        group: ModelGroup::V6,
        name: "PP-OCRv6 Small recognizer",
        file_name: "pp-ocrv6_small_rec.onnx",
        url: "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/b8f84f0b80c529de40b4fbb3544b84fa7233a513/inference.onnx",
        expected_bytes: 21_159_378,
    },
    ModelSpec {
        group: ModelGroup::V6,
        name: "PP-LCNet text-line orientation",
        file_name: "pp-lcnet_x1_0_textline_ori.onnx",
        url: "https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx/resolve/7fdcf3cf7061163eda7183b224aa334bd33068f7/inference.onnx",
        expected_bytes: 6_777_816,
    },
    ModelSpec {
        group: ModelGroup::V6,
        name: "PP-OCRv6 dictionary",
        file_name: "ppocrv6_dict.txt",
        url: "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.7.0/ppocr/utils/dict/ppocrv6_dict.txt",
        expected_bytes: 74_947,
    },
    ModelSpec {
        group: ModelGroup::Kokoro,
        name: "Kokoro voice af_heart",
        file_name: "af_heart.bin",
        url: "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_heart.bin",
        expected_bytes: 522_240,
    },
    ModelSpec {
        group: ModelGroup::Kokoro,
        name: "Kokoro ONNX",
        file_name: "kokoro-model.onnx",
        url: "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx",
        expected_bytes: 325_532_232,
    },
    ModelSpec {
        group: ModelGroup::Kokoro,
        name: "Kokoro tokenizer",
        file_name: "kokoro-tokenizer.json",
        url: "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/tokenizer.json",
        expected_bytes: 3_497,
    },
];

pub fn required_models(
    engine: OcrEngine,
    tts_engine: TtsEngine,
) -> impl Iterator<Item = (usize, &'static ModelSpec)> {
    MODELS.iter().enumerate().filter(move |(_, model)| {
        model.group == ModelGroup::Layout
            || matches!(
                (engine, model.group),
                (OcrEngine::PaddleV6, ModelGroup::V6) | (OcrEngine::PaddleVl16, ModelGroup::Vl)
            )
            || matches!(
                (tts_engine, model.group),
                (TtsEngine::Melo, ModelGroup::Melo) | (TtsEngine::Kokoro, ModelGroup::Kokoro)
            )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Download,
    History,
    Capture,
    Recognizing,
    Result,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SystemInsetsPx {
    pub top: f32,
    pub bottom: f32,
}

impl SystemInsetsPx {
    #[must_use]
    pub fn from_content_bounds(window_height: i32, content_top: i32, content_bottom: i32) -> Self {
        if window_height <= 0 || content_bottom <= content_top {
            return Self::default();
        }
        let top = content_top.clamp(0, window_height);
        let bottom = window_height - content_bottom.clamp(0, window_height);
        Self {
            top: top as f32,
            bottom: bottom as f32,
        }
    }

    #[must_use]
    pub fn runtime_or_fallback(runtime: Option<Self>, fallback: Self) -> Self {
        runtime
            .filter(|insets| insets.top > 0.0 || insets.bottom > 0.0)
            .unwrap_or(fallback)
    }

    #[must_use]
    pub fn with_minimum(self, screen_height: f32) -> Self {
        let minimum = screen_height * 0.08;
        Self {
            top: self.top.max(minimum),
            bottom: self.bottom.max(minimum),
        }
    }
}

pub fn initial_screen(model_dir: &Path, engine: OcrEngine, tts_engine: TtsEngine) -> Screen {
    if required_models(engine, tts_engine).all(|(_, model)| {
        model_dir
            .join(model.file_name)
            .metadata()
            .is_ok_and(|metadata| metadata.len() == model.expected_bytes)
    }) {
        Screen::History
    } else {
        Screen::Download
    }
}

#[cfg(test)]
mod tests {
    use super::{MODELS, Screen, SystemInsetsPx, initial_screen};
    use crate::settings::{OcrEngine, TtsEngine};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn download_catalog_includes_bilingual_melo_tts_assets() {
        for file_name in ["melo-model.onnx", "melo-lexicon.txt", "melo-tokens.txt"] {
            assert!(MODELS.iter().any(|model| model.file_name == file_name));
        }
        assert!(MODELS.iter().any(|model| model.name.starts_with("Kokoro")));
    }

    #[test]
    fn selected_ocr_engine_requires_only_its_models_and_shared_tts() {
        let v6 = super::required_models(OcrEngine::PaddleV6, TtsEngine::Melo)
            .map(|(_, model)| model.file_name)
            .collect::<Vec<_>>();
        assert!(v6.contains(&"pp-ocrv6_small_det.onnx"));
        assert!(v6.contains(&"pp-ocrv6_small_rec.onnx"));
        assert!(v6.contains(&"ppocrv6_dict.txt"));
        assert!(v6.contains(&"pp-doclayoutv3_onnx.onnx"));
        assert!(!v6.contains(&"pp-doclayout_plus-l.onnx"));
        assert_eq!(
            v6.iter()
                .filter(|name| **name == "pp-doclayoutv3_onnx.onnx")
                .count(),
            1
        );
        assert!(v6.contains(&"melo-model.onnx"));
        assert!(!v6.iter().any(|name| name.ends_with(".gguf")));

        let vl = super::required_models(OcrEngine::PaddleVl16, TtsEngine::Melo)
            .map(|(_, model)| model.file_name)
            .collect::<Vec<_>>();
        assert!(vl.contains(&"PaddleOCR-VL-1.6-GGUF.gguf"));
        assert!(vl.contains(&"melo-model.onnx"));
        assert!(!vl.contains(&"pp-ocrv6_small_det.onnx"));
    }

    #[test]
    fn selected_tts_engine_requires_only_its_own_three_files() {
        let melo = super::required_models(OcrEngine::PaddleV6, TtsEngine::Melo)
            .map(|(_, model)| model.file_name)
            .collect::<Vec<_>>();
        assert!(melo.contains(&"melo-model.onnx"));
        assert!(!melo.contains(&"kokoro-model.onnx"));
        let kokoro = super::required_models(OcrEngine::PaddleV6, TtsEngine::Kokoro)
            .map(|(_, model)| model.file_name)
            .collect::<Vec<_>>();
        for name in ["af_heart.bin", "kokoro-model.onnx", "kokoro-tokenizer.json"] {
            assert!(kokoro.contains(&name));
        }
        assert!(!kokoro.contains(&"melo-model.onnx"));
    }

    #[test]
    fn installed_vl_files_do_not_block_v6_and_vice_versa() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-selected-models-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        for (_, model) in super::required_models(OcrEngine::PaddleV6, TtsEngine::Melo) {
            fs::File::create(root.join(model.file_name))
                .unwrap()
                .set_len(model.expected_bytes)
                .unwrap();
        }
        assert_eq!(
            initial_screen(&root, OcrEngine::PaddleV6, TtsEngine::Melo),
            Screen::History
        );
        assert_eq!(
            initial_screen(&root, OcrEngine::PaddleVl16, TtsEngine::Melo),
            Screen::Download
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn first_run_opens_download_until_all_models_exist() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-mobile-models-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();

        assert_eq!(
            initial_screen(&root, OcrEngine::PaddleV6, TtsEngine::Melo),
            Screen::Download
        );

        for model in MODELS {
            let file = fs::File::create(root.join(model.file_name)).unwrap();
            file.set_len(model.expected_bytes).unwrap();
        }
        assert_eq!(
            initial_screen(&root, OcrEngine::PaddleV6, TtsEngine::Melo),
            Screen::History
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_model_does_not_count_as_installed() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-mobile-partial-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        for model in MODELS {
            let file = fs::File::create(root.join(model.file_name)).unwrap();
            file.set_len(model.expected_bytes).unwrap();
        }
        fs::File::create(root.join(MODELS[6].file_name))
            .unwrap()
            .set_len(1)
            .unwrap();

        assert_eq!(
            initial_screen(&root, OcrEngine::PaddleV6, TtsEngine::Melo),
            Screen::Download
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn safe_area_comes_from_the_runtime_content_rectangle() {
        assert_eq!(
            SystemInsetsPx::from_content_bounds(2_772, 96, 2_704),
            SystemInsetsPx {
                top: 96.0,
                bottom: 68.0,
            }
        );
        assert_eq!(
            SystemInsetsPx::from_content_bounds(0, 0, 0),
            SystemInsetsPx::default()
        );
    }

    #[test]
    fn empty_runtime_insets_do_not_erase_the_last_safe_area() {
        let previous = SystemInsetsPx {
            top: 96.0,
            bottom: 68.0,
        };
        assert_eq!(
            SystemInsetsPx::runtime_or_fallback(Some(SystemInsetsPx::default()), previous),
            previous
        );
        assert_eq!(
            SystemInsetsPx::runtime_or_fallback(
                Some(SystemInsetsPx {
                    top: 108.0,
                    bottom: 72.0,
                }),
                previous,
            ),
            SystemInsetsPx {
                top: 108.0,
                bottom: 72.0,
            }
        );
    }

    #[test]
    fn safe_area_keeps_eight_percent_on_both_edges() {
        let insets = SystemInsetsPx {
            top: 100.0,
            bottom: 200.0,
        }
        .with_minimum(2000.0);
        assert_eq!(insets.top, 160.0);
        assert_eq!(insets.bottom, 200.0);
    }
}
