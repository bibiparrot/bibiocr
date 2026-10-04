use oar_ocr::core::config::OrtSessionConfig;
use oar_ocr::prelude::OARStructureBuilder;
use std::{
    error::Error,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub fn recognize(model_dir: &Path, image_path: &Path) -> Result<String, String> {
    recognize_cancellable(model_dir, image_path, &AtomicBool::new(false))
}

pub fn recognize_cancellable(
    model_dir: &Path,
    image_path: &Path,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let _backend_guard = crate::backend_gate::lock()?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Recognition cancelled".to_owned());
    }
    let threads = std::thread::available_parallelism()
        .map_or(2, std::num::NonZero::get)
        .min(4);
    let structure = OARStructureBuilder::new(model_dir.join(crate::core::MODELS[2].file_name))
        .ort_session(
            OrtSessionConfig::new()
                .with_intra_threads(threads)
                .with_inter_threads(1),
        )
        .layout_model_name("PP-DocLayoutV3")
        .with_text_line_orientation(model_dir.join("pp-lcnet_x1_0_textline_ori.onnx"))
        .with_ocr(
            model_dir.join("pp-ocrv6_small_det.onnx"),
            model_dir.join("pp-ocrv6_small_rec.onnx"),
            model_dir.join("ppocrv6_dict.txt"),
        )
        .build()
        .map_err(|error| error_chain(&error))?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Recognition cancelled".to_owned());
    }
    let result = structure
        .predict(image_path)
        .map_err(|error| error_chain(&error))?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Recognition cancelled".to_owned());
    }
    Ok(result.to_markdown())
}

fn error_chain(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "downloads the five PP-OCRv6 assets once, then runs real ONNX inference"]
    fn real_v6_document_produces_markdown() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let models = root.join("downloads");
        let task = crate::download::start(
            models.clone(),
            Default::default(),
            "en",
            crate::settings::OcrEngine::PaddleV6,
            crate::settings::TtsEngine::Melo,
        );
        for event in task.events {
            match event {
                crate::download::DownloadEvent::Failed(_, error) => panic!("{error}"),
                crate::download::DownloadEvent::Finished => break,
                _ => {}
            }
        }
        let markdown = super::recognize(&models, &root.join("docs/text.png")).unwrap();
        assert!(markdown.contains("季度"), "{markdown}");
    }
}
