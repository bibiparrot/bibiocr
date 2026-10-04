use crate::settings::OcrEngine;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanStage {
    Captured,
    LayoutDone,
    #[default]
    Complete,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScanRecord {
    pub id: u64,
    pub title: String,
    pub created_unix: u64,
    pub image_path: PathBuf,
    pub markdown: String,
    #[serde(default)]
    pub stage: ScanStage,
    #[serde(default)]
    pub tts_revision: u64,
    #[serde(default = "legacy_ocr_engine")]
    pub ocr_engine: OcrEngine,
}

fn legacy_ocr_engine() -> OcrEngine {
    OcrEngine::PaddleVl16
}

#[derive(Default, Deserialize, Serialize)]
struct HistoryFile {
    records: Vec<ScanRecord>,
}

#[must_use]
pub fn load(path: &Path) -> Vec<ScanRecord> {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str::<HistoryFile>(&text).ok())
        .map_or_else(Vec::new, |store| store.records)
}

pub fn save(path: &Path, records: &[ScanRecord]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let text = toml::to_string_pretty(&HistoryFile {
        records: records.to_vec(),
    })
    .map_err(|error| error.to_string())?;
    fs::write(path, text).map_err(|error| error.to_string())
}

pub fn add_result(records: &mut Vec<ScanRecord>, id: u64, image_path: PathBuf, markdown: String) {
    let title = result_title(&markdown);
    records.insert(
        0,
        ScanRecord {
            id,
            title,
            created_unix: id,
            image_path,
            markdown,
            stage: ScanStage::Complete,
            tts_revision: 0,
            ocr_engine: OcrEngine::PaddleV6,
        },
    );
}

fn result_title(markdown: &str) -> String {
    markdown
        .lines()
        .map(|line| line.trim().trim_start_matches('#').trim())
        .find(|line| !line.is_empty())
        .unwrap_or("OCR")
        .chars()
        .take(60)
        .collect()
}

pub fn add_pending(
    records: &mut Vec<ScanRecord>,
    id: u64,
    image_path: PathBuf,
    ocr_engine: OcrEngine,
) {
    let title = image_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Photo".to_owned());
    records.insert(
        0,
        ScanRecord {
            id,
            title,
            created_unix: id,
            image_path,
            markdown: String::new(),
            stage: ScanStage::Captured,
            tts_revision: 0,
            ocr_engine,
        },
    );
}

pub fn set_stage(records: &mut [ScanRecord], id: u64, stage: ScanStage) {
    if let Some(record) = records.iter_mut().find(|record| record.id == id) {
        record.stage = stage;
    }
}

pub fn complete(records: &mut [ScanRecord], id: u64, markdown: String) {
    if let Some(record) = records.iter_mut().find(|record| record.id == id) {
        record.title = result_title(&markdown);
        record.markdown = markdown;
        record.stage = ScanStage::Complete;
    }
}

pub fn remove(records: &mut Vec<ScanRecord>, id: u64) -> bool {
    let before = records.len();
    records.retain(|record| record.id != id);
    records.len() != before
}

pub fn matches_query(record: &ScanRecord, query: &str) -> bool {
    let query = query.trim().to_lowercase();
    query.is_empty()
        || record.title.to_lowercase().contains(&query)
        || record.markdown.to_lowercase().contains(&query)
        || record
            .image_path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().to_lowercase().contains(&query))
}

#[cfg(test)]
mod tests {
    use super::{ScanRecord, ScanStage, add_result, load, matches_query, remove, save};
    use crate::settings::OcrEngine;

    #[test]
    fn search_finds_document_text_and_image_name() {
        let record = ScanRecord {
            id: 1,
            title: "Quarterly report".to_owned(),
            created_unix: 1,
            image_path: "meeting-photo.png".into(),
            markdown: "# Notes\n\n合同金额 10 元".to_owned(),
            stage: ScanStage::Complete,
            tts_revision: 0,
            ocr_engine: OcrEngine::PaddleV6,
        };
        assert!(matches_query(&record, "quarterly"));
        assert!(matches_query(&record, "合同金额"));
        assert!(matches_query(&record, "PHOTO"));
        assert!(!matches_query(&record, "invoice"));
    }

    #[test]
    fn real_results_persist_and_can_be_deleted() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-history-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = root.join("history.toml");
        let mut records = Vec::<ScanRecord>::new();
        add_result(
            &mut records,
            7,
            root.join("photo.jpg"),
            "# Invoice\n\nTotal: 10".to_owned(),
        );
        records[0].tts_revision = 1;
        save(&path, &records).unwrap();
        let mut loaded = load(&path);
        assert_eq!(loaded[0].title, "Invoice");
        assert_eq!(loaded[0].tts_revision, 1);
        assert!(remove(&mut loaded, 7));
        assert!(loaded.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_scan_survives_restart_and_completes_once() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-pending-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = root.join("history.toml");
        let mut records = Vec::new();
        super::add_pending(
            &mut records,
            17,
            root.join("photo.jpg"),
            OcrEngine::PaddleV6,
        );
        save(&path, &records).unwrap();
        let mut restored = load(&path);
        assert_eq!(restored[0].stage, ScanStage::Captured);
        super::set_stage(&mut restored, 17, ScanStage::LayoutDone);
        save(&path, &restored).unwrap();
        let mut restored = load(&path);
        assert_eq!(restored[0].stage, ScanStage::LayoutDone);
        super::complete(&mut restored, 17, "# Done".to_owned());
        save(&path, &restored).unwrap();
        let restored = load(&path);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].stage, ScanStage::Complete);
        assert_eq!(restored[0].title, "Done");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn old_history_records_default_to_complete() {
        let text = r##"[[records]]
id = 1
title = "Old"
created_unix = 1
image_path = "old.jpg"
markdown = "# Old"
"##;
        let records: super::HistoryFile = toml::from_str(text).unwrap();
        assert_eq!(records.records[0].stage, ScanStage::Complete);
        assert_eq!(records.records[0].tts_revision, 0);
        assert_eq!(records.records[0].ocr_engine, OcrEngine::PaddleVl16);
    }

    #[test]
    fn pending_scan_remembers_which_ocr_engine_was_selected() {
        let mut records = Vec::new();
        super::add_pending(&mut records, 42, "photo.jpg".into(), OcrEngine::PaddleV6);
        let saved = toml::to_string(&super::HistoryFile { records }).unwrap();
        let restored: super::HistoryFile = toml::from_str(&saved).unwrap();
        assert_eq!(restored.records[0].ocr_engine, OcrEngine::PaddleV6);
    }
}
