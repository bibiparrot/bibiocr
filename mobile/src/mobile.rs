//! Mobile-first UI for download -> camera/gallery -> OCR -> persisted results.

use crate::{
    core::{self, MODELS},
    download::{self, DownloadEvent, DownloadTask},
    history::{self, ScanRecord, ScanStage},
    settings::{DownloadSettings, LocaleManager, OcrEngine, Settings, TtsEngine},
};
use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Id, Margin, Pos2, Rect, Sense, Stroke, StrokeKind,
    Vec2,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::{SystemTime, UNIX_EPOCH},
};

const INK: Color32 = Color32::from_rgb(0x22, 0x27, 0x35);
const MUTED: Color32 = Color32::from_rgb(0x8b, 0x90, 0x9b);
const PAGE_BG: Color32 = Color32::from_rgb(0xf6, 0xf4, 0xf0);
const CARD: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
const LINE: Color32 = Color32::from_rgb(0xe9, 0xe5, 0xdd);
const CHIP: Color32 = Color32::from_rgb(0xf1, 0xef, 0xea);
const INDIGO: Color32 = Color32::from_rgb(0x3b, 0x4b, 0xdb);
const ORANGE: Color32 = Color32::from_rgb(0xee, 0x72, 0x33);
const ORANGE_SOFT: Color32 = Color32::from_rgb(0xfc, 0xe9, 0xdc);
const GREEN: Color32 = Color32::from_rgb(0x2f, 0xa5, 0x4f);
const READ_HIGHLIGHT: Color32 = Color32::from_rgb(0xdc, 0xf5, 0xe2);
const WHITE: Color32 = Color32::WHITE;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Download,
    History,
    Settings,
    Recognizing,
    Result,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResultTab {
    Preview,
    Edit,
    Compare,
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
#[derive(Clone, Copy)]
enum TextTarget {
    Search,
    Markdown,
}

#[derive(Clone, Copy)]
enum Icon {
    Back,
    Close,
    Dots,
    Gear,
    Search,
    Camera,
    Gallery,
    Download,
    Trash,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelStatus {
    Waiting,
    Downloading,
    Complete,
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepState {
    Done,
    Active,
    Pending,
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
enum AppEvent {
    Image(Result<Option<PathBuf>, String>),
    OriginalImage(PathBuf, Result<([usize; 2], [usize; 2], Vec<u8>), String>),
    TextInput(TextTarget, Result<Option<String>, String>),
    Notice(Result<String, String>),
    TtsSentence(usize),
    TtsSentenceDone(usize),
    Tts(Result<(), String>),
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
enum RecognitionEvent {
    Layout,
    Ocr,
    Token(String),
    Complete(String),
    Failed(String),
}

pub struct MobileApp {
    screen: Screen,
    last_rendered_screen: Screen,
    search: String,
    recognition_progress: f32,
    recognition_text: String,
    recognition_error: Option<String>,
    recognition_active: bool,
    result_tab: ResultTab,
    markdown: String,
    data_dir: PathBuf,
    model_dir: PathBuf,
    history_path: PathBuf,
    records: Vec<ScanRecord>,
    current_image: Option<PathBuf>,
    current_record_id: Option<u64>,
    original_texture: Option<egui::TextureHandle>,
    original_size: Option<[usize; 2]>,
    original_loading: bool,
    text_dialog_open: bool,
    app_events: Receiver<AppEvent>,
    app_sender: Sender<AppEvent>,
    recognition_events: Receiver<(u64, RecognitionEvent)>,
    recognition_sender: Sender<(u64, RecognitionEvent)>,
    recognition_generation: u64,
    recognition_cancel: Arc<AtomicBool>,
    notice: Option<String>,
    camera_open: bool,
    download_task: Option<DownloadTask>,
    model_progress: [(u64, u64); MODELS.len()],
    model_status: [ModelStatus; MODELS.len()],
    download_error: Option<String>,
    download_settings: DownloadSettings,
    ocr_engine: OcrEngine,
    download_view_engine: OcrEngine,
    tts_engine: TtsEngine,
    download_view_tts_engine: TtsEngine,
    locale: &'static str,
    language_choice: String,
    tts_active: bool,
    tts_sentences: Vec<String>,
    tts_current: Option<usize>,
    tts_done: usize,
    tts_preparing: bool,
    tts_scroll_to_current: bool,
    tts_speed: f32,
    tts_volume: f32,
    tts_stop: Arc<AtomicBool>,
    tts_volume_shared: Arc<AtomicU32>,
    tts_speed_shared: Arc<AtomicU32>,
    system_insets_px: core::SystemInsetsPx,
    swipe_start: Option<Pos2>,
    #[cfg(target_os = "android")]
    android_app: Option<android_activity::AndroidApp>,
}

impl MobileApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_cjk_font(&cc.egui_ctx);
        // The mobile flow is designed on light surfaces; egui defaults to dark.
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        let configured_data = std::env::var_os("BIBIOCR_DATA").map(PathBuf::from);
        let data_dir = configured_data
            .clone()
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        let settings_path = data_dir.join("settings.toml");
        let settings = Settings::load_or_default(&settings_path);
        if !settings_path.exists() {
            let _ = settings.save(&settings_path);
        }
        let locale = LocaleManager::apply(&settings.locale.language);
        let language_choice = if settings.locale.language.eq_ignore_ascii_case("system") {
            "system".to_owned()
        } else {
            LocaleManager::resolve(&settings.locale.language).to_owned()
        };
        let download_settings = settings.download;
        let model_dir =
            configured_data.map_or_else(|| data_dir.join("downloads"), |path| path.join("models"));
        download::cleanup_obsolete_layout(&model_dir);
        let first_screen =
            if core::initial_screen(&model_dir, settings.ocr_engine, settings.tts_engine)
                == core::Screen::Download
            {
                Screen::Download
            } else {
                Screen::History
            };
        let model_status = std::array::from_fn(|index| {
            if download::installed(&model_dir, &MODELS[index]) {
                ModelStatus::Complete
            } else {
                ModelStatus::Waiting
            }
        });
        let model_progress = std::array::from_fn(|index| {
            if model_status[index] == ModelStatus::Complete {
                (MODELS[index].expected_bytes, MODELS[index].expected_bytes)
            } else {
                (0, MODELS[index].expected_bytes)
            }
        });
        let download_task = (first_screen == Screen::Download
            && std::env::var_os("BIBIOCR_SKIP_DOWNLOAD").is_none())
        .then(|| {
            download::start(
                model_dir.clone(),
                download_settings.clone(),
                locale,
                settings.ocr_engine,
                settings.tts_engine,
            )
        });
        let history_path = data_dir.join("history.toml");
        let records = history::load(&history_path);
        let (app_sender, app_events) = mpsc::channel();
        let (recognition_sender, recognition_events) = mpsc::channel();
        Self {
            screen: first_screen,
            last_rendered_screen: first_screen,
            search: String::new(),
            recognition_progress: 0.0,
            recognition_text: String::new(),
            recognition_error: None,
            recognition_active: false,
            result_tab: ResultTab::Preview,
            markdown: String::new(),
            data_dir,
            model_dir,
            history_path,
            records,
            current_image: None,
            current_record_id: None,
            original_texture: None,
            original_size: None,
            original_loading: false,
            text_dialog_open: false,
            app_events,
            app_sender,
            recognition_events,
            recognition_sender,
            recognition_generation: 0,
            recognition_cancel: Arc::new(AtomicBool::new(false)),
            notice: None,
            camera_open: false,
            download_task,
            model_progress,
            model_status,
            download_error: None,
            download_settings,
            ocr_engine: settings.ocr_engine,
            download_view_engine: settings.ocr_engine,
            tts_engine: settings.tts_engine,
            download_view_tts_engine: settings.tts_engine,
            locale,
            language_choice,
            tts_active: false,
            tts_sentences: Vec::new(),
            tts_current: None,
            tts_done: 0,
            tts_preparing: false,
            tts_scroll_to_current: false,
            tts_speed: 1.0,
            tts_volume: 0.8,
            tts_stop: Arc::new(AtomicBool::new(false)),
            tts_volume_shared: Arc::new(AtomicU32::new(0.8_f32.to_bits())),
            tts_speed_shared: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            system_insets_px: core::SystemInsetsPx::default(),
            swipe_start: None,
            #[cfg(target_os = "android")]
            android_app: None,
        }
    }

    #[cfg(target_os = "android")]
    pub fn new_android(
        cc: &eframe::CreationContext<'_>,
        android_app: android_activity::AndroidApp,
    ) -> Self {
        let mut app = Self::new(cc);
        app.android_app = Some(android_app);
        app
    }

    fn system_insets(&mut self, ctx: &egui::Context) -> (f32, f32) {
        #[cfg(target_os = "android")]
        {
            let mut fallback = self.system_insets_px;
            if let Some(app) = &self.android_app
                && let Some(window) = app.native_window()
            {
                let content = app.content_rect();
                if window.height() > 0 && content.bottom > content.top {
                    fallback = core::SystemInsetsPx::from_content_bounds(
                        window.height(),
                        content.top,
                        content.bottom,
                    );
                }
            }
            self.system_insets_px = core::SystemInsetsPx::runtime_or_fallback(
                crate::android_bridge::system_insets_px().ok().flatten(),
                fallback,
            );
        }
        let scale = ctx.pixels_per_point().max(0.1);
        (
            self.system_insets_px.top / scale,
            self.system_insets_px.bottom / scale,
        )
    }

    fn poll_app_events(&mut self, ctx: &egui::Context) {
        let events: Vec<_> = self.app_events.try_iter().collect();
        for event in events {
            match event {
                AppEvent::Image(Ok(Some(path))) => {
                    self.camera_open = false;
                    self.current_image = Some(path);
                    self.original_texture = None;
                    self.original_size = None;
                    self.original_loading = false;
                    self.start_recognition();
                }
                AppEvent::OriginalImage(path, result) => {
                    if self.current_image.as_ref() != Some(&path) {
                        continue;
                    }
                    self.original_loading = false;
                    match result {
                        Ok((original_size, size, pixels)) => {
                            self.original_size = Some(original_size);
                            self.original_texture = Some(ctx.load_texture(
                                "ocr-original",
                                egui::ColorImage::from_rgba_unmultiplied(size, &pixels),
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                        Err(error) => self.notice = Some(error),
                    }
                }
                AppEvent::TextInput(target, result) => {
                    self.text_dialog_open = false;
                    match result {
                        Ok(Some(text)) => match target {
                            TextTarget::Search => self.search = text,
                            TextTarget::Markdown => {
                                self.markdown = text;
                                if let Some(id) = self.current_record_id {
                                    history::complete(&mut self.records, id, self.markdown.clone());
                                    if let Err(error) =
                                        history::save(&self.history_path, &self.records)
                                    {
                                        self.notice = Some(error);
                                    }
                                }
                            }
                        },
                        Ok(None) => {}
                        Err(error) => self.notice = Some(error),
                    }
                }
                AppEvent::Image(Ok(None)) => {
                    self.camera_open = false;
                    self.screen = Screen::History;
                }
                AppEvent::Image(Err(error)) => {
                    self.camera_open = false;
                    self.screen = Screen::History;
                    self.notice = Some(error);
                }
                AppEvent::Notice(Err(error)) => {
                    self.notice = Some(error);
                }
                AppEvent::Notice(Ok(message)) => self.notice = Some(message),
                AppEvent::TtsSentence(index) => {
                    if self.tts_active {
                        self.tts_current = Some(index);
                        self.tts_preparing = false;
                        self.tts_scroll_to_current = true;
                    }
                }
                AppEvent::TtsSentenceDone(index) => {
                    if self.tts_active {
                        self.tts_done = index + 1;
                        self.tts_current = None;
                        self.tts_preparing = true;
                    }
                }
                AppEvent::Tts(result) => {
                    self.tts_active = false;
                    self.tts_current = None;
                    self.tts_preparing = false;
                    if let Err(error) = result {
                        self.notice = Some(error);
                    }
                }
            }
        }

        let events: Vec<_> = self.recognition_events.try_iter().collect();
        for (generation, event) in events {
            if !accept_recognition_event(
                self.recognition_active,
                self.recognition_generation,
                generation,
            ) {
                continue;
            }
            match event {
                RecognitionEvent::Layout => self.recognition_progress = 0.35,
                RecognitionEvent::Ocr => {
                    self.recognition_progress = 0.65;
                    if let Some(id) = self.current_record_id {
                        if self.records.iter().any(|record| {
                            record.id == id && record.ocr_engine == OcrEngine::PaddleVl16
                        }) {
                            history::set_stage(&mut self.records, id, ScanStage::LayoutDone);
                            if let Err(error) = history::save(&self.history_path, &self.records) {
                                self.notice = Some(error);
                            }
                        }
                    }
                }
                RecognitionEvent::Token(piece) => {
                    self.recognition_progress = 0.8;
                    self.recognition_text.push_str(&piece);
                }
                RecognitionEvent::Complete(markdown) => {
                    self.recognition_active = false;
                    self.recognition_progress = 1.0;
                    self.markdown = crate::markdown::normalize_ocr(&markdown);
                    self.recognition_text.clear();
                    self.result_tab = ResultTab::Preview;
                    if let Some(id) = self.current_record_id {
                        history::complete(&mut self.records, id, self.markdown.clone());
                        if let Err(error) = history::save(&self.history_path, &self.records) {
                            self.notice = Some(error);
                        }
                    }
                    if self.screen == Screen::Recognizing {
                        self.screen = Screen::Result;
                    }
                }
                RecognitionEvent::Failed(error) => {
                    self.recognition_active = false;
                    self.recognition_error = Some(error);
                }
            }
        }
    }

    fn handle_primary_swipe(&mut self, ctx: &egui::Context, bounds: Rect) {
        let (pressed, released, position) = ctx.input(|input| {
            (
                input.pointer.any_pressed(),
                input.pointer.any_released(),
                input.pointer.interact_pos(),
            )
        });
        if pressed {
            self.swipe_start = position.filter(|position| bounds.contains(*position));
        }
        if released {
            if let (Some(start), Some(end)) = (self.swipe_start.take(), position) {
                let delta = end - start;
                self.screen = swiped_primary_screen(self.screen, delta.x, delta.y);
            }
        }
    }

    fn open_camera(&mut self) {
        if self.recognition_active {
            self.screen = Screen::Recognizing;
            return;
        }
        self.screen = Screen::Recognizing;
        self.recognition_progress = 0.0;
        self.recognition_text.clear();
        self.recognition_error = None;
        self.notice = None;
        self.camera_open = true;
        let sender = self.app_sender.clone();
        #[cfg(target_os = "android")]
        if let Err(error) = crate::android_bridge::capture_photo(move |result| {
            let _ = sender.send(AppEvent::Image(result));
        }) {
            self.camera_open = false;
            self.screen = Screen::History;
            self.notice = Some(error);
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = sender;
            self.open_gallery();
        }
    }

    fn open_gallery(&mut self) {
        if self.recognition_active {
            self.screen = Screen::Recognizing;
            return;
        }
        self.screen = Screen::Recognizing;
        self.recognition_progress = 0.0;
        self.recognition_text.clear();
        self.recognition_error = None;
        self.notice = None;
        self.camera_open = true;
        let sender = self.app_sender.clone();
        let scans = self.data_dir.join("scans");
        let result = robius_file_picker::FileDialog::new()
            .set_title(rust_i18n::t!("mobile_choose_photo").into_owned())
            .pick_image(move |result| {
                let result = result.map_err(|error| error.to_string()).and_then(|file| {
                    file.map(|file| persist_picked_image(&file, &scans))
                        .transpose()
                });
                let _ = sender.send(AppEvent::Image(result));
            });
        if let Err(error) = result {
            self.camera_open = false;
            self.screen = Screen::History;
            self.notice = Some(error.to_string());
        }
    }

    fn start_recognition(&mut self) {
        let Some(image) = self.current_image.clone() else {
            return;
        };
        let existing = self.current_record_id.and_then(|id| {
            self.records
                .iter()
                .find(|record| record.id == id && record.image_path == image)
                .map(|record| (record.stage, record.ocr_engine))
        });
        let stage = existing.map(|(stage, _)| stage);
        let engine = existing.map_or(self.ocr_engine, |(_, engine)| engine);
        if stage.is_none() || stage == Some(ScanStage::Complete) {
            let id = now_id();
            history::add_pending(&mut self.records, id, image.clone(), self.ocr_engine);
            self.current_record_id = Some(id);
            if let Err(error) = history::save(&self.history_path, &self.records) {
                self.recognition_error = Some(error);
                self.screen = Screen::Recognizing;
                return;
            }
        }
        self.screen = Screen::Recognizing;
        self.recognition_progress = 0.1;
        self.recognition_text.clear();
        self.recognition_error = None;
        let missing = core::required_models(engine, self.tts_engine)
            .filter(|(_, model)| {
                !matches!(
                    model.group,
                    core::ModelGroup::Melo | core::ModelGroup::Kokoro
                )
            })
            .any(|(_, model)| !download::installed(&self.model_dir, model));
        if missing {
            self.recognition_active = false;
            self.screen = Screen::Download;
            if self.download_task.is_none() {
                self.download_task = Some(download::start(
                    self.model_dir.clone(),
                    self.download_settings.clone(),
                    self.locale,
                    engine,
                    self.tts_engine,
                ));
            }
            return;
        }
        self.recognition_active = true;
        self.recognition_cancel.store(true, Ordering::Relaxed);
        self.recognition_cancel = Arc::new(AtomicBool::new(false));
        self.recognition_generation = self.recognition_generation.wrapping_add(1);
        let generation = self.recognition_generation;
        let cancel = Arc::clone(&self.recognition_cancel);
        let model_dir = self.model_dir.clone();
        let sender = self.recognition_sender.clone();
        #[cfg(target_os = "android")]
        std::thread::spawn(move || {
            let _wake_lock = crate::android_bridge::OcrWakeLock::acquire().ok();
            let emit = |event| sender.send((generation, event));
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            if engine == OcrEngine::PaddleV6 {
                let _ = emit(RecognitionEvent::Layout);
                let _ = emit(RecognitionEvent::Ocr);
                match crate::v6::recognize_cancellable(&model_dir, &image, &cancel) {
                    Ok(markdown) if !markdown.trim().is_empty() => {
                        let _ = emit(RecognitionEvent::Complete(markdown));
                    }
                    Ok(_) => {
                        let _ = emit(RecognitionEvent::Failed("OCR returned no text".to_owned()));
                    }
                    Err(error) => {
                        let _ = emit(RecognitionEvent::Failed(error));
                    }
                }
                return;
            }
            if stage != Some(ScanStage::LayoutDone) {
                let _ = emit(RecognitionEvent::Layout);
                if let Err(error) =
                    crate::layout::analyze(&model_dir.join(MODELS[2].file_name), &image)
                {
                    let _ = emit(RecognitionEvent::Failed(error));
                    return;
                }
            }
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let _ = emit(RecognitionEvent::Ocr);
            let stream = sender.clone();
            match crate::recognizer::recognize_stream_cancellable(
                &model_dir.join(MODELS[0].file_name),
                &model_dir.join(MODELS[1].file_name),
                &image,
                &cancel,
                move |piece| {
                    let _ = stream.send((generation, RecognitionEvent::Token(piece.to_owned())));
                },
            ) {
                Ok(markdown) if !markdown.is_empty() => {
                    let _ = emit(RecognitionEvent::Complete(markdown));
                }
                Ok(_) => {
                    let _ = emit(RecognitionEvent::Failed("OCR returned no text".to_owned()));
                }
                Err(error) => {
                    let _ = emit(RecognitionEvent::Failed(error));
                }
            }
        });
        #[cfg(not(target_os = "android"))]
        {
            let _ = (image, model_dir, stage, engine, cancel);
            let _ = sender.send((
                generation,
                RecognitionEvent::Failed(
                    "On-device OCR is available in the Android build".to_owned(),
                ),
            ));
        }
    }

    fn cancel_recognition(&mut self) {
        self.recognition_cancel.store(true, Ordering::Relaxed);
        self.recognition_active = false;
        self.recognition_text.clear();
        self.screen = Screen::History;
    }

    fn open_record(&mut self, record: ScanRecord) {
        if self.recognition_active {
            self.screen = Screen::Recognizing;
            return;
        }
        self.tts_stop.store(true, Ordering::Relaxed);
        self.tts_active = false;
        self.tts_current = None;
        self.tts_preparing = false;
        if record.stage != ScanStage::Complete && record.ocr_engine != self.ocr_engine {
            self.set_ocr_engine(record.ocr_engine);
            if record.ocr_engine != self.ocr_engine {
                return;
            }
        }
        self.current_record_id = Some(record.id);
        self.current_image = Some(record.image_path);
        self.original_texture = None;
        self.original_size = None;
        self.original_loading = false;
        self.markdown = crate::markdown::normalize_ocr(&record.markdown);
        if self.markdown != record.markdown {
            history::complete(&mut self.records, record.id, self.markdown.clone());
            if let Err(error) = history::save(&self.history_path, &self.records) {
                self.notice = Some(error);
            }
        }
        self.result_tab = ResultTab::Preview;
        if record.stage == ScanStage::Complete {
            self.screen = Screen::Result;
        } else {
            self.start_recognition();
        }
    }

    #[cfg(target_os = "android")]
    fn open_text_dialog(&mut self, ctx: &egui::Context, target: TextTarget) {
        if self.text_dialog_open {
            return;
        }
        let (title, initial, multiline) = match target {
            TextTarget::Search => (
                rust_i18n::t!("mobile_search_hint").into_owned(),
                self.search.as_str(),
                false,
            ),
            TextTarget::Markdown => (
                rust_i18n::t!("mobile_tab_edit").into_owned(),
                self.markdown.as_str(),
                true,
            ),
        };
        let sender = self.app_sender.clone();
        let ctx = ctx.clone();
        match crate::android_bridge::prompt_text(&title, initial, multiline, move |result| {
            let _ = sender.send(AppEvent::TextInput(target, result));
            ctx.request_repaint();
        }) {
            Ok(()) => self.text_dialog_open = true,
            Err(error) => self.notice = Some(error),
        }
    }

    fn start_tts(&mut self) {
        if self.tts_active {
            return;
        }
        self.notice = None;
        self.tts_sentences = crate::tts::sentences(&self.markdown);
        if self.tts_sentences.is_empty() {
            self.notice = Some("There is no text to read".to_owned());
            return;
        }
        self.tts_active = true;
        self.tts_done = 0;
        self.tts_current = None;
        self.tts_preparing = true;
        self.tts_stop = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&self.tts_stop);
        let volume = Arc::clone(&self.tts_volume_shared);
        let playback_speed = Arc::clone(&self.tts_speed_shared);
        let sender = self.app_sender.clone();
        let sentences = self.tts_sentences.clone();
        let model_dir = self.model_dir.clone();
        let tts_engine = self.tts_engine;
        let document_dir = self.current_record_id.and_then(|id| {
            self.records
                .iter()
                .find(|record| record.id == id)
                .map(|record| {
                    let directory = self
                        .data_dir
                        .join("tts")
                        .join(id.to_string())
                        .join(record.tts_revision.to_string());
                    if tts_engine == TtsEngine::Melo {
                        directory
                    } else {
                        directory.join(tts_engine.cache_name())
                    }
                })
        });
        std::thread::spawn(move || {
            let result = (|| {
                let mut generated = 0;
                let mut skipped = 0;
                for (index, sentence) in sentences.iter().enumerate() {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let result = if let Some(document_dir) = &document_dir {
                        crate::tts::synthesize_resilient_cached(
                            sentence,
                            &model_dir,
                            document_dir,
                            &stop,
                            tts_engine,
                        )
                    } else {
                        crate::tts::synthesize_resilient(sentence, &model_dir, &stop, tts_engine)
                    };
                    let audio = match result {
                        Ok(audio) => audio,
                        Err(_) if stop.load(Ordering::Relaxed) => break,
                        Err(_) => {
                            skipped += 1;
                            let _ = sender.send(AppEvent::TtsSentenceDone(index));
                            continue;
                        }
                    };
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let _ = sender.send(AppEvent::TtsSentence(index));
                    #[cfg_attr(not(target_os = "android"), allow(unused_mut))]
                    let mut playback_failed = false;
                    for piece in audio {
                        #[cfg(target_os = "android")]
                        if let Err(error) = crate::android_bridge::play_pcm(
                            &piece.samples,
                            piece.sample_rate,
                            &volume,
                            &playback_speed,
                            &stop,
                        ) {
                            eprintln!("TTS segment {index} playback failed: {error}");
                            playback_failed = true;
                            break;
                        }
                        #[cfg(not(target_os = "android"))]
                        let _ = (&piece, &volume, &playback_speed);
                    }
                    if playback_failed {
                        skipped += 1;
                    } else {
                        generated += 1;
                    }
                    let _ = sender.send(AppEvent::TtsSentenceDone(index));
                }
                if skipped > 0 {
                    let _ = sender.send(AppEvent::Notice(Ok(format!(
                        "Skipped {skipped} unreadable TTS segment(s)"
                    ))));
                }
                if generated == 0 && skipped > 0 {
                    Err(format!(
                        "{} TTS could not read any segment",
                        tts_engine.cache_name()
                    ))
                } else {
                    Ok(())
                }
            })();
            if !stop.load(Ordering::Relaxed) {
                let _ = sender.send(AppEvent::Tts(result));
            }
        });
    }

    fn regenerate_tts(&mut self) {
        if self.tts_active {
            return;
        }
        if !MODELS
            .iter()
            .filter(|model| {
                matches!(
                    (self.tts_engine, model.group),
                    (TtsEngine::Melo, core::ModelGroup::Melo)
                        | (TtsEngine::Kokoro, core::ModelGroup::Kokoro)
                )
            })
            .all(|model| download::installed(&self.model_dir, model))
        {
            self.notice = Some(format!(
                "Download the {} TTS files first",
                self.tts_engine.cache_name()
            ));
            return;
        }
        let Some(index) = self
            .current_record_id
            .and_then(|id| self.records.iter().position(|record| record.id == id))
        else {
            self.notice = Some("No saved scan record".to_owned());
            return;
        };
        let previous = self.records[index].tts_revision;
        let Some(next) = previous.checked_add(1) else {
            self.notice = Some("TTS revision limit reached".to_owned());
            return;
        };
        // ponytail: Retain old revisions so a stopping worker cannot overwrite new audio.
        self.records[index].tts_revision = next;
        if let Err(error) = history::save(&self.history_path, &self.records) {
            self.records[index].tts_revision = previous;
            self.notice = Some(error);
            return;
        }
        self.start_tts();
    }

    fn save_markdown(&mut self) {
        self.save_bytes(
            "bibiocr.md",
            "text/markdown",
            self.markdown.as_bytes().to_vec(),
        );
    }

    fn save_docx(&mut self) {
        match crate::export::docx_bytes(&self.markdown) {
            Ok(bytes) => self.save_bytes(
                "bibiocr.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                bytes,
            ),
            Err(error) => self.notice = Some(error),
        }
    }

    fn save_bytes(&mut self, name: &str, mime: &str, bytes: Vec<u8>) {
        let sender = self.app_sender.clone();
        let result = robius_file_picker::FileDialog::new()
            .set_file_name(name)
            .set_mime_type(mime)
            .save_data(bytes, move |result| {
                let message = result.map_err(|error| error.to_string()).map(|file| {
                    file.map_or_else(
                        || "Canceled".to_owned(),
                        |file| format!("Saved: {}", file.file_name().unwrap_or("file")),
                    )
                });
                let _ = sender.send(AppEvent::Notice(message));
            });
        if let Err(error) = result {
            self.notice = Some(error.to_string());
        }
    }

    fn share_markdown(&mut self) {
        #[cfg(target_os = "android")]
        if let Err(error) = crate::android_bridge::share_text(&self.markdown) {
            self.notice = Some(error);
        }
        #[cfg(not(target_os = "android"))]
        self.save_markdown();
    }

    fn share_docx(&mut self) {
        let bytes = match crate::export::docx_bytes(&self.markdown) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.notice = Some(error);
                return;
            }
        };
        #[cfg(target_os = "android")]
        {
            let path = self.data_dir.join("share.docx");
            if let Err(error) = fs::write(&path, bytes) {
                self.notice = Some(error.to_string());
                return;
            }
            let sender = self.app_sender.clone();
            let result = robius_file_picker::FileDialog::new()
                .set_file_name("bibiocr.docx")
                .set_mime_type(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                .save_to_downloads(path, move |result| {
                    let result = result
                        .map_err(|error| error.to_string())
                        .and_then(|file| {
                            let uri = file
                                .and_then(|file| file.into_uri())
                                .ok_or_else(|| "No share URI returned".to_owned())?;
                            crate::android_bridge::share_uri(
                                &uri,
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            )?;
                            Ok("DOCX ready to share".to_owned())
                        });
                    let _ = sender.send(AppEvent::Notice(result));
                });
            if let Err(error) = result {
                self.notice = Some(error.to_string());
            }
        }
        #[cfg(not(target_os = "android"))]
        self.save_bytes(
            "bibiocr.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            bytes,
        );
    }

    fn poll_downloads(&mut self) {
        let mut events = Vec::new();
        if let Some(task) = &self.download_task {
            events.extend(task.events.try_iter());
        }
        for event in events {
            match event {
                DownloadEvent::Started(index) => {
                    self.model_status[index] = ModelStatus::Downloading
                }
                DownloadEvent::Progress(index, downloaded, total) => {
                    self.model_progress[index] = (downloaded, total);
                }
                DownloadEvent::Complete(index) => {
                    self.model_status[index] = ModelStatus::Complete;
                    self.model_progress[index] =
                        (MODELS[index].expected_bytes, MODELS[index].expected_bytes);
                }
                DownloadEvent::Failed(index, error) => {
                    self.model_status[index] = ModelStatus::Failed;
                    self.download_error = Some(error);
                    self.download_task = None;
                }
                DownloadEvent::Finished => {
                    self.download_task = None;
                }
            }
        }
    }

    fn download_screen(&mut self, ui: &mut egui::Ui, entered: bool) {
        self.poll_downloads();
        if entered {
            self.download_view_engine = self.ocr_engine;
            self.download_view_tts_engine = self.tts_engine;
        }
        let mut required =
            core::required_models(self.download_view_engine, self.download_view_tts_engine)
                .collect::<Vec<_>>();
        required.sort_by_key(|(_, model)| {
            matches!(
                model.group,
                core::ModelGroup::Melo | core::ModelGroup::Kokoro
            )
        });
        let missing = core::required_models(self.ocr_engine, self.tts_engine)
            .map(|(_, model)| model)
            .any(|model| !download::installed(&self.model_dir, model));
        let check_wifi = entered && self.download_task.is_none() && missing;
        #[cfg(target_os = "android")]
        let on_wifi = check_wifi
            && (!self.download_settings.wifi_only
                || crate::android_bridge::is_wifi_connected().unwrap_or(false));
        #[cfg(not(target_os = "android"))]
        let on_wifi = check_wifi;
        if should_auto_resume_download(entered, self.download_task.is_some(), missing, on_wifi)
            && std::env::var_os("BIBIOCR_SKIP_DOWNLOAD").is_none()
        {
            self.download_error = None;
            self.download_task = Some(download::start(
                self.model_dir.clone(),
                self.download_settings.clone(),
                self.locale,
                self.ocr_engine,
                self.tts_engine,
            ));
        }
        let title = rust_i18n::t!("mobile_download_title").into_owned();
        let (back, settings) = top_bar(ui, &title, true, Some(Icon::Gear), INK, CHIP);
        if back {
            self.screen = Screen::History;
            return;
        }
        if settings {
            self.screen = Screen::Settings;
            return;
        }
        if let Some(engine) = engine_picker(ui, self.download_view_engine, "mobile_download_view") {
            self.download_view_engine = engine;
            ui.ctx().request_repaint();
            return;
        }
        let content_height = (ui.available_height() - 74.0).max(0.0);
        let mut next_tts_view = None;
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), content_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                download_scroll_area().show(ui, |ui| {
                    ui.add_space(4.0);

                    let downloaded: u64 = required
                        .iter()
                        .map(|(index, _)| self.model_progress[*index].0)
                        .sum();
                    let total: u64 = required.iter().map(|(_, model)| model.expected_bytes).sum();
                    let overall = downloaded as f32 / total as f32;
                    let mut retry = false;
                    let mut pause_clicked = false;
                    egui::Frame::default()
                        .fill(CARD)
                        .corner_radius(CornerRadius::same(14))
                        .inner_margin(Margin::same(9))
                        .outer_margin(Margin {
                            left: 16,
                            right: 16,
                            top: 0,
                            bottom: 4,
                        })
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(
                                        rust_i18n::t!("mobile_download_total").into_owned(),
                                    )
                                    .size(14.0)
                                    .color(MUTED),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        pause_clicked = ui
                                            .add_enabled(
                                                self.download_task.is_some(),
                                                egui::Button::new(
                                                    if self
                                                        .download_task
                                                        .as_ref()
                                                        .is_some_and(DownloadTask::is_paused)
                                                    {
                                                        "▶"
                                                    } else {
                                                        "Ⅱ"
                                                    },
                                                ),
                                            )
                                            .clicked();
                                        ui.label(
                                            egui::RichText::new(format!("{:.0}%", overall * 100.0))
                                                .size(14.0)
                                                .strong()
                                                .color(INK),
                                        );
                                    },
                                );
                            });
                            ui.add_space(6.0);
                            let (rect, _) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 8.0),
                                Sense::hover(),
                            );
                            ui.painter().rect_filled(rect, CornerRadius::same(4), LINE);
                            ui.painter().rect_filled(
                                Rect::from_min_size(
                                    rect.min,
                                    Vec2::new(rect.width() * overall, rect.height()),
                                ),
                                CornerRadius::same(4),
                                GREEN,
                            );
                            ui.add_space(5.0);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} / {}",
                                    format_bytes(downloaded),
                                    format_bytes(total)
                                ))
                                .size(13.0)
                                .color(MUTED),
                            );
                            if let Some(error) = &self.download_error {
                                ui.colored_label(ORANGE, error);
                                retry = ui
                                    .button(rust_i18n::t!("mobile_retry_download").into_owned())
                                    .clicked();
                            }
                        });
                    if pause_clicked {
                        if let Some(task) = &self.download_task {
                            task.toggle_pause();
                        }
                    }
                    if retry {
                        self.download_error = None;
                        self.download_task = Some(download::start(
                            self.model_dir.clone(),
                            self.download_settings.clone(),
                            self.locale,
                            self.ocr_engine,
                            self.tts_engine,
                        ));
                    }

                    let mut showing_tts = false;
                    for &(index, model) in &required {
                        if !showing_tts
                            && matches!(
                                model.group,
                                core::ModelGroup::Melo | core::ModelGroup::Kokoro
                            )
                        {
                            showing_tts = true;
                            next_tts_view = tts_picker(ui, self.download_view_tts_engine);
                            ui.add_space(8.0);
                        }
                        let (downloaded, total) = self.model_progress[index];
                        let fraction = if total == 0 {
                            0.0
                        } else {
                            downloaded as f32 / total as f32
                        };
                        egui::Frame::default()
                            .fill(CARD)
                            .corner_radius(CornerRadius::same(14))
                            .inner_margin(Margin::same(8))
                            .outer_margin(Margin {
                                left: 16,
                                right: 16,
                                top: 0,
                                bottom: 4,
                            })
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.add_sized(
                                        [ui.available_width() - 68.0, 18.0],
                                        egui::Label::new(
                                            egui::RichText::new(model.file_name)
                                                .size(11.0)
                                                .strong()
                                                .color(INK),
                                        )
                                        .truncate(),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let (key, color) = match self.model_status[index] {
                                                ModelStatus::Complete => {
                                                    ("mobile_status_complete", GREEN)
                                                }
                                                ModelStatus::Downloading => {
                                                    ("mobile_status_downloading", ORANGE)
                                                }
                                                ModelStatus::Failed => {
                                                    ("mobile_status_retry", ORANGE)
                                                }
                                                ModelStatus::Waiting => {
                                                    ("mobile_status_waiting", MUTED)
                                                }
                                            };
                                            ui.label(
                                                egui::RichText::new(
                                                    rust_i18n::t!(key).into_owned(),
                                                )
                                                .size(11.0)
                                                .color(color),
                                            );
                                        },
                                    );
                                });
                                ui.add_space(2.0);
                                progress_bar(ui, ui.available_width(), fraction);
                                ui.add_space(2.0);
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} / {}",
                                            format_bytes(downloaded),
                                            format_bytes(model.expected_bytes)
                                        ))
                                        .size(11.0)
                                        .color(MUTED),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "{:.0}%",
                                                    fraction * 100.0
                                                ))
                                                .size(11.0)
                                                .strong()
                                                .color(INK),
                                            );
                                        },
                                    );
                                });
                            });
                    }
                    ui.add_space(12.0);
                });
            },
        );
        if let Some(engine) = next_tts_view {
            self.download_view_tts_engine = engine;
            ui.ctx().request_repaint();
        }

        let panel = ui.clip_rect();
        let bar = Rect::from_min_max(
            Pos2::new(panel.left(), panel.bottom() - 74.0),
            panel.right_bottom(),
        );
        ui.painter().rect_filled(bar, CornerRadius::same(0), CARD);
        let width = (bar.width() - 44.0) / 2.0;
        let left = Rect::from_min_size(
            Pos2::new(bar.left() + 16.0, bar.top() + 12.0),
            Vec2::new(width, 50.0),
        );
        let right = Rect::from_min_size(
            Pos2::new(left.right() + 12.0, left.top()),
            Vec2::new(width, 50.0),
        );
        let wifi_response = ui.interact(left, Id::new("download_wifi_only"), Sense::click());
        if self.download_settings.wifi_only {
            ui.painter()
                .rect_filled(left, CornerRadius::same(14), ORANGE_SOFT);
        }
        ui.painter().rect_stroke(
            left,
            CornerRadius::same(14),
            Stroke::new(1.0, LINE),
            StrokeKind::Middle,
        );
        ui.painter().text(
            left.center(),
            Align2::CENTER_CENTER,
            format!(
                "{}{}",
                rust_i18n::t!("mobile_wifi_only"),
                if self.download_settings.wifi_only {
                    " ✓"
                } else {
                    ""
                }
            ),
            FontId::proportional(15.0),
            INK,
        );
        let response = ui.interact(right, Id::new("download_background"), Sense::click());
        ui.painter()
            .rect_filled(right, CornerRadius::same(14), INDIGO);
        ui.painter().text(
            right.center(),
            Align2::CENTER_CENTER,
            rust_i18n::t!("mobile_background").into_owned(),
            FontId::proportional(15.0),
            WHITE,
        );
        if response.clicked() {
            self.screen = Screen::History;
        }
        if wifi_response.clicked() {
            self.set_wifi_only(!self.download_settings.wifi_only);
        }
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(100));
    }

    // ---------------------------------------------------------------- settings

    fn set_language(&mut self, language: &str) {
        let path = self.data_dir.join("settings.toml");
        let mut settings = Settings::load_or_default(&path);
        settings.locale.language = language.to_owned();
        if let Err(error) = settings.save(&path) {
            self.notice = Some(error);
            return;
        }
        self.language_choice = language.to_owned();
        self.locale = LocaleManager::apply(language);
        self.notice = None;
    }

    fn set_wifi_only(&mut self, enabled: bool) {
        let path = self.data_dir.join("settings.toml");
        let mut settings = Settings::load_or_default(&path);
        settings.download.wifi_only = enabled;
        if let Err(error) = settings.save(&path) {
            self.notice = Some(error);
            return;
        }
        self.download_settings.wifi_only = enabled;
        if let Some(task) = self.download_task.take() {
            task.cancel();
        }
        if core::required_models(self.ocr_engine, self.tts_engine)
            .any(|(_, model)| !download::installed(&self.model_dir, model))
        {
            self.download_task = Some(download::start(
                self.model_dir.clone(),
                self.download_settings.clone(),
                self.locale,
                self.ocr_engine,
                self.tts_engine,
            ));
        }
    }

    fn set_ocr_engine(&mut self, engine: OcrEngine) {
        if self.ocr_engine == engine {
            return;
        }
        let path = self.data_dir.join("settings.toml");
        let mut settings = Settings::load_or_default(&path);
        settings.ocr_engine = engine;
        if let Err(error) = settings.save(&path) {
            self.notice = Some(error);
            return;
        }
        if let Some(task) = self.download_task.take() {
            task.cancel();
        }
        self.ocr_engine = engine;
        self.download_view_engine = engine;
        self.download_error = None;
        for (index, model) in MODELS.iter().enumerate() {
            let complete = download::installed(&self.model_dir, model);
            self.model_status[index] = if complete {
                ModelStatus::Complete
            } else {
                ModelStatus::Waiting
            };
            self.model_progress[index] = (
                if complete { model.expected_bytes } else { 0 },
                model.expected_bytes,
            );
        }
        if core::initial_screen(&self.model_dir, engine, self.tts_engine) == core::Screen::Download
        {
            self.screen = Screen::Download;
            self.download_task = Some(download::start(
                self.model_dir.clone(),
                self.download_settings.clone(),
                self.locale,
                engine,
                self.tts_engine,
            ));
        } else if self.screen == Screen::Download {
            self.screen = Screen::History;
        }
    }

    fn set_tts_engine(&mut self, engine: TtsEngine) {
        if self.tts_engine == engine {
            return;
        }
        let path = self.data_dir.join("settings.toml");
        let mut settings = Settings::load_or_default(&path);
        settings.tts_engine = engine;
        if let Err(error) = settings.save(&path) {
            self.notice = Some(error);
            return;
        }
        if let Some(task) = self.download_task.take() {
            task.cancel();
        }
        self.tts_stop.store(true, Ordering::Relaxed);
        self.tts_active = false;
        self.tts_engine = engine;
        self.download_view_tts_engine = engine;
        self.download_error = None;
        if core::initial_screen(&self.model_dir, self.ocr_engine, engine) == core::Screen::Download
        {
            self.screen = Screen::Download;
            self.download_task = Some(download::start(
                self.model_dir.clone(),
                self.download_settings.clone(),
                self.locale,
                self.ocr_engine,
                engine,
            ));
        }
    }

    fn settings_screen(&mut self, ui: &mut egui::Ui) {
        let (back, _) = top_bar(
            ui,
            &rust_i18n::t!("mobile_settings_title").into_owned(),
            true,
            None,
            INK,
            CHIP,
        );
        if back {
            self.screen = Screen::History;
            return;
        }
        ui.add_space(12.0);
        if let Some(engine) = engine_picker(ui, self.ocr_engine, "mobile_ocr_engine") {
            self.set_ocr_engine(engine);
            ui.ctx().request_repaint();
            return;
        }
        ui.add_space(12.0);
        if let Some(engine) = tts_picker(ui, self.tts_engine) {
            self.set_tts_engine(engine);
            ui.ctx().request_repaint();
            return;
        }
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.label(
                egui::RichText::new(rust_i18n::t!("mobile_language").into_owned())
                    .size(17.0)
                    .strong()
                    .color(INK),
            );
        });
        ui.add_space(10.0);

        let mut selected = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for &(language, native_name) in LocaleManager::choices() {
                let label = if language == "system" {
                    rust_i18n::t!("mobile_follow_system").into_owned()
                } else {
                    native_name.to_owned()
                };
                let width = (ui.available_width() - 32.0).max(160.0);
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    if ui
                        .add_sized(
                            [width, 44.0],
                            egui::Button::new(label).selected(self.language_choice == language),
                        )
                        .clicked()
                    {
                        selected = Some(language);
                    }
                });
                ui.add_space(5.0);
            }
            if let Some(notice) = &self.notice {
                ui.colored_label(ORANGE, notice);
            }
        });
        if let Some(language) = selected {
            self.set_language(language);
            ui.ctx().request_repaint();
        }
    }

    // ---------------------------------------------------------------- history

    fn history_screen(&mut self, ui: &mut egui::Ui) {
        let (_, gear) = top_bar(
            ui,
            &rust_i18n::t!("mobile_history_title").into_owned(),
            false,
            Some(Icon::Gear),
            INK,
            CHIP,
        );
        if gear {
            self.screen = Screen::Settings;
            return;
        }

        ui.add_space(4.0);
        egui::Frame::default()
            .fill(CARD)
            .corner_radius(CornerRadius::same(14))
            .inner_margin(Margin::same(0))
            .outer_margin(Margin {
                left: 16,
                right: 16,
                top: 2,
                bottom: 8,
            })
            .show(ui, |ui| {
                ui.set_min_size(Vec2::new(ui.available_width(), 46.0));
                ui.horizontal(|ui| {
                    ui.add_space(14.0);
                    paint_icon_in(ui, 18.0, Icon::Search, MUTED);
                    ui.add_space(8.0);
                    let search = ui.add(
                        egui::TextEdit::singleline(&mut self.search)
                            .hint_text(rust_i18n::t!("mobile_search_hint").into_owned())
                            .desired_width(ui.available_width() - 8.0),
                    );
                    if search.clicked() {
                        search.request_focus();
                    }
                    ui.add_space(10.0);
                });
            });

        let records: Vec<_> = self
            .records
            .iter()
            .filter(|record| history::matches_query(record, &self.search))
            .cloned()
            .collect();
        let mut open = None;
        let mut delete = None;
        let list_height = (ui.clip_rect().bottom() - ui.cursor().top() - 54.0).max(0.0);
        egui::ScrollArea::vertical()
            .max_height(list_height)
            .show(ui, |ui| {
                if records.is_empty() {
                    ui.add_space(48.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new(rust_i18n::t!("mobile_history_empty").into_owned())
                                .size(15.0)
                                .color(MUTED),
                        );
                    });
                }
                for record in &records {
                    let response = egui::Frame::default()
                        .fill(CARD)
                        .corner_radius(CornerRadius::same(16))
                        .inner_margin(Margin::same(8))
                        .outer_margin(Margin {
                            left: 16,
                            right: 16,
                            top: 0,
                            bottom: 5,
                        })
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.set_min_width((ui.available_width() - 34.0).max(120.0));
                                    ui.add_space(1.0);
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(&record.title)
                                                .size(14.0)
                                                .strong()
                                                .color(INK),
                                        )
                                        .truncate(),
                                    );
                                    if record.stage == ScanStage::Complete {
                                        let file = record
                                            .image_path
                                            .file_name()
                                            .and_then(|name| name.to_str())
                                            .unwrap_or("image");
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(format!(
                                                    "{} · {}",
                                                    record.ocr_engine.label(),
                                                    file
                                                ))
                                                .size(11.0)
                                                .color(MUTED),
                                            )
                                            .truncate(),
                                        );
                                    } else {
                                        let step = if record.stage == ScanStage::LayoutDone {
                                            "mobile_step_ocr"
                                        } else {
                                            "mobile_step_layout"
                                        };
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(format!(
                                                    "{} · {} · {}",
                                                    record.ocr_engine.label(),
                                                    rust_i18n::t!("mobile_resume_scan"),
                                                    rust_i18n::t!(step)
                                                ))
                                                .size(11.0)
                                                .color(ORANGE),
                                            )
                                            .truncate(),
                                        );
                                    }
                                });
                                let (trash, response) =
                                    ui.allocate_exact_size(Vec2::splat(30.0), Sense::click());
                                paint_icon(ui.painter(), trash.shrink(7.0), Icon::Trash, MUTED);
                                if response.clicked() {
                                    delete = Some(record.id);
                                }
                            });
                        })
                        .response;
                    if history_card_open_response(ui, &response, record.id).clicked() {
                        open = Some(record.clone());
                    }
                }
                ui.add_space(60.0);
            });
        if let Some(id) = delete {
            if history::remove(&mut self.records, id)
                && let Err(error) = history::save(&self.history_path, &self.records)
            {
                self.notice = Some(error);
            }
        } else if let Some(record) = open {
            self.open_record(record);
            return;
        }

        // Keep actions above the scroll layer so visible buttons receive taps.
        let panel = ui.clip_rect();
        let bar = Rect::from_min_max(
            Pos2::new(panel.left(), panel.bottom() - 54.0),
            panel.right_bottom(),
        );
        let mut action = None;
        egui::Area::new(Id::new("history_actions"))
            .order(egui::Order::Foreground)
            .fixed_pos(bar.min)
            .show(ui.ctx(), |ui| {
                ui.set_min_size(bar.size());
                ui.painter().rect_filled(bar, CornerRadius::ZERO, CARD);
                let width = ((bar.width() - 32.0) / 3.0).max(60.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    ui.add_space(8.0);
                    for (index, icon, label, color) in [
                        (0, Icon::Download, rust_i18n::t!("mobile_models"), INDIGO),
                        (1, Icon::Camera, rust_i18n::t!("mobile_new_scan"), ORANGE),
                        (
                            2,
                            Icon::Gallery,
                            rust_i18n::t!("mobile_gallery_short"),
                            INDIGO,
                        ),
                    ] {
                        let (rect, response) =
                            ui.allocate_exact_size(Vec2::new(width, 38.0), Sense::click());
                        if index != 0 {
                            ui.painter()
                                .rect_filled(rect, CornerRadius::same(14), color);
                        }
                        paint_action(
                            ui.painter(),
                            rect,
                            icon,
                            if index == 0 { INDIGO } else { WHITE },
                            &label,
                        );
                        if response.clicked() {
                            action = Some(index);
                        }
                    }
                });
            });
        if action == Some(0) {
            self.screen = Screen::Download;
        } else if action == Some(1) {
            self.open_camera();
        } else if action == Some(2) {
            self.open_gallery();
        }
    }

    // ------------------------------------------------------------ recognizing

    fn recognizing_screen(&mut self, ui: &mut egui::Ui) {
        let (back, close) = top_bar(
            ui,
            &rust_i18n::t!("mobile_recognizing").into_owned(),
            true,
            Some(Icon::Close),
            INK,
            CHIP,
        );
        if back || close {
            self.cancel_recognition();
            return;
        }

        self.request_original_texture(ui.ctx());

        ui.add_space(6.0);
        let text_width = (ui.available_width() - 60.0).max(120.0);
        egui::Frame::default()
            .fill(CARD)
            .corner_radius(CornerRadius::same(18))
            .inner_margin(Margin::same(14))
            .outer_margin(Margin {
                left: 16,
                right: 16,
                top: 0,
                bottom: 0,
            })
            .show(ui, |ui| {
                ui.set_min_width(text_width);
                let height = ui.available_height().min(340.0).max(220.0);
                egui::ScrollArea::vertical()
                    .max_height(height)
                    .show(ui, |ui| {
                        ui.set_min_height(height);
                        if let Some(texture) = &self.original_texture {
                            let available = Vec2::new(text_width, height - 8.0);
                            let scale = (available.x / texture.size_vec2().x)
                                .min(available.y / texture.size_vec2().y)
                                .min(1.0);
                            ui.add(egui::Image::new((
                                texture.id(),
                                texture.size_vec2() * scale,
                            )));
                        } else if self.original_loading || self.camera_open {
                            ui.spinner();
                        } else if self.recognition_text.is_empty() {
                            let status = self.recognition_error.clone().unwrap_or_else(|| {
                                if self.camera_open {
                                    rust_i18n::t!("mobile_recognizing").into_owned()
                                } else if self.recognition_progress < 0.65 {
                                    rust_i18n::t!("mobile_step_layout").into_owned()
                                } else {
                                    rust_i18n::t!("mobile_step_ocr").into_owned()
                                }
                            });
                            ui.label(egui::RichText::new(status).size(15.0).color(
                                if self.recognition_error.is_some() {
                                    ORANGE
                                } else {
                                    MUTED
                                },
                            ));
                        } else {
                            ui.label(
                                egui::RichText::new(&self.recognition_text)
                                    .size(14.0)
                                    .color(INK),
                            );
                        }
                    });
            });

        if let Some(error) = &self.recognition_error {
            ui.colored_label(ORANGE, error);
        }

        ui.add_space(20.0);
        let mut cancel = false;
        ui.horizontal(|ui| {
            ui.add_space(20.0);
            ui.label(
                egui::RichText::new(rust_i18n::t!("mobile_overall_progress").into_owned())
                    .size(14.0)
                    .color(INK),
            );
            if self.recognition_active
                && ui
                    .button(rust_i18n::t!("mobile_cancel_ocr").into_owned())
                    .clicked()
            {
                cancel = true;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(20.0);
                ui.label(
                    egui::RichText::new(format!("{}%", (self.recognition_progress * 100.0) as u32))
                        .size(14.0)
                        .strong()
                        .color(INK),
                );
            });
        });
        if cancel {
            self.cancel_recognition();
            return;
        }
        ui.horizontal(|ui| {
            ui.add_space(20.0);
            progress_bar(ui, ui.available_width() - 20.0, self.recognition_progress);
        });

        ui.add_space(18.0);
        step_row(
            ui,
            if self.camera_open {
                StepState::Pending
            } else if self.recognition_progress >= 0.65 {
                StepState::Done
            } else if self.recognition_active {
                StepState::Active
            } else {
                StepState::Pending
            },
            1,
            rust_i18n::t!("mobile_step_layout").into_owned(),
        );
        step_row(
            ui,
            if self.recognition_progress >= 0.65 && self.recognition_active {
                StepState::Active
            } else {
                StepState::Pending
            },
            2,
            rust_i18n::t!("mobile_step_ocr").into_owned(),
        );
    }

    // ----------------------------------------------------------------- result

    fn request_original_texture(&mut self, ctx: &egui::Context) {
        if self.original_texture.is_some() || self.original_loading {
            return;
        }
        let Some(path) = self.current_image.clone() else {
            return;
        };
        self.original_loading = true;
        let sender = self.app_sender.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = image::open(&path)
                .map_err(|error| error.to_string())
                .map(|image| {
                    let original_size = [image.width() as usize, image.height() as usize];
                    let image = image.thumbnail(1600, 1600).to_rgba8();
                    (
                        original_size,
                        [image.width() as usize, image.height() as usize],
                        image.into_raw(),
                    )
                });
            let _ = sender.send(AppEvent::OriginalImage(path, result));
            ctx.request_repaint();
        });
    }

    fn result_screen(&mut self, ui: &mut egui::Ui) {
        let (back, dots) = top_bar(
            ui,
            &rust_i18n::t!("mobile_result_title").into_owned(),
            true,
            Some(Icon::Dots),
            INK,
            CHIP,
        );
        if back {
            self.tts_stop.store(true, Ordering::Relaxed);
            self.tts_active = false;
            self.tts_current = None;
            self.tts_preparing = false;
            self.screen = Screen::History;
            return;
        }
        if dots {
            self.share_markdown();
        }

        // The markdown viewer can report a content width wider than the
        // screen, so pin every block below to the panel width measured up
        // front.
        let content_width = ui.available_width();

        // Tab pills: Preview / Edit / Original.
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            for (tab, key) in [
                (ResultTab::Preview, "mobile_tab_preview"),
                (ResultTab::Edit, "mobile_tab_edit"),
                (ResultTab::Compare, "mobile_tab_compare"),
            ] {
                let selected = self.result_tab == tab;
                let label = rust_i18n::t!(key).into_owned();
                let galley =
                    ui.painter()
                        .layout_no_wrap(label.clone(), FontId::proportional(15.0), INK);
                let size = Vec2::new(galley.size().x + 26.0, 34.0);
                let (rect, response) = ui.allocate_exact_size(size, Sense::click());
                if response.clicked() {
                    self.result_tab = tab;
                    ui.ctx().request_repaint();
                }
                if selected {
                    ui.painter()
                        .rect_filled(rect, CornerRadius::same(17), ORANGE_SOFT);
                }
                let color = if selected { ORANGE } else { MUTED };
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    label,
                    FontId::proportional(15.0),
                    color,
                );
                ui.add_space(12.0);
            }
        });
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.add_space(16.0);
            if ui
                .button(if self.tts_active {
                    rust_i18n::t!("mobile_tts_stop").into_owned()
                } else {
                    rust_i18n::t!("mobile_tts_play").into_owned()
                })
                .clicked()
            {
                if self.tts_active {
                    self.tts_stop.store(true, Ordering::Relaxed);
                    self.tts_active = false;
                    self.tts_current = None;
                    self.tts_preparing = false;
                } else {
                    self.start_tts();
                }
            }
            if ui
                .add_enabled(
                    !self.tts_active,
                    egui::Button::new(rust_i18n::t!("mobile_tts_regenerate").into_owned()),
                )
                .clicked()
            {
                self.regenerate_tts();
            }
            if self.tts_active {
                if self.tts_preparing {
                    ui.spinner();
                } else {
                    ui.label("▶");
                }
                let total = self.tts_sentences.len().max(1);
                ui.add(
                    egui::ProgressBar::new(self.tts_done as f32 / total as f32)
                        .text(format!("{}/{}", self.tts_done, total))
                        .desired_width(ui.available_width().max(60.0)),
                );
            }
        });
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.label(rust_i18n::t!("mobile_tts_speed").into_owned());
            egui::ComboBox::from_id_salt("tts_speed")
                .width(64.0)
                .selected_text(format!("{}×", self.tts_speed))
                .show_ui(ui, |ui| {
                    for speed in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
                        ui.selectable_value(&mut self.tts_speed, speed, format!("{speed}×"));
                    }
                });
            self.tts_speed_shared
                .store(self.tts_speed.to_bits(), Ordering::Relaxed);
            ui.label(rust_i18n::t!("mobile_tts_volume").into_owned());
            ui.add_sized(
                [ui.available_width().max(60.0), 24.0],
                egui::Slider::new(&mut self.tts_volume, 0.0..=1.0).show_value(false),
            );
            self.tts_volume_shared
                .store(self.tts_volume.to_bits(), Ordering::Relaxed);
        });

        if self.result_tab == ResultTab::Compare
            || (self.result_tab == ResultTab::Preview && self.markdown.contains("!["))
        {
            self.request_original_texture(ui.ctx());
        }
        let body_height = (ui.max_rect().bottom() - ui.cursor().top() - 82.0).max(150.0);
        let (body_rect, _) =
            ui.allocate_exact_size(Vec2::new(content_width, body_height), Sense::hover());
        ui.painter()
            .rect_filled(body_rect, CornerRadius::same(16), WHITE);
        let content_rect = body_rect.shrink(16.0);
        let inner_width = content_rect.width();
        let inner_height = content_rect.height();
        {
            let body_ui = &mut ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(content_rect)
                    .id_salt("result_body"),
            );
            let ui = body_ui;
            ui.set_clip_rect(content_rect);
            match self.result_tab {
                ResultTab::Preview => {
                    egui::ScrollArea::vertical()
                        .max_height(inner_height)
                        .show(ui, |ui| {
                            ui.set_min_width(inner_width);
                            ui.set_max_width(inner_width);
                            if self.tts_active {
                                for (index, sentence) in self.tts_sentences.iter().enumerate() {
                                    let active = self.tts_current == Some(index);
                                    let label = egui::RichText::new(sentence)
                                        .font(FontId::monospace(15.0))
                                        .color(INK)
                                        .background_color(if active {
                                            READ_HIGHLIGHT
                                        } else {
                                            WHITE
                                        });
                                    let response = ui.label(label);
                                    if active && self.tts_scroll_to_current {
                                        response.scroll_to_me(Some(egui::Align::Center));
                                    }
                                    ui.add_space(8.0);
                                }
                                self.tts_scroll_to_current = false;
                            } else {
                                show_markdown_preview(
                                    ui,
                                    &self.markdown,
                                    inner_width,
                                    self.original_texture.as_ref(),
                                    self.original_size,
                                );
                            }
                        });
                }
                ResultTab::Edit => {
                    egui::ScrollArea::vertical()
                        .max_height(inner_height)
                        .show(ui, |ui| {
                            ui.set_min_width(inner_width);
                            ui.set_max_width(inner_width);
                            let mut layouter =
                                |ui: &egui::Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
                                    let mut job = markdown_highlight_job(ui, text.as_str());
                                    job.wrap.max_width = wrap_width;
                                    ui.fonts_mut(|fonts| fonts.layout_job(job))
                                };
                            let response = ui.add(
                                egui::TextEdit::multiline(&mut self.markdown)
                                    .desired_rows((inner_height / 24.0).max(12.0) as usize)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(inner_width)
                                    .layouter(&mut layouter),
                            );
                            if response.clicked() || response.long_touched() {
                                response.request_focus();
                            }
                            if response.changed()
                                && let Some(id) = self.current_record_id
                            {
                                history::complete(&mut self.records, id, self.markdown.clone());
                                if let Err(error) = history::save(&self.history_path, &self.records)
                                {
                                    self.notice = Some(error);
                                }
                            }
                        });
                }
                ResultTab::Compare => {
                    egui::ScrollArea::both()
                        .max_height(inner_height)
                        .show(ui, |ui| {
                            if let Some(texture) = &self.original_texture {
                                let available = Vec2::new(inner_width, inner_height);
                                let scale = (available.x / texture.size_vec2().x)
                                    .min(available.y / texture.size_vec2().y)
                                    .min(1.0);
                                ui.add(egui::Image::new((
                                    texture.id(),
                                    texture.size_vec2() * scale,
                                )));
                            } else if self.original_loading {
                                ui.spinner();
                            }
                        });
                }
            }
        }
        if self.result_tab != ResultTab::Edit {
            let swipe = ui.interact(body_rect, Id::new("result_swipe"), Sense::drag());
            if swipe.drag_stopped() && swipe.drag_delta().x.abs() > 70.0 {
                self.result_tab = match (self.result_tab, swipe.drag_delta().x.is_sign_negative()) {
                    (ResultTab::Preview, true) => ResultTab::Edit,
                    (ResultTab::Compare, false) => ResultTab::Edit,
                    (tab, _) => tab,
                };
            }
        }

        ui.columns(4, |columns| {
            if columns[0]
                .add_sized(
                    [columns[0].available_width(), 36.0],
                    egui::Button::new("↓ DOCX"),
                )
                .on_hover_text(rust_i18n::t!("mobile_save_docx").into_owned())
                .clicked()
            {
                self.save_docx();
            }
            if columns[1]
                .add_sized(
                    [columns[1].available_width(), 36.0],
                    egui::Button::new("↓ MD"),
                )
                .on_hover_text(rust_i18n::t!("mobile_save_markdown").into_owned())
                .clicked()
            {
                self.save_markdown();
            }
            if columns[2]
                .add_sized(
                    [columns[2].available_width(), 36.0],
                    egui::Button::new("↗ DOCX"),
                )
                .on_hover_text(rust_i18n::t!("mobile_share_docx").into_owned())
                .clicked()
            {
                self.share_docx();
            }
            if columns[3]
                .add_sized(
                    [columns[3].available_width(), 36.0],
                    egui::Button::new("↗ MD"),
                )
                .on_hover_text(rust_i18n::t!("mobile_share_markdown").into_owned())
                .clicked()
            {
                self.share_markdown();
            }
        });
        if let Some(notice) = &self.notice {
            ui.label(egui::RichText::new(notice).size(12.0).color(MUTED));
        }
    }
}

impl eframe::App for MobileApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_app_events(ui.ctx());
        let entered_download =
            self.screen == Screen::Download && self.last_rendered_screen != Screen::Download;
        self.last_rendered_screen = self.screen;
        let (inset_top, inset_bottom) = self.system_insets(ui.ctx());
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(PAGE_BG)
                    .inner_margin(Margin::same(0)),
            )
            .show(ui, |ui| {
                let panel = ui.max_rect();
                let insets = core::SystemInsetsPx {
                    top: inset_top,
                    bottom: inset_bottom,
                }
                .with_minimum(panel.height());
                let safe_top = (panel.top() + insets.top).clamp(panel.top(), panel.bottom());
                let safe_bottom = (panel.bottom() - insets.bottom).clamp(safe_top, panel.bottom());
                let safe_rect = Rect::from_min_max(
                    Pos2::new(panel.left(), safe_top),
                    Pos2::new(panel.right(), safe_bottom),
                );
                ui.scope_builder(egui::UiBuilder::new().max_rect(safe_rect), |ui| {
                    ui.set_clip_rect(safe_rect);
                    match self.screen {
                        Screen::Download => self.download_screen(ui, entered_download),
                        Screen::History => self.history_screen(ui),
                        Screen::Settings => self.settings_screen(ui),
                        Screen::Recognizing => self.recognizing_screen(ui),
                        Screen::Result => self.result_screen(ui),
                    }
                });
                self.handle_primary_swipe(ui.ctx(), safe_rect);
            });
        if self.recognition_active || self.camera_open || self.tts_active || self.original_loading {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

// -------------------------------------------------------------------- shared

fn engine_picker(ui: &mut egui::Ui, selected: OcrEngine, label_key: &str) -> Option<OcrEngine> {
    let mut choice = None;
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        ui.label(
            egui::RichText::new(rust_i18n::t!(label_key).into_owned())
                .size(15.0)
                .strong()
                .color(INK),
        );
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        let width = (ui.available_width() - 44.0) / 2.0;
        for engine in [OcrEngine::PaddleV6, OcrEngine::PaddleVl16] {
            if ui
                .add_sized(
                    [width, 42.0],
                    egui::Button::new(engine.label()).selected(selected == engine),
                )
                .clicked()
            {
                choice = Some(engine);
            }
        }
    });
    choice
}

fn tts_picker(ui: &mut egui::Ui, selected: TtsEngine) -> Option<TtsEngine> {
    let mut choice = None;
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        ui.label(egui::RichText::new("TTS").size(15.0).strong().color(INK));
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        let width = (ui.available_width() - 44.0) / 2.0;
        for engine in [TtsEngine::Melo, TtsEngine::Kokoro] {
            if ui
                .add_sized(
                    [width, 42.0],
                    egui::Button::new(engine.label()).selected(selected == engine),
                )
                .clicked()
            {
                choice = Some(engine);
            }
        }
    });
    choice
}

fn now_id() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * KIB;
    const GIB: u64 = MIB * KIB;
    if bytes >= GIB {
        format!("{:.1} GB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{} KB", bytes / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn swiped_primary_screen(screen: Screen, delta_x: f32, delta_y: f32) -> Screen {
    if delta_x.abs() < 70.0 || delta_x.abs() <= delta_y.abs() * 1.2 {
        return screen;
    }
    match (screen, delta_x.is_sign_negative()) {
        (Screen::Download, true) => Screen::History,
        (Screen::History, false) => Screen::Download,
        _ => screen,
    }
}

fn should_auto_resume_download(entered: bool, active: bool, missing: bool, on_wifi: bool) -> bool {
    entered && !active && missing && on_wifi
}

fn accept_recognition_event(active: bool, current_generation: u64, event_generation: u64) -> bool {
    active && current_generation == event_generation
}

fn history_card_open_response(ui: &mut egui::Ui, card: &egui::Response, id: u64) -> egui::Response {
    let mut rect = card.rect;
    rect.max.x = (rect.max.x - 64.0).max(rect.min.x);
    ui.interact(rect, Id::new(("history_card", id)), Sense::click())
}

fn markdown_highlight_job(ui: &egui::Ui, text: &str) -> egui::text::LayoutJob {
    let theme = egui_extras::syntax_highlighting::CodeTheme::from_style(ui.style());
    let mut job =
        egui_extras::syntax_highlighting::highlight(ui.ctx(), ui.style(), &theme, text, "md");
    for section in &mut job.sections {
        section.format.font_id = FontId::monospace(15.0);
    }
    job
}

fn markdown_image_path(line: &str) -> Option<&str> {
    let line = line.trim();
    let image = line.strip_prefix("![")?;
    let (_, path) = image.split_once("](")?;
    path.strip_suffix(')')
}

fn ocr_image_crop(path: &str, original_size: [usize; 2]) -> Option<Rect> {
    let (_, box_text) = path.split_once("image_box_")?;
    let (box_text, _) = box_text.rsplit_once('.')?;
    let coordinates = box_text
        .split('_')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let [left, top, right, bottom] = coordinates.as_slice() else {
        return None;
    };
    if left >= right || top >= bottom || *right > original_size[0] || *bottom > original_size[1] {
        return None;
    }
    Some(Rect::from_min_max(
        egui::pos2(
            *left as f32 / original_size[0] as f32,
            *top as f32 / original_size[1] as f32,
        ),
        egui::pos2(
            *right as f32 / original_size[0] as f32,
            *bottom as f32 / original_size[1] as f32,
        ),
    ))
}

fn show_markdown_preview(
    ui: &mut egui::Ui,
    markdown: &str,
    width: f32,
    texture: Option<&egui::TextureHandle>,
    original_size: Option<[usize; 2]>,
) {
    let mut source = String::new();
    for line in markdown.lines() {
        if let Some(path) = markdown_image_path(line) {
            if !source.is_empty() {
                let mut job = markdown_highlight_job(ui, &source);
                job.wrap.max_width = width;
                ui.label(job);
                source.clear();
            }
            if let Some(texture) = texture {
                let uv = original_size
                    .and_then(|size| ocr_image_crop(path, size))
                    .unwrap_or(Rect::from_min_max(
                        egui::pos2(0.0, 0.0),
                        egui::pos2(1.0, 1.0),
                    ));
                let pixels = texture.size_vec2() * uv.size();
                let scale = (width / pixels.x).min(1.0);
                ui.add(egui::Image::new((texture.id(), pixels * scale)).uv(uv));
            } else {
                ui.spinner();
            }
        } else {
            source.push_str(line);
            source.push('\n');
        }
    }
    if !source.is_empty() {
        let mut job = markdown_highlight_job(ui, &source);
        job.wrap.max_width = width;
        ui.label(job);
    }
}

fn persist_picked_image(
    file: &robius_file_picker::PickedFile,
    scans: &Path,
) -> Result<PathBuf, String> {
    let bytes = file.read_bytes().map_err(|error| error.to_string())?;
    image::load_from_memory(&bytes).map_err(|error| error.to_string())?;
    fs::create_dir_all(scans).map_err(|error| error.to_string())?;
    let extension = file
        .file_name()
        .and_then(|name| Path::new(name).extension())
        .and_then(|extension| extension.to_str())
        .filter(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png"
            )
        })
        .unwrap_or("jpg");
    let path = scans.join(format!("scan-{}.{}", now_id(), extension));
    fs::write(&path, bytes).map_err(|error| error.to_string())?;
    Ok(path)
}

/// Rounded top bar with optional back and right round buttons.
fn top_bar(
    ui: &mut egui::Ui,
    title: &str,
    back: bool,
    right: Option<Icon>,
    fg: Color32,
    chip: Color32,
) -> (bool, bool) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 54.0), Sense::hover());
    let painter = ui.painter();
    let cy = rect.center().y;

    let mut back_clicked = false;
    if back {
        let area = Rect::from_center_size(Pos2::new(rect.left() + 30.0, cy), Vec2::splat(36.0));
        let response = ui.interact(area, ui.id().with("mobile_top_back"), Sense::click());
        painter.rect_filled(area, CornerRadius::same(18), chip);
        paint_icon(painter, area.shrink(9.0), Icon::Back, fg);
        back_clicked = response.clicked();
    }

    painter.text(
        Pos2::new(rect.center().x, cy),
        Align2::CENTER_CENTER,
        title.to_owned(),
        FontId::proportional(17.0),
        fg,
    );

    let mut right_clicked = false;
    if let Some(icon) = right {
        let area = Rect::from_center_size(Pos2::new(rect.right() - 30.0, cy), Vec2::splat(36.0));
        let response = ui.interact(area, ui.id().with("mobile_top_right"), Sense::click());
        painter.rect_filled(area, CornerRadius::same(18), chip);
        paint_icon(painter, area.shrink(9.0), icon, fg);
        right_clicked = response.clicked();
    }

    (back_clicked, right_clicked)
}

/// Icon + label laid out horizontally, centered in the given rect.
fn paint_action(painter: &egui::Painter, rect: Rect, icon: Icon, color: Color32, label: &str) {
    let galley = painter.layout_no_wrap(label.to_owned(), FontId::proportional(13.0), color);
    let icon_size = 16.0;
    let spacing = 6.0;
    let total = icon_size + spacing + galley.size().x;
    let left = rect.center().x - total / 2.0;
    let icon_rect = Rect::from_min_size(
        Pos2::new(left, rect.center().y - icon_size / 2.0),
        Vec2::splat(icon_size),
    );
    paint_icon(painter, icon_rect, icon, color);
    painter.text(
        Pos2::new(left + icon_size + spacing, rect.center().y),
        Align2::LEFT_CENTER,
        label.to_owned(),
        FontId::proportional(13.0),
        color,
    );
}

fn paint_icon_in(ui: &mut egui::Ui, size: f32, icon: Icon, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    paint_icon(ui.painter(), rect, icon, color);
}

fn progress_bar(ui: &mut egui::Ui, width: f32, fraction: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 8.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(4), LINE);
    if fraction > 0.0 {
        let filled = Rect::from_min_size(
            rect.min,
            Vec2::new(rect.width() * fraction.clamp(0.0, 1.0), rect.height()),
        );
        ui.painter()
            .rect_filled(filled, CornerRadius::same(4), INDIGO);
    }
}

fn download_scroll_area() -> egui::ScrollArea {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .scroll_source(egui::containers::scroll_area::ScrollSource {
            drag: egui::containers::scroll_area::DragScroll::Always,
            ..Default::default()
        })
}

fn step_row(ui: &mut egui::Ui, state: StepState, number: u32, label: String) {
    ui.horizontal(|ui| {
        ui.add_space(20.0);
        {
            let (badge, _) = ui.allocate_exact_size(Vec2::splat(22.0), Sense::hover());
            let painter = ui.painter();
            let center = badge.center();
            if state == StepState::Done {
                painter.circle_filled(center, 11.0, GREEN);
                let check = Stroke::new(2.0, WHITE);
                painter.line_segment(
                    [
                        Pos2::new(center.x - 4.0, center.y + 0.5),
                        Pos2::new(center.x - 1.0, center.y + 3.5),
                    ],
                    check,
                );
                painter.line_segment(
                    [
                        Pos2::new(center.x - 1.0, center.y + 3.5),
                        Pos2::new(center.x + 4.5, center.y - 3.0),
                    ],
                    check,
                );
            } else if state == StepState::Active {
                painter.circle_filled(center, 6.0, ORANGE);
            } else {
                painter.circle_stroke(center, 10.0, Stroke::new(1.6, MUTED));
                painter.text(
                    center,
                    Align2::CENTER_CENTER,
                    number.to_string(),
                    FontId::proportional(11.0),
                    MUTED,
                );
            }
        }
        ui.add_space(10.0);
        let color = if state == StepState::Done { INK } else { MUTED };
        ui.label(egui::RichText::new(label).size(14.0).color(color));
        if state == StepState::Active {
            ui.add_space(4.0);
            ui.add(egui::Spinner::new().size(18.0));
        }
    });
}

/// Minimal vector icons so the UI does not depend on emoji font coverage.
fn paint_icon(painter: &egui::Painter, rect: Rect, icon: Icon, color: Color32) {
    let center = rect.center();
    let s = rect.width().min(rect.height()) * 0.5;
    let stroke = Stroke::new((rect.width() * 0.075).max(1.4), color);
    match icon {
        Icon::Back => {
            painter.line_segment(
                [
                    Pos2::new(center.x + s * 0.30, center.y - s * 0.62),
                    Pos2::new(center.x - s * 0.30, center.y),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.30, center.y),
                    Pos2::new(center.x + s * 0.30, center.y + s * 0.62),
                ],
                stroke,
            );
        }
        Icon::Close => {
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.55, center.y - s * 0.55),
                    Pos2::new(center.x + s * 0.55, center.y + s * 0.55),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x + s * 0.55, center.y - s * 0.55),
                    Pos2::new(center.x - s * 0.55, center.y + s * 0.55),
                ],
                stroke,
            );
        }
        Icon::Dots => {
            for dy in [-0.55f32, 0.0, 0.55] {
                painter.circle_filled(
                    Pos2::new(center.x, center.y + s * dy),
                    stroke.width * 0.8,
                    color,
                );
            }
        }
        Icon::Gear => {
            painter.circle_stroke(center, s * 0.42, stroke);
            for k in 0..8 {
                let angle = k as f32 * std::f32::consts::TAU / 8.0;
                let dir = Vec2::new(angle.cos(), angle.sin());
                painter.line_segment(
                    [center + dir * (s * 0.55), center + dir * (s * 0.82)],
                    stroke,
                );
            }
        }
        Icon::Search => {
            let lens = Pos2::new(center.x - s * 0.15, center.y - s * 0.15);
            painter.circle_stroke(lens, s * 0.50, stroke);
            painter.line_segment(
                [
                    Pos2::new(lens.x + s * 0.38, lens.y + s * 0.38),
                    Pos2::new(center.x + s * 0.62, center.y + s * 0.62),
                ],
                stroke,
            );
        }
        Icon::Camera => {
            let body = Rect::from_min_max(
                Pos2::new(center.x - s * 0.80, center.y - s * 0.40),
                Pos2::new(center.x + s * 0.80, center.y + s * 0.62),
            );
            painter.rect_stroke(body, CornerRadius::same(3), stroke, StrokeKind::Middle);
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.25, body.top()),
                    Pos2::new(center.x - s * 0.15, center.y - s * 0.75),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.15, center.y - s * 0.75),
                    Pos2::new(center.x + s * 0.25, center.y - s * 0.75),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x + s * 0.25, center.y - s * 0.75),
                    Pos2::new(center.x + s * 0.35, body.top()),
                ],
                stroke,
            );
            painter.circle_stroke(center + Vec2::new(0.0, s * 0.10), s * 0.30, stroke);
        }
        Icon::Gallery => {
            let frame = Rect::from_center_size(center, Vec2::new(s * 1.55, s * 1.4));
            painter.rect_stroke(frame, CornerRadius::same(2), stroke, StrokeKind::Middle);
            painter.circle_filled(center + Vec2::new(-s * 0.35, -s * 0.3), s * 0.12, color);
            painter.line_segment(
                [
                    Pos2::new(frame.left() + s * 0.15, frame.bottom() - s * 0.2),
                    Pos2::new(center.x, center.y + s * 0.12),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x, center.y + s * 0.12),
                    Pos2::new(frame.right() - s * 0.1, frame.bottom() - s * 0.18),
                ],
                stroke,
            );
        }
        Icon::Download => {
            painter.line_segment(
                [
                    Pos2::new(center.x, center.y - s * 0.70),
                    Pos2::new(center.x, center.y + s * 0.25),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.35, center.y - s * 0.05),
                    Pos2::new(center.x, center.y + s * 0.30),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x + s * 0.35, center.y - s * 0.05),
                    Pos2::new(center.x, center.y + s * 0.30),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.55, center.y + s * 0.68),
                    Pos2::new(center.x + s * 0.55, center.y + s * 0.68),
                ],
                stroke,
            );
        }
        Icon::Trash => {
            let body = Rect::from_min_max(
                Pos2::new(center.x - s * 0.48, center.y - s * 0.25),
                Pos2::new(center.x + s * 0.48, center.y + s * 0.68),
            );
            painter.rect_stroke(body, CornerRadius::same(2), stroke, StrokeKind::Middle);
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.62, center.y - s * 0.43),
                    Pos2::new(center.x + s * 0.62, center.y - s * 0.43),
                ],
                stroke,
            );
            painter.line_segment(
                [
                    Pos2::new(center.x - s * 0.22, center.y - s * 0.62),
                    Pos2::new(center.x + s * 0.22, center.y - s * 0.62),
                ],
                stroke,
            );
        }
    }
}

fn install_cjk_font(ctx: &egui::Context) {
    let bytes = [
        "/system/fonts/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    .into_iter()
    .find_map(|path| std::fs::read(path).ok())
    .unwrap_or_else(|| include_bytes!("../assets/fonts/NotoSansCJKsc-Regular.otf").to_vec());
    let mut fonts = egui::FontDefinitions::default();
    let name = "noto-sans-cjk-sc".to_owned();
    fonts
        .font_data
        .insert(name.clone(), egui::FontData::from_owned(bytes).into());
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, name.clone());
    }
    ctx.set_fonts(fonts);
}

#[cfg(test)]
mod tests {
    use super::{Screen, swiped_primary_screen};
    use eframe::egui::{
        self, Event, MouseWheelUnit, PointerButton, Pos2, RawInput, Rect, TouchPhase, Vec2,
    };

    #[test]
    fn tapping_a_scan_card_opens_it() {
        fn frame(ctx: &egui::Context, events: Vec<Event>) -> bool {
            let mut clicked = false;
            let mut output = ctx.run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(220.0, 160.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let card = egui::Frame::default()
                        .show(ui, |ui| ui.set_min_size(Vec2::new(200.0, 100.0)))
                        .response;
                    clicked = super::history_card_open_response(ui, &card, 7).clicked();
                },
            );
            output.textures_delta.clear();
            clicked
        }

        let ctx = egui::Context::default();
        frame(&ctx, vec![]);
        frame(
            &ctx,
            vec![
                Event::PointerMoved(Pos2::new(50.0, 50.0)),
                Event::PointerButton {
                    pos: Pos2::new(50.0, 50.0),
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
            ],
        );
        assert!(frame(
            &ctx,
            vec![Event::PointerButton {
                pos: Pos2::new(50.0, 50.0),
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Default::default(),
            }],
        ));
    }

    #[test]
    fn markdown_source_uses_multiple_syntax_colors() {
        egui::__run_test_ui(|ui| {
            let job = super::markdown_highlight_job(ui, "# Heading\n\n**bold** text\n");
            let colors: std::collections::HashSet<_> =
                job.sections.iter().map(|part| part.format.color).collect();
            assert!(colors.len() > 1, "Markdown markup should be highlighted");
        });
    }

    #[test]
    fn tiny_and_large_model_sizes_are_not_shown_as_zero_mb() {
        assert_eq!(super::format_bytes(0), "0 B");
        assert_eq!(super::format_bytes(1023), "1023 B");
        assert_eq!(super::format_bytes(1024), "1 KB");
        assert_eq!(super::format_bytes(65_536), "64 KB");
        assert_eq!(super::format_bytes(1_048_576), "1.0 MB");
        assert_eq!(super::format_bytes(2_147_483_648), "2.0 GB");
    }

    #[test]
    fn ocr_markdown_image_uses_the_matching_original_crop() {
        let path = super::markdown_image_path("![Image](imgs/img_in_image_box_76_342_353_617.jpg)");
        assert_eq!(path, Some("imgs/img_in_image_box_76_342_353_617.jpg"));
        let crop = super::ocr_image_crop(path.unwrap(), [1000, 2000]).unwrap();
        assert_eq!(crop.min, egui::pos2(0.076, 0.171));
        assert_eq!(crop.max, egui::pos2(0.353, 0.3085));
    }

    #[test]
    fn cancelled_ocr_results_cannot_overwrite_a_restarted_scan() {
        assert!(!super::accept_recognition_event(true, 8, 7));
        assert!(!super::accept_recognition_event(false, 8, 8));
        assert!(super::accept_recognition_event(true, 8, 8));
    }

    #[test]
    fn returning_to_download_auto_resumes_once_only_on_wifi() {
        assert!(super::should_auto_resume_download(true, false, true, true));
        assert!(!super::should_auto_resume_download(
            false, false, true, true
        ));
        assert!(!super::should_auto_resume_download(true, true, true, true));
        assert!(!super::should_auto_resume_download(
            true, false, false, true
        ));
        assert!(!super::should_auto_resume_download(
            true, false, true, false
        ));
    }

    #[test]
    fn download_content_scrolls_to_reveal_lower_models() {
        fn frame(ctx: &egui::Context, events: Vec<Event>) -> f32 {
            let mut offset = 0.0;
            let mut output = ctx.run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 400.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let output = super::download_scroll_area()
                        .max_height(120.0)
                        .show(ui, |ui| {
                            for _ in 0..30 {
                                ui.label("model progress");
                            }
                        });
                    offset = output.state.offset.y;
                },
            );
            output.textures_delta.clear();
            offset
        }

        let ctx = egui::Context::default();
        frame(&ctx, vec![]);
        let offset = frame(
            &ctx,
            vec![
                Event::PointerMoved(Pos2::new(50.0, 90.0)),
                Event::MouseWheel {
                    unit: MouseWheelUnit::Point,
                    delta: Vec2::new(0.0, -70.0),
                    phase: TouchPhase::Move,
                    modifiers: Default::default(),
                },
            ],
        );
        assert!(
            offset > 0.0,
            "scroll should reveal content below the viewport"
        );
    }

    #[test]
    fn horizontal_swipe_switches_primary_pages_without_hijacking_vertical_scroll() {
        assert!(matches!(
            swiped_primary_screen(Screen::Download, -100.0, 8.0),
            Screen::History
        ));
        assert!(matches!(
            swiped_primary_screen(Screen::History, -100.0, 8.0),
            Screen::History
        ));
        assert!(matches!(
            swiped_primary_screen(Screen::History, 8.0, 100.0),
            Screen::History
        ));
        assert!(matches!(
            swiped_primary_screen(Screen::Result, -100.0, 8.0),
            Screen::Result
        ));
    }
}
