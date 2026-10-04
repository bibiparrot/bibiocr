//! Desktop playback for the mobile Melo/Kokoro synthesis implementation.
use crate::{
    download::{self, DownloadOptions},
    settings::{self, AppSettings, ProxyMode, TtsEngine},
    tts,
};
use eframe::egui;
use rodio::{OutputStreamBuilder, Sink, buffer::SamplesBuffer};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

enum Event {
    Status(String),
    Sentence(usize),
    Finished(Result<(), String>),
}

pub struct Speech {
    pub open: bool,
    events: Option<Receiver<Event>>,
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    speed: Arc<AtomicU32>,
    volume: Arc<AtomicU32>,
    sentences: Vec<String>,
    current: Option<usize>,
    source: String,
    status: String,
    revision: u128,
}

impl Default for Speech {
    fn default() -> Self {
        Self {
            open: false,
            events: None,
            stop: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            speed: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            volume: Arc::new(AtomicU32::new(0.8_f32.to_bits())),
            sentences: Vec::new(),
            current: None,
            source: String::new(),
            status: String::new(),
            revision: 0,
        }
    }
}

impl Drop for Speech {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Speech {
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.events = None;
        self.current = None;
        self.status.clear();
    }

    pub fn poll(&mut self, markdown: &str, ctx: &egui::Context) {
        if self.events.is_some() && self.source != markdown {
            self.stop();
        }
        let mut finished = None;
        if let Some(events) = &self.events {
            while let Ok(event) = events.try_recv() {
                match event {
                    Event::Status(status) => self.status = status,
                    Event::Sentence(index) => self.current = Some(index),
                    Event::Finished(result) => finished = Some(result),
                }
            }
            ctx.request_repaint_after(Duration::from_millis(50));
        }
        if let Some(result) = finished {
            self.events = None;
            self.current = None;
            self.status = result
                .err()
                .unwrap_or_else(|| rust_i18n::t!("tts_finished").into_owned());
        }
    }

    fn start(&mut self, markdown: &str, settings: &AppSettings, download_only: bool) {
        self.stop();
        self.sentences = tts::sentences(markdown);
        if !download_only && self.sentences.is_empty() {
            self.status = rust_i18n::t!("tts_empty").into_owned();
            return;
        }
        self.source = markdown.to_owned();
        self.stop = Arc::new(AtomicBool::new(false));
        self.paused = Arc::new(AtomicBool::new(false));
        self.speed.store(
            settings.tts_speed.clamp(0.5, 2.0).to_bits(),
            Ordering::Relaxed,
        );
        self.volume.store(
            settings.tts_volume.clamp(0.0, 1.0).to_bits(),
            Ordering::Relaxed,
        );
        let stop = Arc::clone(&self.stop);
        let paused = Arc::clone(&self.paused);
        let speed = Arc::clone(&self.speed);
        let volume = Arc::clone(&self.volume);
        let sentences = self.sentences.clone();
        let settings = settings.clone();
        let revision = self.revision;
        let (sender, receiver) = mpsc::channel();
        self.events = Some(receiver);
        self.status = rust_i18n::t!("tts_preparing").into_owned();
        thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let models = model_directory(&settings);
                ensure_models(&models, &settings, &stop, |file| {
                    let _ = sender.send(Event::Status(format!(
                        "{} {file}",
                        rust_i18n::t!("tts_downloading")
                    )));
                })?;
                if download_only || stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let mut stream = OutputStreamBuilder::open_default_stream()
                    .map_err(|error| format!("Audio output: {error}"))?;
                stream.log_on_drop(false);
                let cache = settings::runtime_directory()
                    .join("tts/audio")
                    .join(settings.tts_engine.cache_name())
                    .join(revision.to_string());
                let mut skipped = 0;
                let mut spoken = 0;
                let mut last_error = String::new();
                for (index, sentence) in sentences.iter().enumerate() {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let pieces = match tts::synthesize_resilient_cached(
                        sentence,
                        &models,
                        &cache,
                        &stop,
                        settings.tts_engine,
                    ) {
                        Ok(pieces) => pieces,
                        Err(error) => {
                            skipped += 1;
                            last_error = error;
                            continue;
                        }
                    };
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let _ = sender.send(Event::Sentence(index));
                    for piece in pieces {
                        let sink = Sink::connect_new(stream.mixer());
                        let mut stretcher = wsola::TimeStretch::new(piece.sample_rate as u32, 1)
                            .map_err(|error| error.to_string())?;
                        for chunk in piece
                            .samples
                            .chunks((piece.sample_rate as usize / 10).max(1))
                        {
                            if stop.load(Ordering::Relaxed) {
                                sink.stop();
                                return Ok(());
                            }
                            let tempo =
                                f32::from_bits(speed.load(Ordering::Relaxed)).clamp(0.5, 2.0);
                            let pcm = tts::stretch_pcm_chunk(&mut stretcher, chunk, tempo, false);
                            append_pcm(&sink, &pcm, piece.sample_rate);
                            // Keep less than a few chunks queued so live speed changes respond promptly.
                            while sink.len() > 2 || paused.load(Ordering::Relaxed) {
                                control_sink(&sink, &stop, &paused, &volume)?;
                                thread::sleep(Duration::from_millis(20));
                            }
                            control_sink(&sink, &stop, &paused, &volume)?;
                        }
                        let tempo = f32::from_bits(speed.load(Ordering::Relaxed)).clamp(0.5, 2.0);
                        append_pcm(
                            &sink,
                            &tts::stretch_pcm_chunk(&mut stretcher, &[], tempo, true),
                            piece.sample_rate,
                        );
                        while !sink.empty() {
                            control_sink(&sink, &stop, &paused, &volume)?;
                            thread::sleep(Duration::from_millis(20));
                        }
                    }
                    spoken += 1;
                }
                if skipped > 0 {
                    return Err(format!(
                        "Read {spoken} segments; skipped {skipped}: {last_error}"
                    ));
                }
                Ok(())
            }))
            .unwrap_or_else(|_| Err("TTS worker failed; please retry".to_owned()));
            let _ = sender.send(Event::Finished(result));
        });
    }

    pub fn window(&mut self, ctx: &egui::Context, markdown: &str, settings: &mut AppSettings) {
        let mut open = self.open;
        let before = (
            settings.tts_engine,
            settings.tts_model_dir.clone(),
            settings.tts_speed,
            settings.tts_volume,
        );
        egui::Window::new(rust_i18n::t!("tts_title"))
            .open(&mut open)
            .default_width(540.0)
            .show(ctx, |ui| {
                let active = self.events.is_some();
                ui.add_enabled_ui(!active, |ui| {
                    egui::ComboBox::from_id_salt("desktop-tts-engine")
                        .selected_text(settings.tts_engine.label())
                        .show_ui(ui, |ui| {
                            for engine in [TtsEngine::Melo, TtsEngine::Kokoro] {
                                ui.selectable_value(
                                    &mut settings.tts_engine,
                                    engine,
                                    engine.label(),
                                );
                            }
                        });
                    ui.label(model_directory(settings).display().to_string());
                    ui.horizontal(|ui| {
                        if ui.button(rust_i18n::t!("tts_model_folder")).clicked() {
                            if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                settings.tts_model_dir = Some(path);
                            }
                        }
                        if ui.button(rust_i18n::t!("tts_download_models")).clicked() {
                            self.start(markdown, settings, true);
                        }
                    });
                });
                ui.horizontal(|ui| {
                    if !active {
                        if ui.button(rust_i18n::t!("tts_play")).clicked() {
                            self.start(markdown, settings, false);
                        }
                        if ui.button(rust_i18n::t!("tts_regenerate")).clicked() {
                            self.revision = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos();
                            self.start(markdown, settings, false);
                        }
                    } else {
                        let paused = self.paused.load(Ordering::Relaxed);
                        if ui
                            .button(if paused {
                                rust_i18n::t!("tts_resume")
                            } else {
                                rust_i18n::t!("tts_pause")
                            })
                            .clicked()
                        {
                            self.paused.store(!paused, Ordering::Relaxed);
                        }
                        if ui.button(rust_i18n::t!("tts_stop")).clicked() {
                            self.stop();
                        }
                    }
                });
                ui.add(
                    egui::Slider::new(&mut settings.tts_speed, 0.5..=2.0)
                        .text(rust_i18n::t!("tts_speed")),
                );
                ui.add(
                    egui::Slider::new(&mut settings.tts_volume, 0.0..=1.0)
                        .text(rust_i18n::t!("tts_volume")),
                );
                self.speed
                    .store(settings.tts_speed.to_bits(), Ordering::Relaxed);
                self.volume
                    .store(settings.tts_volume.to_bits(), Ordering::Relaxed);
                ui.label(&self.status);
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for (index, sentence) in self.sentences.iter().enumerate() {
                            let response = ui.add(
                                egui::Label::new(egui::RichText::new(sentence).background_color(
                                    if self.current == Some(index) {
                                        egui::Color32::LIGHT_YELLOW
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    },
                                ))
                                .wrap(),
                            );
                            if self.current == Some(index) {
                                response.scroll_to_me(Some(egui::Align::Center));
                            }
                        }
                    });
            });
        self.open = open;
        if before
            != (
                settings.tts_engine,
                settings.tts_model_dir.clone(),
                settings.tts_speed,
                settings.tts_volume,
            )
        {
            if let Err(error) = settings.save() {
                self.status = error;
            }
        }
    }
}

fn control_sink(
    sink: &Sink,
    stop: &AtomicBool,
    paused: &AtomicBool,
    volume: &AtomicU32,
) -> Result<(), String> {
    if stop.load(Ordering::Relaxed) {
        sink.stop();
        return Err("Stopped".to_owned());
    }
    sink.set_volume(f32::from_bits(volume.load(Ordering::Relaxed)).clamp(0.0, 1.0));
    if paused.load(Ordering::Relaxed) {
        sink.pause();
    } else {
        sink.play();
    }
    Ok(())
}

fn append_pcm(sink: &Sink, pcm: &[i16], sample_rate: i32) {
    if !pcm.is_empty() {
        sink.append(SamplesBuffer::new(
            1,
            sample_rate as u32,
            pcm.iter()
                .map(|sample| f32::from(*sample) / 32768.0)
                .collect::<Vec<_>>(),
        ));
    }
}

fn model_directory(settings: &AppSettings) -> PathBuf {
    settings
        .tts_model_dir
        .clone()
        .unwrap_or_else(|| settings::runtime_directory().join("tts/models"))
}

const MELO: [(&str, &str, u64); 3] = [
    (
        "melo-model.onnx",
        "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/model.onnx",
        170_429_550,
    ),
    (
        "melo-lexicon.txt",
        "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/lexicon.txt",
        6_837_671,
    ),
    (
        "melo-tokens.txt",
        "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6/tokens.txt",
        655,
    ),
];
const KOKORO: [(&str, &str, u64); 3] = [
    (
        "kokoro-model.onnx",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx",
        325_532_232,
    ),
    (
        "af_heart.bin",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_heart.bin",
        522_240,
    ),
    (
        "kokoro-tokenizer.json",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/tokenizer.json",
        3_497,
    ),
];

fn ensure_models(
    directory: &Path,
    settings: &AppSettings,
    stop: &AtomicBool,
    mut status: impl FnMut(&str),
) -> Result<(), String> {
    let models = if settings.tts_engine == TtsEngine::Melo {
        &MELO
    } else {
        &KOKORO
    };
    let options = DownloadOptions {
        proxy: if settings.use_proxy {
            settings.proxy.clone()
        } else {
            String::new()
        },
        no_proxy: settings.proxy_mode() == ProxyMode::None,
        hf_endpoint: if settings.use_hf_mirror {
            settings.hf_endpoint.clone()
        } else {
            "https://huggingface.co".to_owned()
        },
        github_proxy: String::new(),
        resume: settings.resume_downloads,
        retries: settings.download_retries,
    };
    for (name, url, size) in models {
        if stop.load(Ordering::Relaxed) {
            return Err("Stopped".to_owned());
        }
        let path = directory.join(name);
        if path.metadata().is_ok_and(|meta| meta.len() == *size) {
            continue;
        }
        if path.is_file() {
            fs::remove_file(&path).map_err(|error| error.to_string())?;
        }
        status(name);
        let url = url.replacen(
            "https://huggingface.co",
            options.hf_endpoint.trim_end_matches('/'),
            1,
        );
        download::download_asset(url, path.clone(), &options)?;
        if !path.metadata().is_ok_and(|meta| meta.len() == *size) {
            return Err(format!(
                "Invalid downloaded voice model: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_volume_and_stop_control_the_audio_sink() {
        let (sink, _output) = Sink::new();
        let stop = AtomicBool::new(false);
        let paused = AtomicBool::new(true);
        let volume = AtomicU32::new(0.4_f32.to_bits());
        append_pcm(&sink, &[1, 2, 3], 24_000);
        control_sink(&sink, &stop, &paused, &volume).unwrap();
        assert!(sink.is_paused());
        assert_eq!(sink.volume(), 0.4);
        paused.store(false, Ordering::Relaxed);
        control_sink(&sink, &stop, &paused, &volume).unwrap();
        assert!(!sink.is_paused());
        stop.store(true, Ordering::Relaxed);
        assert!(control_sink(&sink, &stop, &paused, &volume).is_err());
    }

    #[test]
    fn opening_a_new_document_cancels_the_previous_worker() {
        let mut speech = Speech::default();
        let (_sender, receiver) = mpsc::channel();
        speech.events = Some(receiver);
        speech.source = "old document".to_owned();
        let cancellation = Arc::clone(&speech.stop);
        speech.poll("new document", &egui::Context::default());
        assert!(cancellation.load(Ordering::Relaxed));
        assert!(speech.events.is_none());
    }

    #[test]
    fn cached_voice_models_work_without_network() {
        let directory = std::env::temp_dir().join(format!(
            "bibiocr-voice-models-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&directory).unwrap();
        for (name, _, size) in MELO {
            fs::File::create(directory.join(name))
                .unwrap()
                .set_len(size)
                .unwrap();
        }
        let settings = AppSettings::default();
        ensure_models(&directory, &settings, &AtomicBool::new(false), |_| {
            panic!("Valid cached models must not trigger a download")
        })
        .unwrap();
        for (name, _, _) in MELO {
            fs::remove_file(directory.join(name)).unwrap();
        }
        fs::remove_dir(directory).unwrap();
    }
}
