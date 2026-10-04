use crate::core::{self, ModelSpec};
use crate::settings::{DownloadSettings, OcrEngine, TtsEngine};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Duration,
};

#[derive(Debug)]
pub enum DownloadEvent {
    Started(usize),
    Progress(usize, u64, u64),
    Complete(usize),
    Failed(usize, String),
    Finished,
}

pub struct DownloadTask {
    pub events: Receiver<DownloadEvent>,
    paused: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

impl DownloadTask {
    pub fn toggle_pause(&self) {
        self.paused.fetch_xor(true, Ordering::Relaxed);
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

pub fn start(
    model_dir: PathBuf,
    settings: DownloadSettings,
    locale: &'static str,
    engine: OcrEngine,
    tts_engine: TtsEngine,
) -> DownloadTask {
    let (sender, events) = mpsc::channel();
    let paused = Arc::new(AtomicBool::new(false));
    let worker_pause = Arc::clone(&paused);
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    thread::spawn(move || {
        for (index, model) in core::required_models(engine, tts_engine) {
            if worker_cancelled.load(Ordering::Relaxed) {
                return;
            }
            if installed(model_dir.as_path(), model) {
                let _ = sender.send(DownloadEvent::Complete(index));
                continue;
            }
            let _ = sender.send(DownloadEvent::Started(index));
            if let Err(error) = download_with_retry(
                model,
                &model_dir,
                index,
                &sender,
                &worker_pause,
                &worker_cancelled,
                &settings,
                locale,
            ) {
                if worker_cancelled.load(Ordering::Relaxed) {
                    return;
                }
                let _ = sender.send(DownloadEvent::Failed(index, error));
                return;
            }
            let _ = sender.send(DownloadEvent::Complete(index));
        }
        let _ = sender.send(DownloadEvent::Finished);
    });
    DownloadTask {
        events,
        paused,
        cancelled,
    }
}

pub fn installed(model_dir: &Path, model: &ModelSpec) -> bool {
    model_dir
        .join(model.file_name)
        .metadata()
        .is_ok_and(|metadata| metadata.len() == model.expected_bytes)
}

pub fn cleanup_obsolete_layout(model_dir: &Path) {
    let previous = model_dir.join("inference.onnx");
    let current = model_dir.join(crate::core::MODELS[2].file_name);
    let previous_partial = model_dir.join("inference.onnx.part");
    let current_partial = model_dir.join("pp-doclayoutv3_onnx.onnx.part");
    if !current_partial.exists() && !current.exists() && previous_partial.exists() {
        let _ = fs::rename(previous_partial, current_partial);
    }
    if !current.exists()
        && previous
            .metadata()
            .is_ok_and(|metadata| metadata.len() == crate::core::MODELS[2].expected_bytes)
    {
        let _ = fs::rename(&previous, &current);
    }
    if installed(model_dir, &crate::core::MODELS[2]) {
        if previous
            .metadata()
            .is_ok_and(|metadata| metadata.len() == crate::core::MODELS[2].expected_bytes)
        {
            let _ = fs::remove_file(previous);
        }
        for name in ["pp-doclayout_plus-l.onnx", "pp-doclayout_plus-l.onnx.part"] {
            let _ = fs::remove_file(model_dir.join(name));
        }
    }
}

fn download_with_retry(
    model: &ModelSpec,
    model_dir: &Path,
    index: usize,
    sender: &Sender<DownloadEvent>,
    paused: &AtomicBool,
    cancelled: &AtomicBool,
    settings: &DownloadSettings,
    locale: &str,
) -> Result<(), String> {
    for attempt in 0..3 {
        if cancelled.load(Ordering::Relaxed) {
            return Err("Download cancelled".to_owned());
        }
        match download(
            model, model_dir, index, sender, paused, cancelled, settings, locale,
        ) {
            Ok(()) => return Ok(()),
            Err(error) if attempt == 2 || cancelled.load(Ordering::Relaxed) => {
                return Err(error);
            }
            Err(_) => thread::sleep(Duration::from_secs(1)),
        }
    }
    unreachable!()
}

fn download(
    model: &ModelSpec,
    model_dir: &Path,
    index: usize,
    sender: &Sender<DownloadEvent>,
    paused: &AtomicBool,
    cancelled: &AtomicBool,
    settings: &DownloadSettings,
    locale: &str,
) -> Result<(), String> {
    fs::create_dir_all(model_dir).map_err(|error| error.to_string())?;
    let destination = model_dir.join(model.file_name);
    let partial = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
    ));
    let offset = resume_offset(&partial, settings.resume);
    let _ = sender.send(DownloadEvent::Progress(index, offset, model.expected_bytes));
    wait_until_ready(paused, cancelled, settings)?;
    let env_endpoint = std::env::var("HF_ENDPOINT").ok();
    let url = settings.resolve_url(model.url, locale, env_endpoint.as_deref());
    let mut request = ureq::get(&url);
    if offset > 0 {
        request = request.header("Range", format!("bytes={offset}-"));
    }
    let mut response = request.call().map_err(|error| error.to_string())?;
    let resumed = response.status().as_u16() == 206 && offset > 0;
    let mut downloaded = if resumed { offset } else { 0 };
    let content_length = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let total = content_length.map_or(model.expected_bytes, |length| downloaded + length);
    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&partial)
        .map_err(|error| error.to_string())?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0_u8; 256 * 1024];
    loop {
        wait_until_ready(paused, cancelled, settings)?;
        let count = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| error.to_string())?;
        downloaded += count as u64;
        let _ = sender.send(DownloadEvent::Progress(index, downloaded, total));
    }
    output.flush().map_err(|error| error.to_string())?;
    if downloaded != model.expected_bytes {
        return Err(format!(
            "{} has {downloaded} bytes; expected {}",
            model.file_name, model.expected_bytes
        ));
    }
    fs::rename(partial, destination).map_err(|error| error.to_string())
}

fn wait_until_ready(
    paused: &AtomicBool,
    cancelled: &AtomicBool,
    settings: &DownloadSettings,
) -> Result<(), String> {
    loop {
        if cancelled.load(Ordering::Relaxed) {
            return Err("Download cancelled".to_owned());
        }
        if !paused.load(Ordering::Relaxed) && !wifi_required_but_unavailable(settings)? {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn wifi_required_but_unavailable(settings: &DownloadSettings) -> Result<bool, String> {
    if !settings.wifi_only {
        return Ok(false);
    }
    #[cfg(target_os = "android")]
    return crate::android_bridge::is_wifi_connected().map(|connected| !connected);
    #[cfg(not(target_os = "android"))]
    Ok(false)
}

fn resume_offset(partial: &Path, enabled: bool) -> u64 {
    if enabled {
        partial.metadata().map_or(0, |metadata| metadata.len())
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{installed, resume_offset};
    use crate::core::{MODELS, ModelGroup, ModelSpec};
    use crate::settings::{OcrEngine, TtsEngine};
    use std::{
        fs,
        io::{Read, Write},
        net::TcpListener,
        sync::{atomic::AtomicBool, mpsc},
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn interrupted_download_resumes_before_reporting_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url: &'static str = Box::leak(
            format!("http://{}/model.onnx", listener.local_addr().unwrap()).into_boxed_str(),
        );
        let server = thread::spawn(move || {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 2048];
                let count = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..count]);
                if attempt == 0 {
                    assert!(!request.to_ascii_lowercase().contains("range:"));
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nabc",
                        )
                        .unwrap();
                } else {
                    assert!(
                        request.to_ascii_lowercase().contains("range: bytes=3-"),
                        "{request}"
                    );
                    stream
                        .write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nContent-Range: bytes 3-7/8\r\nConnection: close\r\n\r\ndefgh")
                        .unwrap();
                }
            }
        });
        let root = std::env::temp_dir().join(format!(
            "bibiocr-retry-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let model = ModelSpec {
            group: ModelGroup::V6,
            name: "test",
            file_name: "model.onnx",
            url,
            expected_bytes: 8,
        };
        let (sender, _) = mpsc::channel();
        super::download_with_retry(
            &model,
            &root,
            0,
            &sender,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &Default::default(),
            "en",
        )
        .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(root.join("model.onnx")).unwrap(), b"abcdefgh");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installed_requires_the_published_size() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-download-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let model = MODELS[2];
        let file = fs::File::create(root.join(model.file_name)).unwrap();
        file.set_len(model.expected_bytes - 1).unwrap();
        assert!(!installed(&root, &model));
        file.set_len(model.expected_bytes).unwrap();
        assert!(installed(&root, &model));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_setting_controls_partial_file_offset() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-resume-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let partial = root.join("model.gguf.part");
        let file = fs::File::create(&partial).unwrap();
        file.set_len(128).unwrap();

        assert_eq!(resume_offset(&partial, true), 128);
        assert_eq!(resume_offset(&partial, false), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selected_download_reuses_cached_v6_without_requesting_vl() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-selected-download-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let required =
            crate::core::required_models(OcrEngine::PaddleV6, TtsEngine::Melo).collect::<Vec<_>>();
        for (_, model) in &required {
            fs::File::create(root.join(model.file_name))
                .unwrap()
                .set_len(model.expected_bytes)
                .unwrap();
        }
        let events = super::start(
            root.clone(),
            Default::default(),
            "en",
            OcrEngine::PaddleV6,
            TtsEngine::Melo,
        )
        .events
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, super::DownloadEvent::Complete(_)))
                .count(),
            required.len()
        );
        assert!(matches!(
            events.last(),
            Some(super::DownloadEvent::Finished)
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn obsolete_plus_l_cache_is_removed_only_after_shared_v3_exists() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-layout-migration-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let obsolete = root.join("pp-doclayout_plus-l.onnx");
        fs::write(&obsolete, b"old").unwrap();
        super::cleanup_obsolete_layout(&root);
        assert!(obsolete.exists());
        fs::File::create(root.join(MODELS[2].file_name))
            .unwrap()
            .set_len(MODELS[2].expected_bytes)
            .unwrap();
        super::cleanup_obsolete_layout(&root);
        assert!(!obsolete.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn previously_downloaded_v3_layout_is_renamed_without_downloading_again() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-layout-cache-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let previous = root.join("inference.onnx");
        fs::File::create(&previous)
            .unwrap()
            .set_len(MODELS[2].expected_bytes)
            .unwrap();
        super::cleanup_obsolete_layout(&root);
        assert!(super::installed(&root, &MODELS[2]));
        assert!(!previous.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_v3_layout_download_keeps_its_resume_offset_after_rename() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-layout-partial-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let previous = root.join("inference.onnx.part");
        fs::write(&previous, b"partially downloaded").unwrap();
        super::cleanup_obsolete_layout(&root);
        let current = root.join("pp-doclayoutv3_onnx.onnx.part");
        assert_eq!(fs::read(current).unwrap(), b"partially downloaded");
        assert!(!previous.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "downloads the real Melo model and lexicon once"]
    fn real_bilingual_tts_assets_download_and_reuse_cache() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("downloads");
        let task = super::start(
            root.clone(),
            Default::default(),
            "zh-CN",
            OcrEngine::PaddleV6,
            TtsEngine::Melo,
        );
        for event in task.events {
            match event {
                super::DownloadEvent::Failed(_, error) => panic!("{error}"),
                super::DownloadEvent::Finished => break,
                _ => {}
            }
        }
        assert!(MODELS[3..6].iter().all(|model| installed(&root, model)));

        let offline_settings = crate::settings::DownloadSettings {
            hf_endpoint: "http://127.0.0.1:9".to_owned(),
            ..Default::default()
        };
        let cached = super::start(
            root,
            offline_settings,
            "zh-CN",
            OcrEngine::PaddleV6,
            TtsEngine::Melo,
        );
        for event in cached.events {
            match event {
                super::DownloadEvent::Failed(_, error) => panic!("cache missed: {error}"),
                super::DownloadEvent::Finished => break,
                _ => {}
            }
        }
    }
}
