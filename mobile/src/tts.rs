use crate::settings::TtsEngine;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsVitsModelConfig,
};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

static ENGINE: OnceLock<Mutex<Option<OfflineTts>>> = OnceLock::new();
// ponytail: Session-only cache; use disk/LRU if long documents cause memory pressure.
static AUDIO_CACHE: OnceLock<Mutex<HashMap<String, Arc<TtsAudio>>>> = OnceLock::new();
// ponytail: One speaker at a time; use per-document locks only if parallel TTS is needed.
static PERSISTENT_CACHE_LOCK: Mutex<()> = Mutex::new(());

pub struct TtsAudio {
    pub samples: Vec<i16>,
    pub sample_rate: i32,
}

fn cache_path(document_dir: &Path, text: &str) -> PathBuf {
    let hash = text
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    document_dir.join(format!("{hash:016x}.pcm"))
}

fn load_cached_audio(path: &Path, text: &str) -> Option<Arc<TtsAudio>> {
    if fs::metadata(path).ok()?.len() > 100 * 1024 * 1024 {
        return None;
    }
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 20 || &bytes[..8] != b"BIBITTS1" {
        return None;
    }
    let sample_rate = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let text_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let sample_count = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    let pcm_start = 20usize.checked_add(text_len)?;
    let expected = pcm_start.checked_add(sample_count.checked_mul(2)?)?;
    if sample_rate <= 0
        || sample_count == 0
        || expected != bytes.len()
        || bytes.get(20..pcm_start)? != text.as_bytes()
    {
        return None;
    }
    let samples = bytes[pcm_start..]
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    Some(Arc::new(TtsAudio {
        samples,
        sample_rate,
    }))
}

fn save_cached_audio(path: &Path, text: &str, audio: &TtsAudio) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let text_len = u32::try_from(text.len()).map_err(|error| error.to_string())?;
    let sample_count = u32::try_from(audio.samples.len()).map_err(|error| error.to_string())?;
    let mut bytes = Vec::with_capacity(20 + text.len() + audio.samples.len() * 2);
    bytes.extend_from_slice(b"BIBITTS1");
    bytes.extend_from_slice(&audio.sample_rate.to_le_bytes());
    bytes.extend_from_slice(&text_len.to_le_bytes());
    bytes.extend_from_slice(&sample_count.to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
    for sample in &audio.samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    let temp = path.with_extension("tmp");
    fs::write(&temp, bytes).map_err(|error| error.to_string())?;
    if path.exists() {
        fs::remove_file(path).map_err(|error| error.to_string())?;
    }
    fs::rename(temp, path).map_err(|error| error.to_string())
}

fn cached_audio(
    document_dir: &Path,
    text: &str,
    generate: impl FnOnce() -> Result<Arc<TtsAudio>, String>,
) -> Result<Arc<TtsAudio>, String> {
    let _guard = PERSISTENT_CACHE_LOCK
        .lock()
        .map_err(|error| error.to_string())?;
    let path = cache_path(document_dir, text);
    if let Some(audio) = load_cached_audio(&path, text) {
        return Ok(audio);
    }
    let audio = generate()?;
    save_cached_audio(&path, text, &audio)?;
    Ok(audio)
}

pub(crate) fn stretch_pcm_chunk(
    stretcher: &mut wsola::TimeStretch,
    samples: &[i16],
    speed: f32,
    finish: bool,
) -> Vec<i16> {
    stretcher.set_tempo(speed);
    stretcher.push(
        &samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect::<Vec<_>>(),
    );
    let mut output = stretcher.pull(usize::MAX);
    if finish {
        output.extend(stretcher.flush());
    }
    output
        .into_iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
        .collect()
}

pub fn synthesize_sentence(
    sentence: &str,
    model_dir: &Path,
    stop: &AtomicBool,
    tts_engine: TtsEngine,
) -> Result<Arc<TtsAudio>, String> {
    let text = sentence.trim().replace('\0', "");
    if text.is_empty() {
        return Err("There is no text to read".to_owned());
    }
    if stop.load(Ordering::Relaxed) {
        return Err("TTS stopped".to_owned());
    }
    let key = format!("{}:{text}", tts_engine.cache_name());
    let cache = AUDIO_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(audio) = cache.lock().map_err(|error| error.to_string())?.get(&key) {
        return Ok(Arc::clone(audio));
    }
    let result = generate_sentence(&text, model_dir, stop, tts_engine)?;
    cache
        .lock()
        .map_err(|error| error.to_string())?
        .insert(key, Arc::clone(&result));
    Ok(result)
}

fn generate_sentence(
    sentence: &str,
    model_dir: &Path,
    stop: &AtomicBool,
    tts_engine: TtsEngine,
) -> Result<Arc<TtsAudio>, String> {
    let text = sentence.trim().replace('\0', "");
    if text.is_empty() {
        return Err("There is no text to read".to_owned());
    }
    if stop.load(Ordering::Relaxed) {
        return Err("TTS stopped".to_owned());
    }
    if tts_engine == TtsEngine::Kokoro {
        return crate::kokoro_tts::synthesize(&text, model_dir);
    }
    let mut engine = ENGINE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|error| error.to_string())?;
    if engine.is_none() {
        let path = |name: &str| model_dir.join(name).to_string_lossy().into_owned();
        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(path("melo-model.onnx")),
                    lexicon: Some(path("melo-lexicon.txt")),
                    tokens: Some(path("melo-tokens.txt")),
                    ..Default::default()
                },
                num_threads: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        *engine = Some(OfflineTts::create(&config).ok_or("Melo TTS could not load model files")?);
    }
    let engine = engine.as_mut().ok_or("Melo TTS did not initialize")?;
    let spoken = speakable_numbers(&text);
    let audio = engine
        .generate_with_config(
            &spoken,
            &GenerationConfig {
                speed: 1.0,
                ..Default::default()
            },
            None::<fn(&[f32], f32) -> bool>,
        )
        .ok_or("Melo TTS failed to generate audio")?;
    let pcm = audio
        .samples()
        .iter()
        .map(|sample| (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
        .collect::<Vec<_>>();
    if pcm.is_empty() {
        return Err("Melo TTS generated no audio".to_owned());
    }
    let result = Arc::new(TtsAudio {
        samples: pcm,
        sample_rate: engine.sample_rate(),
    });
    Ok(result)
}

pub fn synthesize_resilient(
    sentence: &str,
    model_dir: &Path,
    stop: &AtomicBool,
    tts_engine: TtsEngine,
) -> Result<Vec<Arc<TtsAudio>>, String> {
    synthesize_resilient_with(sentence, |part| {
        synthesize_sentence(part, model_dir, stop, tts_engine)
    })
}

pub fn synthesize_resilient_cached(
    sentence: &str,
    model_dir: &Path,
    document_dir: &Path,
    stop: &AtomicBool,
    tts_engine: TtsEngine,
) -> Result<Vec<Arc<TtsAudio>>, String> {
    synthesize_resilient_cached_with(sentence, document_dir, |part| {
        generate_sentence(part, model_dir, stop, tts_engine)
    })
}

fn synthesize_resilient_cached_with(
    sentence: &str,
    document_dir: &Path,
    mut generate: impl FnMut(&str) -> Result<Arc<TtsAudio>, String>,
) -> Result<Vec<Arc<TtsAudio>>, String> {
    let audio = cached_audio(document_dir, sentence, || {
        let parts = synthesize_resilient_with(sentence, &mut generate)?;
        let sample_rate = parts[0].sample_rate;
        let mut samples = Vec::new();
        for part in parts {
            if part.sample_rate != sample_rate {
                return Err("TTS sample rates differ".to_owned());
            }
            samples.extend_from_slice(&part.samples);
        }
        Ok(Arc::new(TtsAudio {
            samples,
            sample_rate,
        }))
    })?;
    Ok(vec![audio])
}

fn synthesize_resilient_with<T>(
    sentence: &str,
    mut synthesize: impl FnMut(&str) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    match synthesize(sentence) {
        Ok(audio) => Ok(vec![audio]),
        Err(error) => {
            if error != "Melo TTS failed to generate audio" {
                return Err(error);
            }
            let mut parts = sentence
                .split(|ch: char| !ch.is_alphanumeric())
                .filter(|part| !part.is_empty())
                .flat_map(|part| {
                    part.chars()
                        .collect::<Vec<_>>()
                        .chunks(24)
                        .map(|chars| chars.iter().collect::<String>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            if parts.len() == 1 && parts[0] == sentence {
                let chars = sentence.chars().collect::<Vec<_>>();
                if chars.len() == 1 {
                    return Err(error);
                }
                let middle = chars.len() / 2;
                parts = vec![
                    chars[..middle].iter().collect(),
                    chars[middle..].iter().collect(),
                ];
            }
            let audio = parts
                .iter()
                .filter_map(|part| synthesize(part).ok())
                .collect::<Vec<_>>();
            if audio.is_empty() {
                Err(error)
            } else {
                Ok(audio)
            }
        }
    }
}

fn speakable_numbers(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if !ch.is_ascii_digit() {
            result.push(ch);
            continue;
        }
        let mut digits = ch.to_string();
        while chars.peek().is_some_and(char::is_ascii_digit) {
            digits.push(chars.next().unwrap());
        }
        result.push_str(&chinese_number(&digits));
        let mut following = chars.clone();
        if following.next() == Some(':') && following.peek().is_some_and(char::is_ascii_digit) {
            chars.next();
            let mut minute = String::new();
            while chars.peek().is_some_and(char::is_ascii_digit) {
                minute.push(chars.next().unwrap());
            }
            result.push('点');
            result.push_str(&chinese_number(&minute));
            result.push('分');
        }
    }
    result
}

fn chinese_number(digits: &str) -> String {
    const CHINESE: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    if let Ok(number) = digits.parse::<usize>() {
        if number < 10 {
            return CHINESE[number].to_string();
        }
        if number < 100 {
            let tens = number / 10;
            let units = number % 10;
            let mut result = String::new();
            if tens != 1 {
                result.push(CHINESE[tens]);
            }
            result.push('十');
            if units != 0 {
                result.push(CHINESE[units]);
            }
            return result;
        }
    }
    digits
        .bytes()
        .map(|digit| CHINESE[usize::from(digit - b'0')])
        .collect()
}

pub fn sentences(markdown: &str) -> Vec<String> {
    sentence_chunks(&markdown_text(markdown))
}

fn sentence_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut count = 0;
    let lines = text.split('\n').collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        let mut characters = line.trim().chars().peekable();
        while let Some(character) = characters.next() {
            chunk.push(character);
            count += 1;
            let sentence_end = matches!(character, '。' | '！' | '？' | '!' | '?' | '；' | ';')
                || (character == '.' && characters.peek().is_none_or(|next| next.is_whitespace()))
                || (matches!(character, '，' | ',') && count >= 30);
            if sentence_end || count >= 120 {
                push_sentence(&mut chunks, &mut chunk);
                count = 0;
            }
        }
        if index + 1 < lines.len() {
            // ponytail: Markdown has no page width; use OCR geometry if line boxes become available.
            let short_line =
                line.trim().chars().count() < 40 || lines[index + 1].trim().chars().count() < 40;
            if short_line || lines[index + 1].trim().is_empty() {
                push_sentence(&mut chunks, &mut chunk);
                count = 0;
            } else if !chunk.is_empty() {
                chunk.push(' ');
                count += 1;
            }
        }
    }
    push_sentence(&mut chunks, &mut chunk);
    chunks
}

fn push_sentence(chunks: &mut Vec<String>, chunk: &mut String) {
    if chunk
        .chars()
        .filter(|character| character.is_alphabetic())
        .count()
        >= 2
    {
        chunks.push(chunk.trim().to_owned());
    }
    chunk.clear();
}

pub fn markdown_text(markdown: &str) -> String {
    let mut text = String::new();
    let mut in_image = false;
    for event in pulldown_cmark::Parser::new(markdown) {
        match event {
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::Image { .. }) => in_image = true,
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Image) => in_image = false,
            pulldown_cmark::Event::Text(piece) | pulldown_cmark::Event::Code(piece)
                if !in_image =>
            {
                text.push_str(&piece);
            }
            pulldown_cmark::Event::SoftBreak | pulldown_cmark::Event::HardBreak if !in_image => {
                text.push('\n');
            }
            pulldown_cmark::Event::End(
                pulldown_cmark::TagEnd::Paragraph
                | pulldown_cmark::TagEnd::Heading(_)
                | pulldown_cmark::TagEnd::Item,
            ) if !text.ends_with('\n') => text.push_str("\n\n"),
            _ => {}
        }
    }
    text.trim().to_owned()
}

#[cfg(test)]
mod tests {
    #[test]
    fn fallback_audio_is_cached_as_a_complete_sentence() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-tts-fallback-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first = super::synthesize_resilient_cached_with("hello world", &root, |part| {
            if part == "hello world" {
                return Err("Melo TTS failed to generate audio".to_owned());
            }
            Ok(std::sync::Arc::new(super::TtsAudio {
                samples: vec![part.len() as i16],
                sample_rate: 44_100,
            }))
        })
        .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].samples, [5, 5]);
        let second = super::synthesize_resilient_cached_with("hello world", &root, |_| {
            panic!("fallback audio must not be regenerated")
        })
        .unwrap();
        assert_eq!(second[0].samples, [5, 5]);
        std::fs::remove_file(super::cache_path(&root, "hello world")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn document_audio_is_reused_until_regenerated() {
        let root = std::env::temp_dir().join(format!(
            "bibiocr-tts-cache-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let document = root.join("7").join("0");
        let original = super::cached_audio(&document, "你好。", || {
            Ok(std::sync::Arc::new(super::TtsAudio {
                samples: vec![1, 2, 3],
                sample_rate: 44_100,
            }))
        })
        .unwrap();
        drop(original);
        let reused = super::cached_audio(&document, "你好。", || {
            panic!("cached audio must not be generated again")
        })
        .unwrap();
        assert_eq!(reused.samples, [1, 2, 3]);
        let next_revision = root.join("7").join("1");
        let regenerated = super::cached_audio(&next_revision, "你好。", || {
            Ok(std::sync::Arc::new(super::TtsAudio {
                samples: vec![4, 5],
                sample_rate: 44_100,
            }))
        })
        .unwrap();
        assert_eq!(regenerated.samples, [4, 5]);
        assert_eq!(
            super::cached_audio(&document, "你好。", || panic!("old cache deleted"))
                .unwrap()
                .samples,
            [1, 2, 3]
        );
        std::fs::remove_file(super::cache_path(&document, "你好。")).unwrap();
        std::fs::remove_file(super::cache_path(&next_revision, "你好。")).unwrap();
        std::fs::remove_dir(next_revision).unwrap();
        std::fs::remove_dir(document).unwrap();
        std::fs::remove_dir(root.join("7")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn wsola_changes_tempo_mid_stream_without_changing_sample_rate() {
        let input: Vec<i16> = (0..44_100)
            .map(|i| ((i as f32 * 0.07).sin() * 10_000.0) as i16)
            .collect();
        let mut stretcher = wsola::TimeStretch::new(22_050, 1).unwrap();
        let mut output = Vec::new();
        for (index, chunk) in input.chunks(2_205).enumerate() {
            let speed = if index < 10 { 1.0 } else { 2.0 };
            output.extend(super::stretch_pcm_chunk(
                &mut stretcher,
                chunk,
                speed,
                false,
            ));
        }
        output.extend(super::stretch_pcm_chunk(&mut stretcher, &[], 2.0, true));
        assert!((28_000..38_000).contains(&output.len()));
        assert!(output.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn failed_slice_does_not_silence_readable_parts() {
        let mut attempted = Vec::new();
        let audio = super::synthesize_resilient_with("你好 @ 世界", |part| {
            attempted.push(part.to_owned());
            if part.contains('@') {
                Err("Melo TTS failed to generate audio".to_owned())
            } else {
                Ok(part.to_owned())
            }
        });
        assert_eq!(audio.unwrap(), ["你好", "世界"]);
        assert_eq!(attempted, ["你好 @ 世界", "你好", "世界"]);
        let unspaced = super::synthesize_resilient_with("你好世界", |part| {
            if part == "你好世界" {
                Err("Melo TTS failed to generate audio".to_owned())
            } else {
                Ok(part.to_owned())
            }
        });
        assert_eq!(unspaced.unwrap(), ["你好", "世界"]);
        assert!(
            super::synthesize_resilient_with("@@@", |_| -> Result<(), String> {
                Err("Melo TTS failed to generate audio".to_owned())
            })
            .is_err()
        );
    }

    #[test]
    #[ignore = "loads the downloaded Melo TTS model"]
    fn real_melo_model_generates_bilingual_audio() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("downloads");
        let audio = super::synthesize_sentence(
            "你好, hello world!",
            &model_dir,
            &std::sync::atomic::AtomicBool::new(false),
            crate::settings::TtsEngine::Melo,
        )
        .unwrap();
        assert_eq!(audio.sample_rate, 44_100);
        assert!(audio.samples.len() > 4_410);
        assert!(audio.samples.iter().any(|sample| *sample != 0));
        let reused = super::synthesize_sentence(
            "你好, hello world!",
            &model_dir,
            &std::sync::atomic::AtomicBool::new(false),
            crate::settings::TtsEngine::Melo,
        )
        .unwrap();
        assert!(std::sync::Arc::ptr_eq(&audio, &reused));
        let clock = super::synthesize_sentence(
            "13:37",
            &model_dir,
            &std::sync::atomic::AtomicBool::new(false),
            crate::settings::TtsEngine::Melo,
        )
        .unwrap();
        assert!(clock.samples.len() > 4_410);
    }

    #[test]
    fn markdown_reader_uses_document_text_not_markup() {
        assert_eq!(
            super::markdown_text("# Hello\n\n**world** [link](https://example.com)"),
            "Hello\n\nworld link"
        );
        assert_eq!(
            super::markdown_text("前文\n\n![Image](imgs/crop.jpg)\n\n后文"),
            "前文\n\n后文"
        );
    }

    #[test]
    fn unspaced_chinese_is_split_without_losing_text() {
        let text = "中文".repeat(100);
        let chunks = super::sentence_chunks(&text);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn reading_splits_at_sentence_boundaries_without_splitting_domains() {
        assert_eq!(
            super::sentence_chunks("你好。 Hello world. Visit example.com now! 再见。"),
            ["你好。", "Hello world.", "Visit example.com now!", "再见。"]
        );
    }

    #[test]
    fn short_markdown_lines_become_sentences_but_long_wrapped_lines_continue() {
        assert_eq!(super::sentences("短行\n下一句。"), ["短行", "下一句。"]);
        let long_line = "中文".repeat(25);
        let next_long_line = "继续".repeat(25);
        assert_eq!(
            super::sentences(&format!("{long_line}\n{next_long_line}。")),
            [format!("{long_line} {next_long_line}。")]
        );
        assert_eq!(
            super::sentences(&format!("{long_line}\n继续。")),
            [long_line.clone(), "继续。".to_owned()]
        );
    }

    #[test]
    fn unreadable_leading_ocr_noise_does_not_block_later_sentences() {
        assert_eq!(
            super::sentences("O\n0\n△○□\n安全提示\n后面的文字可以朗读。"),
            ["安全提示", "后面的文字可以朗读。"]
        );
    }

    #[test]
    fn long_clause_uses_comma_as_a_speech_boundary() {
        let first = "中".repeat(35);
        assert_eq!(
            super::sentence_chunks(&format!("{first}，后半句。")),
            [format!("{first}，"), "后半句。".to_owned()]
        );
    }
}
