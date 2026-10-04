use crate::tts::TtsAudio;
use ort::{session::Session, value::Tensor};
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

struct KokoroRuntime {
    session: Session,
    voice: Vec<f32>,
    vocab: HashMap<char, i64>,
}

static ENGINE: OnceLock<Mutex<Option<KokoroRuntime>>> = OnceLock::new();

pub fn synthesize(text: &str, model_dir: &Path) -> Result<Arc<TtsAudio>, String> {
    if text.chars().any(|ch| ch.is_alphabetic() && !ch.is_ascii()) {
        return Err("Kokoro 1.0 supports English text only; select Melo for Chinese".to_owned());
    }
    let phonemes = voice_g2p::english_to_phonemes(text).map_err(|error| error.to_string())?;
    let mut engine = ENGINE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|error| error.to_string())?;
    if engine.is_none() {
        *engine = Some(KokoroRuntime::load(model_dir)?);
    }
    let mut samples = Vec::new();
    for chunk in phonemes.chars().collect::<Vec<_>>().chunks(500) {
        let audio = engine
            .as_mut()
            .unwrap()
            .synthesize(&chunk.iter().collect::<String>())?;
        samples.extend_from_slice(&audio.samples);
    }
    if samples.is_empty() {
        return Err("Kokoro found no readable phonemes".to_owned());
    }
    Ok(Arc::new(TtsAudio {
        samples,
        sample_rate: 24_000,
    }))
}

impl KokoroRuntime {
    fn load(model_dir: &Path) -> Result<Self, String> {
        let model = model_dir.join("kokoro-model.onnx");
        let session = Session::builder()
            .map_err(|error| error.to_string())?
            .with_intra_threads(2)
            .map_err(|error| error.to_string())?
            .commit_from_file(&model)
            .map_err(|error| format!("Kokoro model load failed: {error}"))?;
        let voice_bytes =
            fs::read(model_dir.join("af_heart.bin")).map_err(|error| error.to_string())?;
        if voice_bytes.len() < 256 * 4 || voice_bytes.len() % (256 * 4) != 0 {
            return Err("Kokoro voice file has an invalid length".to_owned());
        }
        let voice = voice_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let tokenizer =
            fs::read(model_dir.join("kokoro-tokenizer.json")).map_err(|error| error.to_string())?;
        let parsed: serde_json::Value =
            serde_json::from_slice(&tokenizer).map_err(|error| error.to_string())?;
        let vocab = parsed["model"]["vocab"]
            .as_object()
            .ok_or("Kokoro tokenizer has no vocabulary")?
            .iter()
            .filter_map(|(phoneme, id)| Some((phoneme.chars().next()?, id.as_i64()?)))
            .collect();
        Ok(Self {
            session,
            voice,
            vocab,
        })
    }

    fn synthesize(&mut self, phonemes: &str) -> Result<Arc<TtsAudio>, String> {
        let ids = token_ids(phonemes, &self.vocab);
        if ids.len() <= 2 {
            return Err("Kokoro found no readable phonemes".to_owned());
        }
        let style_index = (ids.len() - 2).min(self.voice.len() / 256 - 1);
        let style = self.voice[style_index * 256..(style_index + 1) * 256].to_vec();
        let ids_tensor =
            Tensor::from_array(([1_usize, ids.len()], ids)).map_err(|error| error.to_string())?;
        let style_tensor =
            Tensor::from_array(([1_usize, 256], style)).map_err(|error| error.to_string())?;
        let speed_tensor =
            Tensor::from_array(([1_usize], vec![1.0_f32])).map_err(|error| error.to_string())?;
        let output = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "style" => style_tensor,
                "speed" => speed_tensor,
            ])
            .map_err(|error| format!("Kokoro inference failed: {error}"))?;
        let (_, samples) = output[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| error.to_string())?;
        if samples.is_empty() {
            return Err("Kokoro generated no audio".to_owned());
        }
        Ok(Arc::new(TtsAudio {
            samples: samples
                .iter()
                .map(|sample| (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
                .collect(),
            sample_rate: 24_000,
        }))
    }
}

fn token_ids(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
    let mut ids = Vec::with_capacity(512);
    ids.push(0);
    ids.extend(phonemes.chars().filter_map(|ch| vocab.get(&ch).copied()));
    ids.push(0);
    ids
}

#[cfg(test)]
mod tests {
    use super::token_ids;
    use std::collections::HashMap;

    #[test]
    fn kokoro_tokens_are_padded_without_silently_dropping_text() {
        let vocab = HashMap::from([('h', 50), ('ə', 83)]);
        assert_eq!(token_ids("hə?", &vocab), [0, 50, 83, 0]);
        assert_eq!(token_ids(&"h".repeat(600), &vocab).len(), 602);
    }

    #[test]
    #[ignore = "downloads the Kokoro model and runs real ONNX inference once"]
    fn real_kokoro_model_generates_english_audio() {
        let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("downloads");
        let task = crate::download::start(
            models.clone(),
            Default::default(),
            "zh-CN",
            crate::settings::OcrEngine::PaddleV6,
            crate::settings::TtsEngine::Kokoro,
        );
        for event in task.events {
            match event {
                crate::download::DownloadEvent::Failed(_, error) => panic!("{error}"),
                crate::download::DownloadEvent::Finished => break,
                _ => {}
            }
        }
        let audio = super::synthesize("Hello world.", &models).unwrap();
        assert_eq!(audio.sample_rate, 24_000);
        assert!(audio.samples.len() > 2_400);
        assert!(audio.samples.iter().any(|sample| *sample != 0));
    }
}
