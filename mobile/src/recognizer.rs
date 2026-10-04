use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker},
    sampling::LlamaSampler,
};
use std::{
    ffi::CString,
    num::NonZeroU32,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub fn recognize(
    model_path: &Path,
    mmproj_path: &Path,
    image_path: &Path,
) -> Result<String, String> {
    recognize_stream(model_path, mmproj_path, image_path, |_| {})
}

pub fn recognize_stream(
    model_path: &Path,
    mmproj_path: &Path,
    image_path: &Path,
    on_piece: impl FnMut(&str),
) -> Result<String, String> {
    recognize_stream_cancellable(
        model_path,
        mmproj_path,
        image_path,
        &AtomicBool::new(false),
        on_piece,
    )
}

pub fn recognize_stream_cancellable(
    model_path: &Path,
    mmproj_path: &Path,
    image_path: &Path,
    cancel: &AtomicBool,
    mut on_piece: impl FnMut(&str),
) -> Result<String, String> {
    let _backend_guard = crate::backend_gate::lock()?;
    if cancel.load(Ordering::Relaxed) {
        return Err("Recognition cancelled".to_owned());
    }
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get) as i32;
    let backend = LlamaBackend::init().map_err(|error| error.to_string())?;
    let model = LlamaModel::load_from_file(&backend, model_path, &LlamaModelParams::default())
        .map_err(|error| error.to_string())?;
    let mut context = model
        .new_context(
            &backend,
            LlamaContextParams::default()
                .with_n_threads(threads)
                .with_n_batch(512)
                .with_n_ctx(NonZeroU32::new(8192)),
        )
        .map_err(|error| error.to_string())?;
    let marker = mtmd_default_marker();
    let mtmd = MtmdContext::init_from_file(
        mmproj_path.to_str().ok_or("MMProj path is not UTF-8")?,
        &model,
        &MtmdContextParams {
            use_gpu: false,
            print_timings: false,
            n_threads: threads,
            media_marker: CString::new(marker).map_err(|error| error.to_string())?,
            image_min_tokens: -1,
            image_max_tokens: -1,
        },
    )
    .map_err(|error| error.to_string())?;
    let bitmap = MtmdBitmap::from_file(
        &mtmd,
        image_path.to_str().ok_or("image path is not UTF-8")?,
        false,
    )
    .map_err(|error| error.to_string())?;
    // This model's GGUF chat template is Jinja; llama_chat_apply_template
    // rejects it with FFI error -1. Use its single-image OCR form directly.
    let prompt = format!("<|begin_of_sentence|>User: {marker}OCR:\nAssistant:\n");
    let chunks = mtmd
        .tokenize(
            MtmdInputText {
                text: prompt,
                add_special: false,
                parse_special: true,
            },
            &[&bitmap],
        )
        .map_err(|error| error.to_string())?;
    let mut n_past = chunks
        .eval_chunks(&mtmd, &context, 0, 0, 512, true)
        .map_err(|error| error.to_string())?;
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut batch = LlamaBatch::new(1, 1);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    for _ in 0..4096 {
        if cancel.load(Ordering::Relaxed) {
            return Err("Recognition cancelled".to_owned());
        }
        let token = sampler.sample(&context, -1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|error| error.to_string())?;
        output.push_str(&piece);
        on_piece(&piece);
        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|error| error.to_string())?;
        n_past += 1;
        context
            .decode(&mut batch)
            .map_err(|error| error.to_string())?;
    }
    Ok(output.trim().to_owned())
}
