use crate::{dependencies::RuntimeConfig, export, model_runtimes};
use eframe::egui::{self, Color32, ProgressBar, RichText};
use pdf_inspector::vision::{PdfiumRenderer, RenderOptions, RenderPixelFormat};
use std::{
    collections::HashMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

enum BatchEvent {
    Started(usize),
    Processing(usize, PathBuf),
    Completed(PathBuf),
    Failed(PathBuf, String),
    Finished { succeeded: usize, failed: usize },
}

#[derive(Default)]
pub struct BatchPanel {
    input_dir: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    worker: Option<Receiver<BatchEvent>>,
    total: usize,
    processed: usize,
    current: String,
    log: Vec<String>,
}

impl BatchPanel {
    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some(receiver) = self.worker.take() else {
            return;
        };
        let mut keep = true;
        loop {
            match receiver.try_recv() {
                Ok(BatchEvent::Started(total)) => self.total = total,
                Ok(BatchEvent::Processing(index, path)) => {
                    self.processed = index;
                    self.current = path.display().to_string();
                }
                Ok(BatchEvent::Completed(path)) => {
                    self.processed += 1;
                    self.log.push(format!("✓ {}", path.display()));
                }
                Ok(BatchEvent::Failed(path, error)) => {
                    self.processed += 1;
                    self.log.push(format!("✗ {}: {error}", path.display()));
                }
                Ok(BatchEvent::Finished { succeeded, failed }) => {
                    self.current =
                        rust_i18n::t!("batch_finished", succeeded = succeeded, failed = failed)
                            .into_owned();
                    keep = false;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.log
                        .push(rust_i18n::t!("batch_disconnected").into_owned());
                    keep = false;
                    break;
                }
            }
        }
        if keep {
            self.worker = Some(receiver);
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &RuntimeConfig) {
        let busy = self.worker.is_some();
        ui.label(rust_i18n::t!("batch_intro"));
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(rust_i18n::t!("batch_input"));
            ui.label(
                self.input_dir
                    .as_deref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "—".to_owned()),
            );
            if ui
                .add_enabled(!busy, egui::Button::new(rust_i18n::t!("browse_folder")))
                .clicked()
                && let Some(path) = rfd::FileDialog::new().pick_folder()
            {
                self.input_dir = Some(path);
            }
        });
        ui.horizontal(|ui| {
            ui.label(rust_i18n::t!("batch_output"));
            ui.label(
                self.output_dir
                    .as_deref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "—".to_owned()),
            );
            if ui
                .add_enabled(!busy, egui::Button::new(rust_i18n::t!("browse_folder")))
                .clicked()
                && let Some(path) = rfd::FileDialog::new().pick_folder()
            {
                self.output_dir = Some(path);
            }
        });
        ui.add_space(8.0);
        if ui
            .add_enabled(
                !busy && self.input_dir.is_some() && self.output_dir.is_some(),
                egui::Button::new(rust_i18n::t!("start_batch")),
            )
            .clicked()
        {
            self.start(config.clone());
        }
        let ratio = if self.total == 0 {
            0.0
        } else {
            self.processed as f32 / self.total as f32
        };
        ui.add(ProgressBar::new(ratio).show_percentage());
        if !self.current.is_empty() {
            ui.label(&self.current);
        }
        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for line in &self.log {
                ui.label(RichText::new(line).color(if line.starts_with('✗') {
                    Color32::DARK_RED
                } else {
                    Color32::DARK_GREEN
                }));
            }
        });
    }

    fn start(&mut self, config: RuntimeConfig) {
        let input = self
            .input_dir
            .clone()
            .expect("button validates input folder");
        let output = self
            .output_dir
            .clone()
            .expect("button validates output folder");
        self.total = 0;
        self.processed = 0;
        self.current.clear();
        self.log.clear();
        self.worker = Some(start_worker(input, output, config));
    }
}

fn start_worker(input: PathBuf, output: PathBuf, config: RuntimeConfig) -> Receiver<BatchEvent> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let result = run_batch(&input, &output, &config, &sender);
        if let Err(error) = result {
            let _ = sender.send(BatchEvent::Failed(input, error));
            let _ = sender.send(BatchEvent::Finished {
                succeeded: 0,
                failed: 1,
            });
        }
    });
    receiver
}

fn run_batch(
    input: &Path,
    output: &Path,
    config: &RuntimeConfig,
    sender: &std::sync::mpsc::Sender<BatchEvent>,
) -> Result<(), String> {
    if !input.is_dir() {
        return Err(format!("Input folder does not exist: {}", input.display()));
    }
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    if input.canonicalize().ok() == output.canonicalize().ok() {
        return Err("Input and output folders must be different".to_owned());
    }
    if !config.tools.pandoc.is_file() {
        return Err(format!(
            "Pandoc not found: {}",
            config.tools.pandoc.display()
        ));
    }
    let mut files: Vec<_> = fs::read_dir(input)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && supported(path))
        .collect();
    files.sort_by_key(|path| path.file_name().map(OsString::from));
    let _ = sender.send(BatchEvent::Started(files.len()));
    let mut succeeded = 0;
    let mut failed = 0;
    for (index, path) in files.into_iter().enumerate() {
        let _ = sender.send(BatchEvent::Processing(index, path.clone()));
        match process_file(&path, output, config) {
            Ok(docx) => {
                succeeded += 1;
                let _ = sender.send(BatchEvent::Completed(docx));
            }
            Err(error) => {
                failed += 1;
                let _ = sender.send(BatchEvent::Failed(path, error));
            }
        }
    }
    let _ = sender.send(BatchEvent::Finished { succeeded, failed });
    Ok(())
}

fn supported(path: &Path) -> bool {
    is_image(path) || anydoc::Format::from_path(path).is_some()
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "gif" | "tif" | "tiff" | "webp"
            )
        })
}

fn process_file(path: &Path, output_dir: &Path, config: &RuntimeConfig) -> Result<PathBuf, String> {
    let (markdown, search_dirs) = if is_image(path) {
        let result = model_runtimes::process_image(path)?;
        (result.markdown, vec![result.output_dir])
    } else {
        match anydoc::to_markdown(path) {
            Ok(markdown) => (
                markdown,
                vec![path.parent().unwrap_or(Path::new(".")).to_path_buf()],
            ),
            Err(anydoc::ConvertError::NeedsOcr { pages, page_count }) => {
                process_pdf_with_ocr(path, &pages, page_count, config)?
            }
            Err(error) => return Err(format!("AnyDoc: {error}")),
        }
    };
    let markdown_path = output_dir.join(appended_name(path, ".md")?);
    let docx_path = output_dir.join(appended_name(path, ".docx")?);
    export::save_markdown(&markdown_path, &markdown, &search_dirs)?;
    let saved_markdown = fs::read_to_string(&markdown_path).map_err(|error| error.to_string())?;
    export::save_docx(
        &docx_path,
        &saved_markdown,
        &[output_dir.to_path_buf()],
        &config.tools.pandoc,
    )?;
    Ok(docx_path)
}

fn process_pdf_with_ocr(
    path: &Path,
    ocr_pages: &[u32],
    page_count: u32,
    config: &RuntimeConfig,
) -> Result<(String, Vec<PathBuf>), String> {
    if !config.dependencies.pdfium.is_file() {
        return Err(format!(
            "PDFium not found: {}",
            config.dependencies.pdfium.display()
        ));
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let extracted = pdf_inspector::extract_pages_markdown_mem(&bytes, None)
        .map_err(|error| format!("PDF extraction: {error}"))?;
    let native: HashMap<_, _> = extracted
        .pages
        .into_iter()
        .map(|page| (page.page + 1, page.markdown))
        .collect();
    let renderer = PdfiumRenderer::load_from_path(&config.dependencies.pdfium)
        .map_err(|error| format!("PDFium: {error}"))?;
    let rendered = renderer
        .render_pages(&bytes, ocr_pages, None, &RenderOptions::new())
        .map_err(|error| format!("PDF render: {error}"))?;
    let temp = temp_batch_dir();
    fs::create_dir_all(&temp).map_err(|error| error.to_string())?;
    let mut recognized = HashMap::new();
    let mut search_dirs = Vec::new();
    for page in rendered {
        if page.format() != RenderPixelFormat::Rgb8 {
            return Err("PDFium returned an unsupported pixel format".to_owned());
        }
        let row_bytes = page.width() as usize * 3;
        let pixels = if page.stride() == row_bytes {
            page.pixels().to_vec()
        } else {
            page.pixels()
                .chunks(page.stride())
                .flat_map(|row| row[..row_bytes].iter().copied())
                .collect()
        };
        let image = image::RgbImage::from_raw(page.width(), page.height(), pixels)
            .ok_or_else(|| "Invalid PDF page bitmap".to_owned())?;
        let image_path = temp.join(format!("page-{}.png", page.page()));
        image.save(&image_path).map_err(|error| error.to_string())?;
        let result = model_runtimes::process_image(&image_path)?;
        recognized.insert(page.page(), result.markdown);
        search_dirs.push(result.output_dir);
        let _ = fs::remove_file(image_path);
    }
    let _ = fs::remove_dir(temp);
    let mut markdown = String::new();
    for page in 1..=page_count {
        markdown.push_str(&format!("<!-- Page {page} -->\n\n"));
        if let Some(text) = recognized.get(&page).or_else(|| native.get(&page)) {
            markdown.push_str(text.trim());
        }
        markdown.push_str("\n\n");
    }
    Ok((markdown, search_dirs))
}

fn appended_name(path: &Path, suffix: &str) -> Result<OsString, String> {
    let mut name = path
        .file_name()
        .ok_or_else(|| format!("File has no name: {}", path.display()))?
        .to_os_string();
    name.push(suffix);
    Ok(name)
}

fn temp_batch_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir()
        .join("bibiocr")
        .join(format!("batch-{stamp}"))
}

#[cfg(test)]
mod tests {
    use super::{appended_name, supported};
    use std::{ffi::OsStr, path::Path};

    #[test]
    fn batch_names_append_without_replacing_original_extension() {
        assert_eq!(
            appended_name(Path::new("report.pdf"), ".md").unwrap(),
            OsStr::new("report.pdf.md")
        );
        assert_eq!(
            appended_name(Path::new("report.pdf"), ".docx").unwrap(),
            OsStr::new("report.pdf.docx")
        );
        assert!(supported(Path::new("report.xlsx")));
        assert!(supported(Path::new("scan.png")));
        assert!(!supported(Path::new("notes.txt")));
    }
}
