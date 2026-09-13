use crate::{
    batch::BatchPanel,
    dependencies::{DependencyKey, RuntimeConfig},
    download::{self, DownloadEvent, DownloadOptions},
    export, html_preview,
    locale::{LocaleManager, SUPPORTED_LOCALES},
    model_runtimes::{self, BackendEvent},
    settings::{AppSettings, ProxyMode},
    yaml_panel::YamlPanel,
};
use arboard::Clipboard;
use eframe::egui::{
    self, Align, Color32, FontData, FontDefinitions, FontFamily, Layout, Margin, Pos2, ProgressBar,
    Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, Vec2,
};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use image::RgbaImage;
use std::{
    any::Any,
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::mpsc::{Receiver, TryRecvError},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const GREEN: Color32 = Color32::from_rgb(0, 173, 79);
const BLUE: Color32 = Color32::from_rgb(0, 116, 188);
const WORKSPACE_BG: Color32 = Color32::from_rgb(238, 241, 245);
const MAX_IMAGE_PIXELS: usize = 64 * 1024 * 1024;

struct LoadedImage {
    size: [usize; 2],
    texture: TextureHandle,
}

struct ImageViewState {
    zoom: f32,
    pan: Vec2,
}

impl Default for ImageViewState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
        }
    }
}

impl ImageViewState {
    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum WorkspaceTab {
    Original,
    Layout,
    Markdown,
    Batch,
    Yaml,
}

impl WorkspaceTab {
    fn title(self) -> String {
        match self {
            Self::Original => rust_i18n::t!("panel_original").into_owned(),
            Self::Layout => rust_i18n::t!("panel_layout").into_owned(),
            Self::Markdown => rust_i18n::t!("panel_markdown").into_owned(),
            Self::Batch => rust_i18n::t!("panel_batch").into_owned(),
            Self::Yaml => rust_i18n::t!("panel_yaml").into_owned(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MarkdownTab {
    Edit,
    Preview,
}

struct Workspace {
    original: Option<LoadedImage>,
    layout: Option<LoadedImage>,
    original_view: ImageViewState,
    layout_view: ImageViewState,
    markdown: String,
    markdown_tab: MarkdownTab,
    markdown_cache: CommonMarkCache,
    output_dir: Option<PathBuf>,
    input_path: Option<PathBuf>,
    batch: BatchPanel,
    yaml: YamlPanel,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            original: None,
            layout: None,
            original_view: ImageViewState::default(),
            layout_view: ImageViewState::default(),
            markdown: rust_i18n::t!("initial_markdown").into_owned(),
            markdown_tab: MarkdownTab::Edit,
            markdown_cache: CommonMarkCache::default(),
            output_dir: None,
            input_path: None,
            batch: BatchPanel::default(),
            yaml: YamlPanel::default(),
        }
    }
}

pub struct BibiOcrApp {
    workspace: Workspace,
    dock_state: DockState<WorkspaceTab>,
    maximized: Option<WorkspaceTab>,
    pipeline: Option<Receiver<BackendEvent>>,
    downloads: Option<Receiver<DownloadEvent>>,
    download_current: Option<DependencyKey>,
    download_progress: (u64, Option<u64>),
    progress: f32,
    status: String,
    backend_available: bool,
    about_open: bool,
    downloads_open: bool,
    failure_dialog: Option<String>,
    settings: AppSettings,
    runtime_config: RuntimeConfig,
    locale: LocaleManager,
}

impl BibiOcrApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let settings = AppSettings::load();
        let locale = LocaleManager::from_settings(&settings);
        egui_extras::install_image_loaders(&cc.egui_ctx);
        configure_style(&cc.egui_ctx);
        install_cjk_font(&cc.egui_ctx);
        let availability = model_runtimes::availability();
        let runtime_config = RuntimeConfig::load();
        let downloads_open = !runtime_config.missing().is_empty();
        Self {
            workspace: Workspace::default(),
            dock_state: default_dock_state(),
            maximized: None,
            pipeline: None,
            downloads: None,
            download_current: None,
            download_progress: (0, None),
            progress: 0.0,
            status: availability
                .as_ref()
                .map(|_| rust_i18n::t!("ready").into_owned())
                .unwrap_or_else(|error| error.clone()),
            backend_available: availability.is_ok(),
            about_open: false,
            downloads_open,
            failure_dialog: None,
            settings,
            runtime_config,
            locale,
        }
    }

    fn read_image_file(&mut self, ctx: &egui::Context) {
        self.run_guarded(|app| app.try_read_image_file(ctx));
    }

    fn try_read_image_file(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let mut dialog = rfd::FileDialog::new().add_filter(
            rust_i18n::t!("images_filter").as_ref(),
            &["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp"],
        );
        if let Some(directory) = &self.settings.last_input_dir {
            dialog = dialog.set_directory(directory);
        }
        let Some(path) = dialog.pick_file() else {
            return Ok(());
        };
        self.try_load_and_process(ctx, path)
    }

    fn read_clipboard(&mut self, ctx: &egui::Context) {
        self.run_guarded(|app| app.try_read_clipboard(ctx));
    }

    fn try_read_clipboard(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let image = Clipboard::new()
            .and_then(|mut clipboard| clipboard.get_image())
            .map_err(|error| rust_i18n::t!("clipboard_empty", error = error).into_owned())?;
        let max_texture_side = ctx.input(|input| input.max_texture_side);
        let rgba = clipboard_rgba(
            image.width,
            image.height,
            image.bytes.into_owned(),
            max_texture_side,
        )?;
        let path = temporary_image_path();
        if let Some(parent) = path.parent()
            && let Err(error) = fs::create_dir_all(parent)
        {
            return Err(rust_i18n::t!("temp_dir_error", error = error).into_owned());
        }
        rgba.save(&path)
            .map_err(|error| rust_i18n::t!("clipboard_save_error", error = error).into_owned())?;
        self.set_original_texture(ctx, &rgba)?;
        self.set_input_path(&path);
        self.start_pipeline(path);
        Ok(())
    }

    fn load_and_process(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.run_guarded(|app| app.try_load_and_process(ctx, path));
    }

    fn try_load_and_process(&mut self, ctx: &egui::Context, path: PathBuf) -> Result<(), String> {
        let image = image::open(&path).map_err(|error| {
            rust_i18n::t!("read_error", path = path.display(), error = error).into_owned()
        })?;
        self.set_original_texture(ctx, &image.to_rgba8())?;
        self.set_input_path(&path);
        self.start_pipeline(path);
        Ok(())
    }

    fn set_original_texture(
        &mut self,
        ctx: &egui::Context,
        rgba: &RgbaImage,
    ) -> Result<(), String> {
        self.workspace.original = Some(texture_from_rgba(ctx, "original-image", rgba)?);
        self.workspace.layout = None;
        self.workspace.original_view.reset();
        self.workspace.layout_view.reset();
        self.workspace.output_dir = None;
        self.workspace.markdown = rust_i18n::t!("processing_markdown").into_owned();
        Ok(())
    }

    fn set_input_path(&mut self, path: &std::path::Path) {
        self.workspace.input_path = Some(path.to_path_buf());
        if let Some(parent) = path.parent() {
            self.settings.last_input_dir = Some(parent.to_path_buf());
            let _ = self.settings.save();
        }
    }

    fn start_pipeline(&mut self, path: PathBuf) {
        if self.pipeline.is_some() {
            self.fail(rust_i18n::t!("already_processing").into_owned());
            return;
        }
        self.progress = 0.02;
        self.status = rust_i18n::t!("starting").into_owned();
        self.pipeline = Some(model_runtimes::start_pipeline(path));
    }

    fn poll_pipeline(&mut self, ctx: &egui::Context) {
        let Some(receiver) = self.pipeline.take() else {
            return;
        };
        let mut keep = true;
        loop {
            let event = match receiver.try_recv() {
                Ok(event) => event,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.fail(rust_i18n::t!("backend_disconnected").into_owned());
                    keep = false;
                    break;
                }
            };
            let terminal = backend_event_is_terminal(&event);
            match event {
                BackendEvent::Progress(progress, status) => {
                    self.progress = progress;
                    self.status = status;
                }
                BackendEvent::Complete(output) => {
                    match image::open(&output.overlay_path) {
                        Ok(image) => {
                            match texture_from_rgba(ctx, "layout-image", &image.to_rgba8()) {
                                Ok(texture) => {
                                    self.workspace.layout = Some(texture);
                                    self.workspace.markdown = output.markdown;
                                    self.workspace.output_dir = Some(output.output_dir);
                                    self.workspace.markdown_tab = MarkdownTab::Preview;
                                    self.progress = 1.0;
                                    self.status = rust_i18n::t!("complete").into_owned();
                                }
                                Err(error) => self.fail(error),
                            }
                        }
                        Err(error) => self
                            .fail(rust_i18n::t!("layout_load_error", error = error).into_owned()),
                    }
                    keep = false;
                }
                BackendEvent::Failed(error) => {
                    self.fail(error);
                    keep = false;
                }
            }
            if terminal {
                break;
            }
        }
        if keep {
            self.pipeline = Some(receiver);
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn handle_drop(&mut self, ctx: &egui::Context) {
        let path = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .first()
                .map(|file| file.path().to_path_buf())
        });
        if let Some(path) = path {
            self.load_and_process(ctx, path);
        }
    }

    fn save_markdown(&mut self) {
        self.run_guarded(Self::try_save_markdown);
    }

    fn try_save_markdown(&mut self) -> Result<(), String> {
        let (directory, filename) = self.default_save_target("md");
        let Some(path) = rfd::FileDialog::new()
            .set_directory(directory)
            .set_file_name(filename)
            .add_filter("Markdown", &["md", "markdown"])
            .save_file()
        else {
            return Ok(());
        };
        let search_dirs = self.image_search_dirs();
        export::save_markdown(&path, &self.workspace.markdown, &search_dirs)?;
        self.status = rust_i18n::t!("saved", path = path.display()).into_owned();
        Ok(())
    }

    fn save_docx(&mut self) {
        self.run_guarded(Self::try_save_docx);
    }

    fn try_save_docx(&mut self) -> Result<(), String> {
        let (directory, filename) = self.default_save_target("docx");
        let Some(path) = rfd::FileDialog::new()
            .set_directory(directory)
            .set_file_name(filename)
            .add_filter(rust_i18n::t!("word_filter").as_ref(), &["docx"])
            .save_file()
        else {
            return Ok(());
        };
        let search_dirs = self.image_search_dirs();
        let pandoc = model_runtimes::pandoc_path()?;
        export::save_docx(&path, &self.workspace.markdown, &search_dirs, &pandoc)?;
        self.status = rust_i18n::t!("saved", path = path.display()).into_owned();
        Ok(())
    }

    fn run_guarded(&mut self, operation: impl FnOnce(&mut Self) -> Result<(), String>) {
        if let Err(error) = guard_operation(|| operation(self)) {
            self.fail(error);
        }
    }

    fn default_save_target(&self, extension: &str) -> (PathBuf, String) {
        save_target_for(
            self.workspace.input_path.as_deref(),
            self.settings.last_input_dir.as_deref(),
            extension,
        )
    }

    fn image_search_dirs(&self) -> Vec<PathBuf> {
        let mut directories = Vec::new();
        if let Some(output_dir) = &self.workspace.output_dir {
            directories.push(output_dir.clone());
        }
        if let Some(input_dir) = self
            .workspace
            .input_path
            .as_deref()
            .and_then(std::path::Path::parent)
            && !directories.contains(&input_dir.to_path_buf())
        {
            directories.push(input_dir.to_path_buf());
        }
        directories
    }

    fn fail(&mut self, error: String) {
        self.progress = 0.0;
        self.status = error.clone();
        self.failure_dialog = Some(error);
    }

    fn browse_dependency(&mut self, key: DependencyKey) {
        let mut dialog = rfd::FileDialog::new();
        if let Some(parent) = self.runtime_config.path(key).parent() {
            dialog = dialog.set_directory(parent);
        }
        if let Some(path) = dialog.pick_file() {
            self.runtime_config.set_path(key, path);
            if let Err(error) = self.runtime_config.save() {
                self.fail(error);
            } else {
                self.refresh_backend();
            }
        }
    }

    fn start_downloads(&mut self, keys: Vec<DependencyKey>) {
        if self.downloads.is_some() || keys.is_empty() {
            return;
        }
        if let Err(error) = self.settings.save() {
            self.fail(error);
            return;
        }
        self.downloads = Some(download::start(
            keys,
            DownloadOptions {
                proxy: self
                    .settings
                    .use_proxy
                    .then(|| self.settings.proxy.clone())
                    .unwrap_or_default(),
                no_proxy: self.settings.proxy_mode() == ProxyMode::None,
                hf_endpoint: if self.settings.use_hf_mirror {
                    self.settings.hf_endpoint.clone()
                } else {
                    "https://huggingface.co".to_owned()
                },
                github_proxy: self
                    .settings
                    .use_github_proxy
                    .then(|| self.settings.github_proxy.clone())
                    .unwrap_or_default(),
                resume: self.settings.resume_downloads,
                retries: self.settings.download_retries,
            },
        ));
    }

    fn poll_downloads(&mut self, ctx: &egui::Context) {
        let Some(receiver) = self.downloads.take() else {
            return;
        };
        let mut keep = true;
        loop {
            match receiver.try_recv() {
                Ok(DownloadEvent::Started(key)) => {
                    self.download_current = Some(key);
                    self.download_progress = (0, None);
                }
                Ok(DownloadEvent::Progress(key, downloaded, total)) => {
                    self.download_current = Some(key);
                    self.download_progress = (downloaded, total);
                }
                Ok(DownloadEvent::Complete(key, path)) => {
                    self.runtime_config.set_path(key, path);
                    if let Err(error) = self.runtime_config.save() {
                        self.fail(error);
                    }
                }
                Ok(DownloadEvent::Failed(key, error)) => {
                    self.download_current = Some(key);
                    self.fail(error);
                    keep = false;
                    break;
                }
                Ok(DownloadEvent::Finished) | Err(TryRecvError::Disconnected) => {
                    self.download_current = None;
                    self.refresh_backend();
                    keep = false;
                    break;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        if keep {
            self.downloads = Some(receiver);
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn refresh_backend(&mut self) {
        let availability = model_runtimes::availability();
        self.backend_available = availability.is_ok();
        self.status = availability
            .map(|_| rust_i18n::t!("ready").into_owned())
            .unwrap_or_else(|error| error);
    }

    fn title_and_toolbar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        // `horizontal_centered` expands to all remaining vertical space in egui.
        // This header must stay at its intrinsic toolbar height so it cannot hide
        // the file actions, progress bar, workspace, and save controls below it.
        ui.horizontal(|ui| {
            ui.add(
                egui::Image::new(egui::include_image!("../assets/bibiocr-logo.png"))
                    .fit_to_exact_size(Vec2::new(158.0, 28.0)),
            );
            ui.separator();
            ui.heading(rust_i18n::t!("app_title"));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().interact_size.y = 28.0;
                if ui
                    .add_sized([90.0, 28.0], egui::Button::new(rust_i18n::t!("about")))
                    .clicked()
                {
                    self.about_open = true;
                }
                if ui
                    .add_sized(
                        [130.0, 28.0],
                        egui::Button::new(rust_i18n::t!("dependencies")),
                    )
                    .clicked()
                {
                    self.downloads_open = true;
                }
                if ui
                    .add_sized(
                        [125.0, 28.0],
                        egui::Button::new(rust_i18n::t!("reset_panels")),
                    )
                    .clicked()
                {
                    self.dock_state = default_dock_state();
                    self.maximized = None;
                }
                self.language_selector(ui);
            });
        });
        ui.add_space(5.0);
        ui.horizontal(|ui| {
            let busy = self.pipeline.is_some();
            if ui
                .add_enabled(!busy, egui::Button::new(rust_i18n::t!("read_image")))
                .clicked()
            {
                self.read_image_file(ctx);
            }
            if ui
                .add_enabled(!busy, egui::Button::new(rust_i18n::t!("read_clipboard")))
                .clicked()
            {
                self.read_clipboard(ctx);
            }
            ui.separator();
            ui.colored_label(
                if self.backend_available {
                    GREEN
                } else {
                    Color32::from_rgb(205, 82, 82)
                },
                if self.backend_available {
                    rust_i18n::t!("backend_ready")
                } else {
                    rust_i18n::t!("backend_unavailable")
                },
            );
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(rust_i18n::t!("progress"));
            ui.scope(|ui| {
                ui.visuals_mut().extreme_bg_color = Color32::from_rgb(42, 57, 74);
                ui.visuals_mut().override_text_color = Some(Color32::WHITE);
                ui.add(
                    ProgressBar::new(self.progress)
                        .fill(if self.progress >= 1.0 { GREEN } else { BLUE })
                        .text(
                            RichText::new(format!("{}%", (self.progress * 100.0).round() as u32))
                                .strong()
                                .color(Color32::WHITE),
                        )
                        .animate(self.pipeline.is_some())
                        .desired_height(26.0)
                        .desired_width(ui.available_width()),
                );
            });
        });
    }

    fn language_selector(&mut self, ui: &mut egui::Ui) {
        let mut selected = self.locale.selected().to_owned();
        let active_name = SUPPORTED_LOCALES
            .iter()
            .find(|locale| locale.code == self.locale.active())
            .map(|locale| locale.native_name)
            .unwrap_or("English");
        let selected_text = if selected == "system" {
            format!("{} ({active_name})", rust_i18n::t!("system_language"))
        } else {
            SUPPORTED_LOCALES
                .iter()
                .find(|locale| locale.code == selected)
                .map(|locale| locale.native_name.to_owned())
                .unwrap_or_else(|| "English".to_owned())
        };
        egui::ComboBox::from_id_salt("locale-selector")
            .width(190.0)
            .selected_text(format!("{}: {selected_text}", rust_i18n::t!("language")))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut selected,
                    "system".to_owned(),
                    rust_i18n::t!("system_language"),
                );
                for locale in SUPPORTED_LOCALES {
                    ui.selectable_value(&mut selected, locale.code.to_owned(), locale.native_name);
                }
            });
        if selected != self.locale.selected() {
            let previous_initial = rust_i18n::t!("initial_markdown").into_owned();
            let previous_processing = rust_i18n::t!("processing_markdown").into_owned();
            self.locale.set(&selected, &mut self.settings);
            let translated_initial = rust_i18n::t!("initial_markdown").into_owned();
            let translated_processing = rust_i18n::t!("processing_markdown").into_owned();
            translate_markdown_placeholder(
                &mut self.workspace.markdown,
                &previous_initial,
                &previous_processing,
                &translated_initial,
                &translated_processing,
            );
            apply_dock_translations(&mut self.dock_state);
            self.status = rust_i18n::t!("ready").into_owned();
        }
    }

    fn workspace_ui(&mut self, ui: &mut egui::Ui) {
        let mut maximize_request: Option<Option<WorkspaceTab>> = None;
        let mut viewer = WorkspaceViewer {
            workspace: &mut self.workspace,
            runtime_config: &self.runtime_config,
            maximize_request: &mut maximize_request,
        };
        if let Some(tab) = self.maximized {
            viewer.tab_ui(ui, tab, true);
        } else {
            DockArea::new(&mut self.dock_state)
                .style(Style::from_egui(ui.style().as_ref()))
                .show_add_buttons(false)
                .show_close_buttons(false)
                .show_inside(ui, &mut viewer);
        }
        if let Some(request) = maximize_request {
            self.maximized = request;
        }
    }

    fn bottom_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button(rust_i18n::t!("save_markdown")).clicked() {
                self.save_markdown();
            }
            if ui.button(rust_i18n::t!("save_docx")).clicked() {
                self.save_docx();
            }
            ui.separator();
            ui.label(
                RichText::new(&self.status)
                    .small()
                    .color(Color32::DARK_GRAY),
            );
        });
    }

    fn about_window(&mut self, ctx: &egui::Context) {
        egui::Window::new(rust_i18n::t!("about_title"))
            .open(&mut self.about_open)
            .resizable(false)
            .default_width(640.0)
            .show(ctx, |ui| {
                ui.add(
                    egui::Label::new(RichText::new(rust_i18n::t!("about_description")).size(17.0))
                        .wrap(),
                );
                ui.add_space(10.0);
                let width = ui.available_width().min(620.0);
                ui.add(
                    egui::Image::new(egui::include_image!("../assets/sponsor.png"))
                        .fit_to_exact_size(Vec2::new(width, width / 2.1)),
                );
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new(rust_i18n::t!("about_slogan"))
                            .strong()
                            .size(18.0),
                    );
                });
            });
    }

    fn failure_window(&mut self, ctx: &egui::Context) {
        let Some(message) = self.failure_dialog.clone() else {
            return;
        };
        let mut open = true;
        let mut close_requested = false;
        egui::Window::new(rust_i18n::t!("failure_title"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.add(egui::Label::new(RichText::new(message).color(Color32::DARK_RED)).wrap());
                ui.add_space(10.0);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button(rust_i18n::t!("close")).clicked() {
                        close_requested = true;
                    }
                });
            });
        if !open || close_requested {
            self.failure_dialog = None;
        }
    }

    fn downloads_window(&mut self, ctx: &egui::Context) {
        let mut open = self.downloads_open;
        let mut browse = None;
        let mut download_keys = None;
        egui::Window::new(rust_i18n::t!("dependencies_title"))
            .open(&mut open)
            .resizable(true)
            .default_size([820.0, 620.0])
            .show(ctx, |ui| {
                ui.label(rust_i18n::t!("dependencies_intro"));
                ui.separator();
                ui.group(|ui| {
                    ui.set_min_width(ui.available_width());
                    ui.strong(rust_i18n::t!("proxy_title"));
                    let mut mode = self.settings.proxy_mode();
                    ui.radio_value(&mut mode, ProxyMode::None, rust_i18n::t!("proxy_none"));
                    ui.radio_value(&mut mode, ProxyMode::System, rust_i18n::t!("proxy_system"));
                    ui.radio_value(&mut mode, ProxyMode::Manual, rust_i18n::t!("proxy_manual"));
                    self.settings.set_proxy_mode(mode);
                    ui.add_enabled_ui(mode == ProxyMode::Manual, |ui| {
                        ui.indent("manual-proxy", |ui| {
                            ui.horizontal(|ui| {
                                ui.label(rust_i18n::t!("proxy"));
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.settings.proxy)
                                        .desired_width(520.0)
                                        .hint_text(rust_i18n::t!("proxy_hint")),
                                );
                            });
                        });
                    });
                });
                ui.add_space(4.0);
                ui.group(|ui| {
                    ui.set_min_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.checkbox(
                            &mut self.settings.use_hf_mirror,
                            rust_i18n::t!("use_hf_mirror"),
                        );
                        ui.label(rust_i18n::t!("hf_endpoint"));
                        ui.add_enabled_ui(self.settings.use_hf_mirror, |ui| {
                            ui.text_edit_singleline(&mut self.settings.hf_endpoint);
                            if ui.button("HF Mirror").clicked() {
                                self.settings.hf_endpoint = "https://hf-mirror.com".to_owned();
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(
                            &mut self.settings.use_github_proxy,
                            rust_i18n::t!("use_github_proxy"),
                        );
                        ui.label(rust_i18n::t!("github_proxy"));
                        ui.add_enabled_ui(self.settings.use_github_proxy, |ui| {
                            ui.text_edit_singleline(&mut self.settings.github_proxy);
                        });
                    });
                    ui.small(rust_i18n::t!("github_proxy_hint"));
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.settings.resume_downloads, rust_i18n::t!("resume"));
                    ui.label(rust_i18n::t!("retries"));
                    ui.add(egui::DragValue::new(&mut self.settings.download_retries).range(0..=10));
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for key in DependencyKey::ALL {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                let installed = self.runtime_config.path(key).is_file();
                                ui.colored_label(
                                    if installed { GREEN } else { Color32::DARK_RED },
                                    if installed { "●" } else { "○" },
                                );
                                ui.strong(key.name());
                                ui.label(self.runtime_config.path(key).display().to_string());
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if ui
                                        .add_enabled(
                                            self.downloads.is_none(),
                                            egui::Button::new(rust_i18n::t!("download")),
                                        )
                                        .clicked()
                                    {
                                        download_keys = Some(vec![key]);
                                    }
                                    if ui.button(rust_i18n::t!("browse")).clicked() {
                                        browse = Some(key);
                                    }
                                });
                            });
                        });
                    }
                });
                if let Some(key) = self.download_current {
                    let (downloaded, total) = self.download_progress;
                    let ratio = total
                        .filter(|value| *value > 0)
                        .map(|value| downloaded as f32 / value as f32)
                        .unwrap_or(0.0);
                    ui.add(ProgressBar::new(ratio).show_percentage().text(format!(
                            "{}: {:.1} MiB{}",
                            key.name(),
                            downloaded as f64 / 1_048_576.0,
                            total
                                .map(|value| format!(" / {:.1} MiB", value as f64 / 1_048_576.0))
                                .unwrap_or_default()
                        )));
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            self.downloads.is_none(),
                            egui::Button::new(rust_i18n::t!("download_missing")),
                        )
                        .clicked()
                    {
                        download_keys = Some(self.runtime_config.missing());
                    }
                    if ui.button(rust_i18n::t!("save_configuration")).clicked()
                        && let Err(error) = self
                            .runtime_config
                            .save()
                            .and_then(|_| self.settings.save())
                    {
                        self.fail(error);
                    }
                });
            });
        self.downloads_open = open;
        if let Some(key) = browse {
            self.browse_dependency(key);
        }
        if let Some(keys) = download_keys {
            self.start_downloads(keys);
        }
    }
}

impl eframe::App for BibiOcrApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let result = guard_operation(|| {
            self.frame_ui(ui, &ctx);
            Ok(())
        });
        if let Err(error) = result {
            self.pipeline = None;
            self.fail(error);
            ctx.request_repaint();
        }
    }
}

impl BibiOcrApp {
    fn frame_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        self.poll_pipeline(ctx);
        self.poll_downloads(ctx);
        self.workspace.batch.poll(ctx);
        self.handle_drop(ctx);

        egui::Panel::top("top-toolbar")
            .frame(
                egui::Frame::new()
                    .fill(Color32::WHITE)
                    .inner_margin(Margin::symmetric(12, 8))
                    .stroke(Stroke::new(1.0, Color32::from_gray(210))),
            )
            .show(ui, |ui| self.title_and_toolbar(ctx, ui));
        egui::Panel::bottom("bottom-actions")
            .frame(
                egui::Frame::new()
                    .fill(Color32::WHITE)
                    .inner_margin(Margin::symmetric(12, 8))
                    .stroke(Stroke::new(1.0, Color32::from_gray(210))),
            )
            .show(ui, |ui| self.bottom_bar(ui));
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(WORKSPACE_BG)
                    .inner_margin(Margin::same(6)),
            )
            .show(ui, |ui| self.workspace_ui(ui));
        if self.about_open {
            self.about_window(ctx);
        }
        if self.downloads_open {
            self.downloads_window(ctx);
        }
        self.failure_window(ctx);
    }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "unknown panic".to_owned()
    }
}

fn guard_operation<T>(operation: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => {
            Err(rust_i18n::t!("unexpected_error", error = panic_message(payload)).into_owned())
        }
    }
}

fn translate_markdown_placeholder(
    markdown: &mut String,
    previous_initial: &str,
    previous_processing: &str,
    translated_initial: &str,
    translated_processing: &str,
) {
    if markdown == previous_initial {
        *markdown = translated_initial.to_owned();
    } else if markdown == previous_processing {
        *markdown = translated_processing.to_owned();
    }
}

fn backend_event_is_terminal(event: &BackendEvent) -> bool {
    matches!(event, BackendEvent::Complete(_) | BackendEvent::Failed(_))
}

struct WorkspaceViewer<'a> {
    workspace: &'a mut Workspace,
    runtime_config: &'a RuntimeConfig,
    maximize_request: &'a mut Option<Option<WorkspaceTab>>,
}

impl WorkspaceViewer<'_> {
    fn tab_ui(&mut self, ui: &mut egui::Ui, tab: WorkspaceTab, maximized: bool) {
        ui.horizontal(|ui| {
            ui.strong(tab.title());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let label = if maximized {
                    rust_i18n::t!("restore")
                } else {
                    rust_i18n::t!("maximize")
                };
                if ui.small_button(label).clicked() {
                    *self.maximize_request = Some(if maximized { None } else { Some(tab) });
                }
            });
        });
        ui.separator();
        match tab {
            WorkspaceTab::Original => image_view(
                ui,
                self.workspace.original.as_ref(),
                &mut self.workspace.original_view,
                rust_i18n::t!("empty_original").as_ref(),
            ),
            WorkspaceTab::Layout => image_view(
                ui,
                self.workspace.layout.as_ref(),
                &mut self.workspace.layout_view,
                rust_i18n::t!("empty_layout").as_ref(),
            ),
            WorkspaceTab::Markdown => markdown_workspace(ui, self.workspace),
            WorkspaceTab::Batch => self.workspace.batch.ui(ui, self.runtime_config),
            WorkspaceTab::Yaml => self.workspace.yaml.ui(ui),
        }
    }
}

impl TabViewer for WorkspaceViewer<'_> {
    type Tab = WorkspaceTab;

    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id {
        egui::Id::new(*tab)
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        self.tab_ui(ui, *tab, false);
    }

    fn is_closeable(&self, _tab: &Self::Tab) -> bool {
        false
    }
}

fn default_dock_state() -> DockState<WorkspaceTab> {
    let mut state = DockState::new(vec![
        WorkspaceTab::Markdown,
        WorkspaceTab::Batch,
        WorkspaceTab::Yaml,
    ]);
    apply_dock_translations(&mut state);
    let surface = state.main_surface_mut();
    let [_markdown, left] =
        surface.split_left(NodeIndex::root(), 0.52, vec![WorkspaceTab::Original]);
    surface.split_below(left, 0.55, vec![WorkspaceTab::Layout]);
    state
}

fn apply_dock_translations(state: &mut DockState<WorkspaceTab>) {
    state.translations.tab_context_menu.close_button = rust_i18n::t!("dock_close").into_owned();
    state.translations.tab_context_menu.eject_button = rust_i18n::t!("dock_float").into_owned();
    state.translations.tab_context_menu.hide_tab_bar_button =
        rust_i18n::t!("dock_hide_bar").into_owned();
    state.translations.tab_context_menu.show_tab_bar_button =
        rust_i18n::t!("dock_show_bar").into_owned();
    state.translations.leaf.close_button_disabled_tooltip =
        rust_i18n::t!("dock_cannot_close").into_owned();
    state.translations.leaf.close_all_button = rust_i18n::t!("dock_close_window").into_owned();
    state.translations.leaf.minimize_button = rust_i18n::t!("dock_minimize_window").into_owned();
}

fn image_view(
    ui: &mut egui::Ui,
    image: Option<&LoadedImage>,
    view: &mut ImageViewState,
    empty_text: &str,
) {
    ui.horizontal(|ui| {
        ui.label(rust_i18n::t!("zoom"));
        ui.add(egui::Slider::new(&mut view.zoom, 0.25..=4.0).logarithmic(true));
        if ui.small_button(rust_i18n::t!("fit")).clicked() {
            view.reset();
        }
        ui.label(rust_i18n::t!("drag_pan"));
    });
    let size = ui.available_size().max(Vec2::splat(80.0));
    let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
    painter.rect_filled(response.rect, 4.0, Color32::from_rgb(224, 229, 235));
    if response.dragged() {
        view.pan += response.drag_delta();
    }
    let Some(image) = image else {
        painter.text(
            response.rect.center(),
            egui::Align2::CENTER_CENTER,
            empty_text,
            egui::FontId::proportional(15.0),
            Color32::from_gray(105),
        );
        return;
    };
    let native = Vec2::new(image.size[0] as f32, image.size[1] as f32);
    let fit = (response.rect.width() / native.x)
        .min(response.rect.height() / native.y)
        .max(0.001);
    let drawn = Rect::from_center_size(response.rect.center() + view.pan, native * fit * view.zoom);
    painter.image(
        image.texture.id(),
        drawn,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
}

fn markdown_workspace(ui: &mut egui::Ui, workspace: &mut Workspace) {
    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut workspace.markdown_tab,
            MarkdownTab::Edit,
            rust_i18n::t!("edit"),
        );
        ui.selectable_value(
            &mut workspace.markdown_tab,
            MarkdownTab::Preview,
            rust_i18n::t!("preview"),
        );
    });
    ui.separator();
    match workspace.markdown_tab {
        MarkdownTab::Edit => {
            ui.add_sized(
                ui.available_size(),
                egui::TextEdit::multiline(&mut workspace.markdown)
                    .font(egui::TextStyle::Monospace)
                    .code_editor()
                    .desired_rows(24),
            );
        }
        MarkdownTab::Preview => {
            let preview =
                markdown_for_preview(&workspace.markdown, workspace.output_dir.as_deref());
            egui::ScrollArea::vertical().show(ui, |ui| {
                let viewer = CommonMarkViewer::new()
                    .render_html_fn(Some(&html_preview::render_html))
                    .max_image_width(Some(ui.available_width().max(120.0) as usize));
                viewer.show(ui, &mut workspace.markdown_cache, &preview);
            });
        }
    }
}

fn markdown_for_preview(markdown: &str, output_dir: Option<&std::path::Path>) -> String {
    let Some(output_dir) = output_dir else {
        return markdown.to_owned();
    };
    let base = output_dir.to_string_lossy().replace('\\', "/");
    markdown.replace("](imgs/", &format!("](file:///{base}/imgs/"))
}

fn clipboard_rgba(
    width: usize,
    height: usize,
    bytes: Vec<u8>,
    max_texture_side: usize,
) -> Result<RgbaImage, String> {
    validate_image_dimensions(width, height, bytes.len(), max_texture_side)?;
    let width =
        u32::try_from(width).map_err(|_| rust_i18n::t!("clipboard_invalid").into_owned())?;
    let height =
        u32::try_from(height).map_err(|_| rust_i18n::t!("clipboard_invalid").into_owned())?;
    RgbaImage::from_raw(width, height, bytes)
        .ok_or_else(|| rust_i18n::t!("clipboard_invalid").into_owned())
}

fn validate_image_dimensions(
    width: usize,
    height: usize,
    byte_len: usize,
    max_texture_side: usize,
) -> Result<(), String> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| rust_i18n::t!("clipboard_invalid").into_owned())?;
    let expected_bytes = pixels
        .checked_mul(4)
        .ok_or_else(|| rust_i18n::t!("clipboard_invalid").into_owned())?;
    if width == 0 || height == 0 || expected_bytes != byte_len {
        return Err(rust_i18n::t!("clipboard_invalid").into_owned());
    }
    if width > max_texture_side || height > max_texture_side || pixels > MAX_IMAGE_PIXELS {
        return Err(rust_i18n::t!(
            "image_too_large",
            width = width,
            height = height,
            max = max_texture_side
        )
        .into_owned());
    }
    Ok(())
}

fn texture_from_rgba(
    ctx: &egui::Context,
    name: &str,
    rgba: &RgbaImage,
) -> Result<LoadedImage, String> {
    let size = [rgba.width() as usize, rgba.height() as usize];
    let max_texture_side = ctx.input(|input| input.max_texture_side);
    validate_image_dimensions(size[0], size[1], rgba.as_raw().len(), max_texture_side)?;
    let color = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
    Ok(LoadedImage {
        size,
        texture: ctx.load_texture(name, color, TextureOptions::LINEAR),
    })
}

fn configure_style(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();
    visuals.panel_fill = WORKSPACE_BG;
    visuals.window_fill = Color32::WHITE;
    visuals.selection.bg_fill = BLUE.gamma_multiply(0.35);
    ctx.set_visuals(visuals);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = Vec2::new(7.0, 6.0);
        style.spacing.button_padding = Vec2::new(11.0, 5.0);
    });
}

fn install_cjk_font(ctx: &egui::Context) {
    let candidates = if cfg!(target_os = "windows") {
        vec![
            PathBuf::from(r"C:\Windows\Fonts\msyh.ttc"),
            PathBuf::from(r"C:\Windows\Fonts\simhei.ttf"),
            PathBuf::from(r"C:\Windows\Fonts\meiryo.ttc"),
            PathBuf::from(r"C:\Windows\Fonts\malgun.ttf"),
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            PathBuf::from("/System/Library/Fonts/PingFang.ttc"),
            PathBuf::from("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
            PathBuf::from("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        ]
    } else {
        vec![
            PathBuf::from("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            PathBuf::from("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
            PathBuf::from("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ]
    };
    let loaded: Vec<_> = candidates
        .into_iter()
        .filter_map(|path| fs::read(path).ok())
        .collect();
    if loaded.is_empty() {
        return;
    }
    let mut fonts = FontDefinitions::default();
    let names: Vec<_> = loaded
        .into_iter()
        .enumerate()
        .map(|(index, bytes)| {
            let name = format!("system-i18n-{index}");
            fonts
                .font_data
                .insert(name.clone(), FontData::from_owned(bytes).into());
            name
        })
        .collect();
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        let fallback = fonts.families.entry(family).or_default();
        for name in names.iter().rev() {
            fallback.insert(0, name.clone());
        }
    }
    ctx.set_fonts(fonts);
}

fn temporary_image_path() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    std::env::temp_dir()
        .join("bibiocr")
        .join(format!("clipboard-{stamp}.png"))
}

fn save_target_for(
    input: Option<&std::path::Path>,
    last_input_dir: Option<&std::path::Path>,
    extension: &str,
) -> (PathBuf, String) {
    if let Some(input) = input {
        let directory = input
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let stem = input
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("document");
        return (directory, format!("{stem}.{extension}"));
    }
    (
        last_input_dir
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
        format!("document.{extension}"),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        backend_event_is_terminal, clipboard_rgba, guard_operation, save_target_for,
        translate_markdown_placeholder, validate_image_dimensions,
    };
    use crate::model_runtimes::{BackendEvent, PipelineOutput};
    use std::path::Path;

    #[test]
    fn export_defaults_to_input_directory_and_stem() {
        let input = Path::new("scans").join("invoice.001.png");
        let (directory, filename) = save_target_for(Some(&input), None, "docx");
        assert_eq!(directory, Path::new("scans"));
        assert_eq!(filename, "invoice.001.docx");
    }

    #[test]
    fn locale_switch_translates_untouched_markdown_placeholder() {
        let mut markdown = "# Image to Markdown\n\nOpen an image to begin.\n".to_owned();
        translate_markdown_placeholder(
            &mut markdown,
            "# Image to Markdown\n\nOpen an image to begin.\n",
            "processing",
            "# 图片转 Markdown\n\n请读取图片开始处理。\n",
            "处理中",
        );
        assert_eq!(markdown, "# 图片转 Markdown\n\n请读取图片开始处理。\n");
    }

    #[test]
    fn locale_switch_preserves_user_markdown() {
        let mut markdown = "# My edited document".to_owned();
        translate_markdown_placeholder(&mut markdown, "initial", "processing", "初始", "处理中");
        assert_eq!(markdown, "# My edited document");
    }

    #[test]
    fn backend_success_and_failure_are_terminal_but_progress_is_not() {
        assert!(!backend_event_is_terminal(&BackendEvent::Progress(
            0.5,
            "working".to_owned(),
        )));
        assert!(backend_event_is_terminal(&BackendEvent::Complete(
            PipelineOutput {
                markdown: String::new(),
                overlay_path: "overlay.png".into(),
                output_dir: "output".into(),
            },
        )));
        assert!(backend_event_is_terminal(&BackendEvent::Failed(
            "error".to_owned(),
        )));
    }

    #[test]
    fn clipboard_rgba_rejects_zero_and_mismatched_images() {
        assert!(clipboard_rgba(0, 1, Vec::new(), 16_384).is_err());
        assert!(clipboard_rgba(2, 2, vec![0; 15], 16_384).is_err());
    }

    #[test]
    fn clipboard_rgba_accepts_valid_image_data() {
        let image = clipboard_rgba(2, 2, vec![255; 16], 16_384).unwrap();
        assert_eq!(image.dimensions(), (2, 2));
    }

    #[test]
    fn image_validation_rejects_texture_limit_and_integer_overflow() {
        assert!(validate_image_dimensions(4097, 1, 4097 * 4, 4096).is_err());
        assert!(validate_image_dimensions(usize::MAX, 2, 0, usize::MAX).is_err());
    }

    #[test]
    fn ui_operation_guard_converts_panics_to_errors() {
        let error = guard_operation::<()>(|| panic!("simulated clipboard panic")).unwrap_err();
        assert!(error.contains("simulated clipboard panic"));
    }
}
