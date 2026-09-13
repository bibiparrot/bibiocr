#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

rust_i18n::i18n!("locales", fallback = "en");

mod app;
mod backend;
mod dependencies;
mod download;
mod export;
mod ffi;
mod html_preview;
mod locale;
mod settings;

use app::BibiOcrApp;
use eframe::egui;

fn main() -> eframe::Result {
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../assets/bibiocr-icon.png"))
        .expect("embedded BIBIOCR icon must be a valid PNG");

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default()
            .with_title("BIBIOCR")
            .with_inner_size([1100.0, 900.0])
            .with_min_inner_size([880.0, 680.0])
            .with_icon(icon),
        ..Default::default()
    };

    eframe::run_native(
        "BIBIOCR",
        options,
        Box::new(|cc| Ok(Box::new(BibiOcrApp::new(cc)))),
    )
}
