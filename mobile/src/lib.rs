rust_i18n::i18n!("locales", fallback = "zh-CN");

#[cfg(target_os = "android")]
pub mod android_bridge;
pub mod app_entry;
mod backend_gate;
pub mod core;
pub mod download;
pub mod export;
pub mod history;
mod kokoro_tts;
pub mod layout;
pub mod markdown;
pub mod mobile;
#[cfg(target_os = "android")]
pub mod recognizer;
pub mod settings;
pub mod tts;
pub mod v6;
