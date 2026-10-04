fn options() -> eframe::NativeOptions {
    eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("BIBIOCR")
            .with_inner_size([464.0, 928.0])
            .with_min_inner_size([360.0, 640.0]),
        ..Default::default()
    }
}

#[cfg(not(target_os = "android"))]
pub fn run() -> eframe::Result {
    eframe::run_native(
        "BIBIOCR",
        options(),
        Box::new(|cc| Ok(Box::new(crate::mobile::MobileApp::new(cc)))),
    )
}

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub fn android_main(app: android_activity::AndroidApp) {
    // SAFETY: AndroidApp owns this Activity reference for the current lifecycle.
    unsafe { robius_android_env::set_activity(app.activity_as_ptr()) }
        .expect("could not register Android Activity");
    if let Some(data) = app.internal_data_path() {
        // SAFETY: NativeActivity invokes this before eframe starts worker threads.
        unsafe { std::env::set_var("BIBIOCR_DATA", data) };
    }
    let insets_app = app.clone();
    let mut native_options = options();
    native_options.android_app = Some(app);
    eframe::run_native(
        "BIBIOCR",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(crate::mobile::MobileApp::new_android(
                cc, insets_app,
            )))
        }),
    )
    .expect("BIBIOCR failed to start");
}
