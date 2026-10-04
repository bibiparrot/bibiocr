fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let mut build = cxx_build::bridge("src/ffi.rs");
    build
        .include("src/model_runtimes/include")
        .include("src/model_runtimes/third_party/onnxruntime")
        .flag_if_supported("/std:c++20")
        .flag_if_supported("/permissive-")
        .flag_if_supported("/utf-8")
        .flag_if_supported("-std=c++20");

    for source in [
        "src/model_runtimes/src/bridge.cpp",
        "src/model_runtimes/src/config.cpp",
        "src/model_runtimes/src/document_pipeline.cpp",
        "src/model_runtimes/src/document_result.cpp",
        "src/model_runtimes/src/layout_analyzer.cpp",
    ] {
        build.file(source);
    }
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dl");
    }
    build.compile("bibiocr_backend");

    if target_os == "windows" {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/bibiocr.ico");
        resource.set("ProductName", "BIBIOCR");
        resource.set("FileDescription", "BIBIOCR image-to-Markdown workspace");
        resource.set("LegalCopyright", "BIBIOCR contributors");
        if let Err(error) = resource.compile() {
            println!("cargo:warning=Could not embed Windows resources: {error}");
        }
    }

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=src/model_runtimes/include");
    println!("cargo:rerun-if-changed=src/model_runtimes/src");
}
