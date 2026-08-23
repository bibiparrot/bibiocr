fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let mut build = cxx_build::bridge("src/ffi.rs");
    build
        .include("backend/include")
        .include("backend/third_party/onnxruntime")
        .flag_if_supported("/std:c++20")
        .flag_if_supported("/permissive-")
        .flag_if_supported("/utf-8")
        .flag_if_supported("-std=c++20");

    if target_os == "windows" {
        for source in [
            "backend/src/bridge.cpp",
            "backend/src/config.cpp",
            "backend/src/document_pipeline.cpp",
            "backend/src/document_result.cpp",
            "backend/src/layout_analyzer.cpp",
            "backend/src/llama_cpp.cpp",
        ] {
            build.file(source);
        }
        for library in ["windowscodecs", "ole32", "winhttp", "ws2_32", "gdiplus"] {
            println!("cargo:rustc-link-lib={library}");
        }
    } else {
        build.file("backend/src/bridge_stub.cpp");
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
    println!("cargo:rerun-if-changed=backend/include");
    println!("cargo:rerun-if-changed=backend/src");
}
