#include "bibiocr/bridge.hpp"

#include "bibiocr/config.hpp"
#include "bibiocr/document_pipeline.hpp"
#include "bibiocr/llama_cpp.hpp"
#include "bibiocr/src/ffi.rs.h"

#include <filesystem>
#include <string>

namespace bibiocr {
namespace {

std::filesystem::path utf8_path(const std::string& value) {
    return std::filesystem::path(
        reinterpret_cast<const char8_t*>(value.data()),
        reinterpret_cast<const char8_t*>(value.data() + value.size()));
}

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}

}  // namespace

PipelineResponse run_pipeline(const std::string& image_path,
                              const std::string& output_dir,
                              const std::string& config_path) {
    const std::filesystem::path input = utf8_path(image_path);
    const std::filesystem::path output = utf8_path(output_dir);
    const AppConfig config = load_config(utf8_path(config_path));

    const LayoutAnalyzer layout(config.layout_model, config.ort_dll);
    LlamaServerConfig server_config;
    server_config.executable = config.llama_server;
    server_config.model = config.vlm_model;
    server_config.mmproj = config.mmproj;
    server_config.startup_timeout_seconds = 150;
    LlamaServerProcess server(server_config);
    LlamaCppRecognizer recognizer(server.endpoint());
    const DocumentPipeline pipeline(layout, recognizer);
    const DocumentResult result = pipeline.process(input);
    result.save_all(output);

    const std::filesystem::path layout_path =
        output / (result.input_path.stem().wstring() + L"_layout_det_res.png");
    PipelineResponse response;
    response.markdown = result.markdown;
    response.layout_path = path_utf8(layout_path);
    response.output_dir = path_utf8(output);
    return response;
}

}  // namespace bibiocr
