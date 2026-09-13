#include "bibiocr/bridge.hpp"

#include "bibiocr/config.hpp"
#include "bibiocr/document_pipeline.hpp"
#include "bibiocr/llama_cpp.hpp"
#include "bibiocr/src/ffi.rs.h"

#include <cstdint>
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

class RustRecognizer final : public RegionRecognizer {
public:
    explicit RustRecognizer(const LlamaSession& session) : session_(session) {}

    std::string recognize(std::span<const std::byte> png,
                          std::string_view prompt) override {
        const auto bytes = rust::Slice<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(png.data()), png.size());
        return std::string(session_.recognize(
            bytes, rust::Str(prompt.data(), prompt.size())));
    }

private:
    const LlamaSession& session_;
};

}  // namespace

PipelineResponse run_pipeline(const std::string& image_path,
                              const std::string& output_dir,
                              const std::string& config_path) {
    const std::filesystem::path input = utf8_path(image_path);
    const std::filesystem::path output = utf8_path(output_dir);
    const AppConfig config = load_config(utf8_path(config_path));

    const LayoutAnalyzer layout(config.layout_model, config.ort_dll);
    auto server = start_llama_server(
        rust::Str(path_utf8(config.llama_server)),
        rust::Str(path_utf8(config.vlm_model)),
        rust::Str(path_utf8(config.mmproj)), 150);
    RustRecognizer recognizer(*server);
    const DocumentPipeline pipeline(layout, recognizer);
    const DocumentResult result = pipeline.process(input);
    result.save_all(output);

    const std::filesystem::path layout_path = output / utf8_path(
        path_utf8(result.input_path.stem()) + "_layout_det_res.png");
    PipelineResponse response;
    response.markdown = result.markdown;
    response.layout_path = path_utf8(layout_path);
    response.output_dir = path_utf8(output);
    return response;
}

}  // namespace bibiocr
