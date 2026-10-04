#include "bibiocr/config.hpp"
#include "bibiocr/document_pipeline.hpp"
#include "bibiocr/llama_cpp.hpp"

#include <Windows.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void print_help() {
    std::cout
        << "bibiocr_pipeline - document image to Markdown\n\n"
        << "Usage:\n"
        << "  bibiocr_pipeline --image <path> --save_path <directory> [--config <file>]\n"
        << "  bibiocr_pipeline -i <path> --save_path <directory>\n\n"
        << "Options:\n"
        << "  -i, --image <path>        Input PNG or JPEG image\n"
        << "      --save_path <dir>     Directory for Markdown, JSON, overlay, and imgs\n"
        << "      --config <file>       TOML config (default: bibiocr.toml)\n"
        << "      --vlm_url <url>       Existing llama.cpp /v1 endpoint\n"
        << "  -h, --help                Show this help\n";
}

struct CliOptions {
    std::filesystem::path image;
    std::filesystem::path save_path;
    std::filesystem::path config_path;
    std::wstring vlm_url;
};

std::optional<CliOptions> parse_cli(int argc, wchar_t** argv, std::wstring& error) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::wstring_view arg(argv[i]);
        if (arg == L"-i" || arg == L"--image") {
            if (++i >= argc) {
                error = L"--image requires a path";
                return std::nullopt;
            }
            options.image = argv[i];
        } else if (arg == L"--save_path") {
            if (++i >= argc) {
                error = L"--save_path requires a directory";
                return std::nullopt;
            }
            options.save_path = argv[i];
        } else if (arg == L"--vlm_url") {
            if (++i >= argc) {
                error = L"--vlm_url requires a URL";
                return std::nullopt;
            }
            options.vlm_url = argv[i];
        } else if (arg == L"--config") {
            if (++i >= argc) {
                error = L"--config requires a TOML file";
                return std::nullopt;
            }
            options.config_path = argv[i];
        } else {
            error = L"unknown option: " + std::wstring(arg);
            return std::nullopt;
        }
    }
    if (options.image.empty()) {
        error = L"--image is required";
        return std::nullopt;
    }
    if (options.save_path.empty()) {
        error = L"--save_path is required";
        return std::nullopt;
    }
    if (!std::filesystem::is_regular_file(options.image)) {
        error = L"input image does not exist: " + options.image.wstring();
        return std::nullopt;
    }
    return options;
}

std::filesystem::path executable_directory() {
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                            static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        throw std::runtime_error("cannot determine executable directory");
    }
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
}

std::filesystem::path locate_config(const std::filesystem::path& explicit_path) {
    if (!explicit_path.empty()) return explicit_path;
    const std::filesystem::path current =
        std::filesystem::current_path() / L"bibiocr.toml";
    if (std::filesystem::is_regular_file(current)) return current;
    const std::filesystem::path beside_executable =
        executable_directory() / L"bibiocr.toml";
    if (std::filesystem::is_regular_file(beside_executable)) return beside_executable;
    throw std::runtime_error(
        "cannot find bibiocr.toml in the current or executable directory; use --config");
}

std::string wide_to_utf8(std::wstring_view value) {
    if (value.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                                         static_cast<int>(value.size()), nullptr, 0,
                                         nullptr, nullptr);
    if (size <= 0) throw std::runtime_error("URL is not valid Unicode");
    std::string output(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                        static_cast<int>(value.size()), output.data(), size,
                        nullptr, nullptr);
    return output;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::wstring_view arg(argv[i]);
        if (arg == L"--help" || arg == L"-h") {
            print_help();
            return 0;
        }
    }
    std::wstring error;
    const auto options = parse_cli(argc, argv, error);
    if (!options) {
        std::wcerr << L"error: " << error << L'\n';
        return 2;
    }
    try {
        const std::filesystem::path config_path = locate_config(options->config_path);
        const bibiocr::AppConfig app_config = bibiocr::load_config(config_path);
        const bibiocr::LayoutAnalyzer layout(app_config.layout_model, app_config.ort_dll);
        std::unique_ptr<bibiocr::LlamaServerProcess> owned_server;
        std::string endpoint;
        if (options->vlm_url.empty()) {
            std::wcerr << L"loading PaddleOCR-VL-1.6 with llama.cpp...\n";
            bibiocr::LlamaServerConfig server_config;
            server_config.executable = app_config.llama_server;
            server_config.model = app_config.vlm_model;
            server_config.mmproj = app_config.mmproj;
            server_config.startup_timeout_seconds = 150;
            owned_server =
                std::make_unique<bibiocr::LlamaServerProcess>(server_config);
            endpoint = owned_server->endpoint();
        } else {
            endpoint = wide_to_utf8(options->vlm_url);
        }
        std::wcerr << L"analyzing layout and recognizing regions...\n";
        bibiocr::LlamaCppRecognizer recognizer(endpoint);
        const bibiocr::DocumentPipeline pipeline(layout, recognizer);
        const bibiocr::DocumentResult result = pipeline.process(options->image);
        result.save_all(options->save_path);
        std::wcout << L"saved Markdown, JSON, layout visualization, and "
                      L"image crops to "
                   << std::filesystem::absolute(options->save_path).wstring() << L'\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
