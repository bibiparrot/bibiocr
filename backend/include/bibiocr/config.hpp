#pragma once

#include <filesystem>

namespace bibiocr {

struct AppConfig {
    std::filesystem::path vlm_model;
    std::filesystem::path mmproj;
    std::filesystem::path layout_model;
    std::filesystem::path ort_dll;
    std::filesystem::path llama_server;
};

[[nodiscard]] AppConfig load_config(const std::filesystem::path& config_path);

}  // namespace bibiocr
