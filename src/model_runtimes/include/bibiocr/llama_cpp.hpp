#pragma once

#include "bibiocr/document_pipeline.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace bibiocr {

struct LlamaServerConfig {
    std::filesystem::path executable;
    std::filesystem::path model;
    std::filesystem::path mmproj;
    int startup_timeout_seconds = 120;
};

class LlamaServerProcess {
public:
    explicit LlamaServerProcess(const LlamaServerConfig& config);
    ~LlamaServerProcess();

    LlamaServerProcess(const LlamaServerProcess&) = delete;
    LlamaServerProcess& operator=(const LlamaServerProcess&) = delete;
    LlamaServerProcess(LlamaServerProcess&&) noexcept;
    LlamaServerProcess& operator=(LlamaServerProcess&&) noexcept;

    [[nodiscard]] const std::string& endpoint() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class LlamaCppRecognizer final : public RegionRecognizer {
public:
    explicit LlamaCppRecognizer(std::string endpoint);
    ~LlamaCppRecognizer() override;

    LlamaCppRecognizer(const LlamaCppRecognizer&) = delete;
    LlamaCppRecognizer& operator=(const LlamaCppRecognizer&) = delete;
    LlamaCppRecognizer(LlamaCppRecognizer&&) noexcept;
    LlamaCppRecognizer& operator=(LlamaCppRecognizer&&) noexcept;

    std::string recognize(std::span<const std::byte> png,
                          std::string_view prompt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace bibiocr
