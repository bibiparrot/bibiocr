#pragma once

#include "bibiocr/layout_analyzer.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bibiocr {

class RegionRecognizer {
public:
    virtual ~RegionRecognizer() = default;
    virtual std::string recognize(std::span<const std::byte> png,
                                  std::string_view prompt) = 0;
};

struct ParsedBlock {
    LayoutBlock layout;
    std::string content;
    int block_id{};
    int block_order{};
    bool has_order{};
    int group_id{};
};

struct DocumentResult {
    std::filesystem::path input_path;
    int width{};
    int height{};
    std::vector<LayoutBlock> layout_blocks;
    std::vector<ParsedBlock> parsing_blocks;
    std::string markdown;

    void save_all(const std::filesystem::path& save_path) const;
};

class DocumentPipeline {
public:
    DocumentPipeline(const LayoutAnalyzer& layout, RegionRecognizer& recognizer);

    [[nodiscard]] DocumentResult process(
        const std::filesystem::path& image_path) const;

    [[nodiscard]] std::string convert(const std::filesystem::path& image_path) const;

private:
    const LayoutAnalyzer& layout_;
    RegionRecognizer& recognizer_;
};

}  // namespace bibiocr
