#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace bibiocr {

struct LayoutBlock {
    int class_id{};
    std::string label;
    float score{};
    float left{};
    float top{};
    float right{};
    float bottom{};
    int order{};
    bool has_order{};
};

struct LayoutAnalysis {
    int width{};
    int height{};
    std::vector<LayoutBlock> layout_blocks;
    std::vector<LayoutBlock> parsing_blocks;
};

class LayoutAnalyzer {
public:
    LayoutAnalyzer(std::filesystem::path model_path, std::filesystem::path runtime_dll);

    [[nodiscard]] LayoutAnalysis analyze_document(
        const std::filesystem::path& image_path) const;

    [[nodiscard]] std::vector<LayoutBlock> analyze(
        const std::filesystem::path& image_path) const;

private:
    std::filesystem::path model_path_;
    std::filesystem::path runtime_dll_;
};

}  // namespace bibiocr
