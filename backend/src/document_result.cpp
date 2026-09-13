#include "bibiocr/document_pipeline.hpp"
#include "bibiocr/src/ffi.rs.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bibiocr {
namespace {

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::string json_escape(std::string_view value) {
    std::string output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (ch < 0x20) {
                    std::ostringstream escaped;
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(ch);
                    output += escaped.str();
                } else {
                    output.push_back(static_cast<char>(ch));
                }
        }
    }
    return output;
}

int coordinate(float value) {
    return static_cast<int>(std::round(value));
}

void write_points(std::ostream& output, const LayoutBlock& block) {
    const int left = coordinate(block.left);
    const int top = coordinate(block.top);
    const int right = coordinate(block.right);
    const int bottom = coordinate(block.bottom);
    output << "[[" << left << ", " << top << "], [" << right << ", " << top
           << "], [" << right << ", " << bottom << "], [" << left << ", "
           << bottom << "]]";
}

void write_bbox(std::ostream& output, const LayoutBlock& block) {
    output << '[' << coordinate(block.left) << ", " << coordinate(block.top) << ", "
           << coordinate(block.right) << ", " << coordinate(block.bottom) << ']';
}

void write_json(const DocumentResult& result, const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create result JSON");
    const std::string input_path = json_escape(path_utf8(result.input_path));
    output << std::setprecision(8);
    output << "{\n"
           << "  \"input_path\": \"" << input_path << "\",\n"
           << "  \"page_index\": null,\n"
           << "  \"page_count\": null,\n"
           << "  \"width\": " << result.width << ",\n"
           << "  \"height\": " << result.height << ",\n"
           << "  \"model_settings\": {\n"
           << "    \"use_doc_preprocessor\": false,\n"
           << "    \"use_layout_detection\": true,\n"
           << "    \"use_chart_recognition\": false,\n"
           << "    \"use_seal_recognition\": false,\n"
           << "    \"use_ocr_for_image_block\": false,\n"
           << "    \"format_block_content\": false,\n"
           << "    \"merge_layout_blocks\": true,\n"
           << "    \"markdown_ignore_labels\": [\"number\", \"footnote\", \"header\", \"header_image\", \"footer\", \"footer_image\", \"aside_text\"],\n"
           << "    \"return_layout_polygon_points\": true\n"
           << "  },\n"
           << "  \"parsing_res_list\": [\n";
    for (std::size_t index = 0; index < result.parsing_blocks.size(); ++index) {
        const ParsedBlock& parsed = result.parsing_blocks[index];
        output << "    {\n"
               << "      \"block_label\": \"" << json_escape(parsed.layout.label)
               << "\",\n"
               << "      \"block_content\": \"" << json_escape(parsed.content)
               << "\",\n"
               << "      \"block_bbox\": ";
        write_bbox(output, parsed.layout);
        output << ",\n      \"block_id\": " << parsed.block_id
               << ",\n      \"block_order\": ";
        if (parsed.has_order) output << parsed.block_order;
        else output << "null";
        output << ",\n      \"group_id\": " << parsed.group_id
               << ",\n      \"block_polygon_points\": ";
        write_points(output, parsed.layout);
        output << "\n    }" << (index + 1 == result.parsing_blocks.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"layout_det_res\": {\n"
           << "    \"input_path\": \"" << input_path << "\",\n"
           << "    \"page_index\": null,\n"
           << "    \"boxes\": [\n";
    for (std::size_t index = 0; index < result.layout_blocks.size(); ++index) {
        const LayoutBlock& block = result.layout_blocks[index];
        output << "      {\n"
               << "        \"cls_id\": " << block.class_id << ",\n"
               << "        \"label\": \"" << json_escape(block.label) << "\",\n"
               << "        \"score\": " << block.score << ",\n"
               << "        \"coordinate\": ";
        write_bbox(output, block);
        output << ",\n        \"order\": ";
        if (block.has_order) output << block.order;
        else output << "null";
        output << ",\n        \"polygon_points\": ";
        write_points(output, block);
        output << "\n      }" << (index + 1 == result.layout_blocks.size() ? "\n" : ",\n");
    }
    output << "    ]\n  }\n}\n";
    if (!output) throw std::runtime_error("cannot write result JSON");
}

std::filesystem::path utf8_path(const std::string& value) {
    return std::filesystem::path(
        reinterpret_cast<const char8_t*>(value.data()),
        reinterpret_cast<const char8_t*>(value.data() + value.size()));
}

void save_images(const DocumentResult& result, const std::filesystem::path& save_path) {
    std::vector<ArtifactBlock> blocks;
    for (const LayoutBlock& block : result.layout_blocks) {
        ArtifactBlock artifact;
        artifact.class_id = block.class_id;
        artifact.label = block.label;
        artifact.left = block.left;
        artifact.top = block.top;
        artifact.right = block.right;
        artifact.bottom = block.bottom;
        blocks.push_back(std::move(artifact));
    }
    save_image_artifacts(
        rust::Str(path_utf8(result.input_path)), rust::Str(path_utf8(save_path)),
        rust::Str(path_utf8(result.input_path.stem())),
        rust::Slice<const ArtifactBlock>(blocks.data(), blocks.size()));
}

}  // namespace

void DocumentResult::save_all(const std::filesystem::path& save_path) const {
    std::error_code error;
    const std::filesystem::path image_directory = save_path / "imgs";
    std::filesystem::create_directories(image_directory, error);
    if (error) throw std::runtime_error("cannot create result directory");
    for (const auto& entry : std::filesystem::directory_iterator(image_directory)) {
        const std::string filename = path_utf8(entry.path().filename());
        if (entry.is_regular_file() && filename.starts_with("img_in_") &&
            entry.path().extension() == ".jpg") {
            std::filesystem::remove(entry.path(), error);
            if (error) throw std::runtime_error("cannot replace prior image artifacts");
        }
    }

    const std::filesystem::path markdown_path = save_path / utf8_path(
        path_utf8(input_path.stem()) + ".md");
    std::ofstream markdown_output(markdown_path, std::ios::binary | std::ios::trunc);
    if (!markdown_output) throw std::runtime_error("cannot create Markdown output file");
    markdown_output.write(markdown.data(), static_cast<std::streamsize>(markdown.size()));
    if (!markdown_output) throw std::runtime_error("cannot write Markdown output file");
    markdown_output.close();

    write_json(*this, save_path / utf8_path(
        path_utf8(input_path.stem()) + "_res.json"));
    save_images(*this, save_path);
}

}  // namespace bibiocr
