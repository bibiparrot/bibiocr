#include "bibiocr/document_pipeline.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <gdiplus.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace bibiocr {
namespace {

class GdiplusSession {
public:
    GdiplusSession() {
        Gdiplus::GdiplusStartupInput input;
        if (Gdiplus::GdiplusStartup(&token_, &input, nullptr) != Gdiplus::Ok) {
            throw std::runtime_error("cannot initialize image output support");
        }
    }
    ~GdiplusSession() { Gdiplus::GdiplusShutdown(token_); }

private:
    ULONG_PTR token_{};
};

CLSID encoder_clsid(const wchar_t* mime_type) {
    UINT count = 0;
    UINT bytes = 0;
    if (Gdiplus::GetImageEncodersSize(&count, &bytes) != Gdiplus::Ok || bytes == 0) {
        throw std::runtime_error("cannot enumerate image encoders");
    }
    std::vector<std::byte> storage(bytes);
    auto* encoders = reinterpret_cast<Gdiplus::ImageCodecInfo*>(storage.data());
    if (Gdiplus::GetImageEncoders(count, bytes, encoders) != Gdiplus::Ok) {
        throw std::runtime_error("cannot read image encoders");
    }
    for (UINT index = 0; index < count; ++index) {
        if (std::wstring_view(encoders[index].MimeType) == mime_type) {
            return encoders[index].Clsid;
        }
    }
    throw std::runtime_error("required image encoder is unavailable");
}

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

std::wstring widen_ascii(std::string_view value) {
    return {value.begin(), value.end()};
}

std::wstring image_filename(const LayoutBlock& block) {
    return L"img_in_" + widen_ascii(block.label) + L"_box_" +
           std::to_wstring(coordinate(block.left)) + L"_" +
           std::to_wstring(coordinate(block.top)) + L"_" +
           std::to_wstring(coordinate(block.right)) + L"_" +
           std::to_wstring(coordinate(block.bottom)) + L".jpg";
}

bool is_image_block(const LayoutBlock& block) {
    return block.label == "image" || block.label == "figure" || block.label == "seal";
}

void save_images(const DocumentResult& result, const std::filesystem::path& save_path) {
    GdiplusSession session;
    Gdiplus::Bitmap source(result.input_path.c_str());
    if (source.GetLastStatus() != Gdiplus::Ok) {
        throw std::runtime_error("cannot load input image for result artifacts");
    }
    const CLSID png = encoder_clsid(L"image/png");
    const CLSID jpeg = encoder_clsid(L"image/jpeg");

    Gdiplus::Bitmap overlay(result.input_path.c_str());
    if (overlay.GetLastStatus() != Gdiplus::Ok) {
        throw std::runtime_error("cannot create layout visualization");
    }
    Gdiplus::Graphics graphics(&overlay);
    graphics.SetSmoothingMode(Gdiplus::SmoothingModeHighQuality);
    Gdiplus::FontFamily family(L"Arial");
    Gdiplus::Font font(&family, 12.0F, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
    for (const LayoutBlock& block : result.layout_blocks) {
        const BYTE red = static_cast<BYTE>(40 + (block.class_id * 71) % 190);
        const BYTE green = static_cast<BYTE>(40 + (block.class_id * 47) % 190);
        const BYTE blue = static_cast<BYTE>(40 + (block.class_id * 29) % 190);
        Gdiplus::Pen pen(Gdiplus::Color(255, red, green, blue), 2.0F);
        const Gdiplus::RectF rectangle(
            block.left, block.top, block.right - block.left, block.bottom - block.top);
        graphics.DrawRectangle(&pen, rectangle);
        std::wostringstream label;
        label << widen_ascii(block.label) << L" " << std::fixed << std::setprecision(2)
              << block.score;
        Gdiplus::SolidBrush background(Gdiplus::Color(190, 255, 255, 255));
        const Gdiplus::RectF label_box(block.left, std::max(0.0F, block.top - 15.0F),
                                      180.0F, 15.0F);
        graphics.FillRectangle(&background, label_box);
        Gdiplus::SolidBrush text(Gdiplus::Color(255, red, green, blue));
        graphics.DrawString(label.str().c_str(), -1, &font,
                            Gdiplus::PointF(label_box.X, label_box.Y), &text);
    }
    const std::filesystem::path overlay_path =
        save_path / (result.input_path.stem().wstring() + L"_layout_det_res.png");
    if (overlay.Save(overlay_path.c_str(), &png, nullptr) != Gdiplus::Ok) {
        throw std::runtime_error("cannot save layout visualization");
    }

    const std::filesystem::path image_directory = save_path / L"imgs";
    for (const LayoutBlock& block : result.layout_blocks) {
        if (!is_image_block(block)) continue;
        const int left = std::clamp(coordinate(block.left), 0,
                                    static_cast<int>(source.GetWidth()));
        const int top = std::clamp(coordinate(block.top), 0,
                                   static_cast<int>(source.GetHeight()));
        const int right = std::clamp(coordinate(block.right), 0,
                                     static_cast<int>(source.GetWidth()));
        const int bottom = std::clamp(coordinate(block.bottom), 0,
                                      static_cast<int>(source.GetHeight()));
        if (right <= left || bottom <= top) continue;
        const Gdiplus::Rect crop_rectangle(left, top, right - left, bottom - top);
        std::unique_ptr<Gdiplus::Bitmap> crop(
            source.Clone(crop_rectangle, source.GetPixelFormat()));
        if (!crop || crop->GetLastStatus() != Gdiplus::Ok ||
            crop->Save((image_directory / image_filename(block)).c_str(), &jpeg, nullptr) !=
                Gdiplus::Ok) {
            throw std::runtime_error("cannot save extracted image block");
        }
    }
}

}  // namespace

void DocumentResult::save_all(const std::filesystem::path& save_path) const {
    std::error_code error;
    const std::filesystem::path image_directory = save_path / L"imgs";
    std::filesystem::create_directories(image_directory, error);
    if (error) throw std::runtime_error("cannot create result directory");
    for (const auto& entry : std::filesystem::directory_iterator(image_directory)) {
        const std::wstring filename = entry.path().filename().wstring();
        if (entry.is_regular_file() && filename.starts_with(L"img_in_") &&
            entry.path().extension() == L".jpg") {
            std::filesystem::remove(entry.path(), error);
            if (error) throw std::runtime_error("cannot replace prior image artifacts");
        }
    }

    const std::filesystem::path markdown_path =
        save_path / (input_path.stem().wstring() + L".md");
    std::ofstream markdown_output(markdown_path, std::ios::binary | std::ios::trunc);
    if (!markdown_output) throw std::runtime_error("cannot create Markdown output file");
    markdown_output.write(markdown.data(), static_cast<std::streamsize>(markdown.size()));
    if (!markdown_output) throw std::runtime_error("cannot write Markdown output file");
    markdown_output.close();

    write_json(*this, save_path / (input_path.stem().wstring() + L"_res.json"));
    save_images(*this, save_path);
}

}  // namespace bibiocr
