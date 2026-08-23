#include "bibiocr/document_pipeline.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <wincodec.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bibiocr {
namespace {

template <typename T>
struct ComReleaser {
    void operator()(T* value) const noexcept {
        if (value != nullptr) {
            value->Release();
        }
    }
};

template <typename T>
using ComPtr = std::unique_ptr<T, ComReleaser<T>>;

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

class ComApartment {
public:
    ComApartment() {
        const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (result != RPC_E_CHANGED_MODE) {
            check_hresult(result, "CoInitializeEx");
            initialized_ = true;
        }
    }
    ~ComApartment() {
        if (initialized_) {
            CoUninitialize();
        }
    }

private:
    bool initialized_ = false;
};

class ImageCropEncoder {
public:
    explicit ImageCropEncoder(const std::filesystem::path& path) {
        IWICImagingFactory* factory_raw = nullptr;
        check_hresult(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                       IID_PPV_ARGS(&factory_raw)),
                      "create WIC factory");
        factory_.reset(factory_raw);

        IWICBitmapDecoder* decoder_raw = nullptr;
        check_hresult(factory_->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
                                                          WICDecodeMetadataCacheOnLoad,
                                                          &decoder_raw),
                      "decode input image");
        decoder_.reset(decoder_raw);

        IWICBitmapFrameDecode* frame_raw = nullptr;
        check_hresult(decoder_->GetFrame(0, &frame_raw), "read input image frame");
        frame_.reset(frame_raw);
    }

    std::vector<std::byte> crop_png(const LayoutBlock& block) const {
        const int left = static_cast<int>(std::floor(block.left));
        const int top = static_cast<int>(std::floor(block.top));
        const int width = static_cast<int>(std::ceil(block.right)) - left;
        const int height = static_cast<int>(std::ceil(block.bottom)) - top;
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("layout block has invalid dimensions");
        }

        IWICBitmapClipper* clipper_raw = nullptr;
        check_hresult(factory_->CreateBitmapClipper(&clipper_raw), "create region cropper");
        ComPtr<IWICBitmapClipper> clipper(clipper_raw);
        const WICRect rectangle{left, top, width, height};
        check_hresult(clipper->Initialize(frame_.get(), &rectangle), "crop layout region");

        IStream* stream_raw = nullptr;
        check_hresult(CreateStreamOnHGlobal(nullptr, TRUE, &stream_raw),
                      "create PNG memory stream");
        ComPtr<IStream> stream(stream_raw);

        IWICBitmapEncoder* encoder_raw = nullptr;
        check_hresult(factory_->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder_raw),
                      "create PNG encoder");
        ComPtr<IWICBitmapEncoder> encoder(encoder_raw);
        check_hresult(encoder->Initialize(stream.get(), WICBitmapEncoderNoCache),
                      "initialize PNG encoder");

        IWICBitmapFrameEncode* encoded_frame_raw = nullptr;
        IPropertyBag2* properties_raw = nullptr;
        check_hresult(encoder->CreateNewFrame(&encoded_frame_raw, &properties_raw),
                      "create PNG frame");
        ComPtr<IWICBitmapFrameEncode> encoded_frame(encoded_frame_raw);
        ComPtr<IPropertyBag2> properties(properties_raw);
        check_hresult(encoded_frame->Initialize(properties.get()), "initialize PNG frame");
        check_hresult(encoded_frame->SetSize(static_cast<UINT>(width),
                                             static_cast<UINT>(height)),
                      "size PNG frame");
        check_hresult(encoded_frame->WriteSource(clipper.get(), nullptr), "encode PNG crop");
        check_hresult(encoded_frame->Commit(), "commit PNG frame");
        check_hresult(encoder->Commit(), "commit PNG image");

        HGLOBAL memory = nullptr;
        check_hresult(GetHGlobalFromStream(stream.get(), &memory), "read PNG memory stream");
        const SIZE_T size = GlobalSize(memory);
        const void* data = GlobalLock(memory);
        if (data == nullptr || size == 0) {
            throw std::runtime_error("PNG crop encoding returned no data");
        }
        const auto* begin = static_cast<const std::byte*>(data);
        std::vector<std::byte> png(begin, begin + size);
        GlobalUnlock(memory);
        return png;
    }

private:
    ComPtr<IWICImagingFactory> factory_;
    ComPtr<IWICBitmapDecoder> decoder_;
    ComPtr<IWICBitmapFrameDecode> frame_;
};

std::string_view prompt_for_label(std::string_view label) {
    if (label == "table") {
        return "Table Recognition:";
    }
    if (label == "chart") {
        return "Chart Recognition:";
    }
    if (label == "seal") {
        return "Seal Recognition:";
    }
    if (label.find("formula") != std::string_view::npos && label != "formula_number") {
        return "Formula Recognition:";
    }
    return "OCR:";
}

std::string trim(std::string value) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(),
                std::find_if_not(value.begin(), value.end(), is_space));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(), value.end());
    return value;
}

struct OtslCell {
    std::string tag;
    std::string text;
};

std::pair<std::size_t, std::string_view> find_otsl_tag(std::string_view value,
                                                       std::size_t start) {
    static constexpr std::array<std::string_view, 6> tags = {
        "<fcel>", "<ecel>", "<nl>", "<lcel>", "<ucel>", "<xcel>"};
    std::size_t best = std::string_view::npos;
    std::string_view best_tag;
    for (const std::string_view tag : tags) {
        const std::size_t position = value.find(tag, start);
        if (position < best) {
            best = position;
            best_tag = tag;
        }
    }
    return {best, best_tag};
}

std::string html_escape(std::string_view value) {
    std::string output;
    for (const char ch : value) {
        switch (ch) {
            case '&': output += "&amp;"; break;
            case '<': output += "&lt;"; break;
            case '>': output += "&gt;"; break;
            case '"': output += "&quot;"; break;
            default: output.push_back(ch); break;
        }
    }
    return output;
}

std::string convert_otsl_to_html(std::string_view otsl) {
    std::vector<std::vector<OtslCell>> rows;
    std::vector<OtslCell> row;
    std::size_t cursor = 0;
    for (;;) {
        const auto [position, tag] = find_otsl_tag(otsl, cursor);
        if (position == std::string_view::npos) break;
        const std::size_t content_start = position + tag.size();
        const auto [next_position, unused] = find_otsl_tag(otsl, content_start);
        (void)unused;
        if (tag == "<nl>") {
            if (!row.empty()) {
                rows.push_back(std::move(row));
                row.clear();
            }
        } else {
            const std::size_t content_end =
                next_position == std::string_view::npos ? otsl.size() : next_position;
            std::string text;
            if (tag == "<fcel>") {
                text = trim(std::string(otsl.substr(content_start,
                                                    content_end - content_start)));
            }
            row.push_back({std::string(tag), std::move(text)});
        }
        if (next_position == std::string_view::npos) break;
        cursor = next_position;
    }
    if (!row.empty()) rows.push_back(std::move(row));
    if (rows.empty()) return {};

    std::size_t columns = 0;
    for (const auto& current : rows) columns = std::max(columns, current.size());
    for (auto& current : rows) {
        current.resize(columns, OtslCell{"<ecel>", {}});
    }

    std::string html = "<table>";
    for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
        html += "<tr>";
        for (std::size_t column = 0; column < columns; ++column) {
            const OtslCell& cell = rows[row_index][column];
            if (cell.tag != "<fcel>" && cell.tag != "<ecel>") continue;
            std::size_t colspan = 1;
            while (column + colspan < columns &&
                   (rows[row_index][column + colspan].tag == "<lcel>" ||
                    rows[row_index][column + colspan].tag == "<xcel>")) {
                ++colspan;
            }
            std::size_t rowspan = 1;
            while (row_index + rowspan < rows.size() &&
                   (rows[row_index + rowspan][column].tag == "<ucel>" ||
                    rows[row_index + rowspan][column].tag == "<xcel>")) {
                ++rowspan;
            }
            html += "<td";
            if (rowspan > 1) html += " rowspan=\"" + std::to_string(rowspan) + "\"";
            if (colspan > 1) html += " colspan=\"" + std::to_string(colspan) + "\"";
            html += ">" + html_escape(cell.text) + "</td>";
        }
        html += "</tr>";
    }
    html += "</table>";
    return html;
}

const std::unordered_set<std::string> kMarkdownIgnoredLabels = {
    "number", "footnote", "header", "header_image", "footer", "footer_image",
    "aside_text"};

const std::unordered_set<std::string> kImageLabels = {
    "image", "figure", "seal"};

std::string normalize_content(std::string_view label, std::string content) {
    content = trim(std::move(content));
    if (content.empty()) {
        return {};
    }
    if (label == "table" && content.find("<fcel>") != std::string::npos) {
        const std::string html = convert_otsl_to_html(content);
        if (!html.empty()) return html;
    }
    return content;
}

std::string format_markdown(std::string_view label, std::string content) {
    if (label == "doc_title") {
        return "# " + content;
    }
    if (label == "paragraph_title") {
        return "## " + content;
    }
    return content;
}

std::string image_filename(const LayoutBlock& block) {
    return "img_in_" + block.label + "_box_" +
           std::to_string(static_cast<int>(block.left)) + "_" +
           std::to_string(static_cast<int>(block.top)) + "_" +
           std::to_string(static_cast<int>(block.right)) + "_" +
           std::to_string(static_cast<int>(block.bottom)) + ".jpg";
}

}  // namespace

DocumentPipeline::DocumentPipeline(const LayoutAnalyzer& layout, RegionRecognizer& recognizer)
    : layout_(layout), recognizer_(recognizer) {}

DocumentResult DocumentPipeline::process(const std::filesystem::path& image_path) const {
    LayoutAnalysis analysis = layout_.analyze_document(image_path);
    if (analysis.layout_blocks.empty()) {
        throw std::runtime_error("DocLayoutV3 found no document content");
    }

    ComApartment apartment;
    const ImageCropEncoder image(image_path);
    DocumentResult result;
    result.input_path = std::filesystem::absolute(image_path);
    result.width = analysis.width;
    result.height = analysis.height;
    result.layout_blocks = std::move(analysis.layout_blocks);

    int ordered_index = 1;
    for (std::size_t index = 0; index < analysis.parsing_blocks.size(); ++index) {
        const LayoutBlock& block = analysis.parsing_blocks[index];
        std::string content;
        if (!kImageLabels.contains(block.label) && block.label != "header_image" &&
            block.label != "footer_image") {
            const std::vector<std::byte> png = image.crop_png(block);
            content = recognizer_.recognize(png, prompt_for_label(block.label));
            content = normalize_content(block.label, std::move(content));
        }

        ParsedBlock parsed;
        parsed.layout = block;
        parsed.content = content;
        parsed.block_id = static_cast<int>(index);
        parsed.has_order = block.has_order;
        parsed.block_order = block.has_order ? ordered_index++ : 0;
        parsed.group_id = static_cast<int>(index);
        result.parsing_blocks.push_back(std::move(parsed));

        if (kMarkdownIgnoredLabels.contains(block.label)) {
            continue;
        }
        std::string markdown_block;
        if (kImageLabels.contains(block.label)) {
            markdown_block = "![](imgs/" + image_filename(block) + ")";
        } else {
            markdown_block = format_markdown(block.label, content);
        }
        if (markdown_block.empty()) continue;
        if (!result.markdown.empty()) result.markdown += "\n\n";
        result.markdown += markdown_block;
    }
    if (!result.markdown.empty()) result.markdown += '\n';
    return result;
}

std::string DocumentPipeline::convert(const std::filesystem::path& image_path) const {
    return process(image_path).markdown;
}

}  // namespace bibiocr
