#include "bibiocr/layout_analyzer.hpp"

#include "onnxruntime_c_api.h"
#include "bibiocr/src/ffi.rs.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bibiocr {
namespace {

constexpr std::uint32_t kModelWidth = 800;
constexpr std::uint32_t kModelHeight = 800;

const std::array<const char*, 25> kLabels = {
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote"};

const std::unordered_set<std::string> kMarkdownIgnoredLabels = {
    "aside_text", "footer", "footer_image", "footnote", "header", "header_image",
    "number"};

const std::unordered_set<std::string> kSkipOrderLabels = {
    "figure_title", "vision_footnote", "image", "chart", "table", "header",
    "header_image", "footer", "footer_image", "footnote", "aside_text"};

const std::unordered_set<std::string> kImageLabels = {
    "image", "figure", "seal"};

const std::unordered_set<std::string> kVisualLabels = {
    "image", "table", "seal", "chart"};

const std::unordered_set<int> kLargeCategoryIds = {3, 5, 6, 15, 17};

struct ImageTensor {
    std::uint32_t original_width{};
    std::uint32_t original_height{};
    std::vector<float> nchw;
};

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}

ImageTensor load_image(const std::filesystem::path& path) {
    const std::string value = path_utf8(path);
    ImageTensorResponse image = bibiocr::load_image_tensor(rust::Str(value));
    return {image.width, image.height,
            std::vector<float>(image.nchw.begin(), image.nchw.end())};
}

#ifdef _WIN32
using LibraryHandle = HMODULE;
LibraryHandle open_library(const std::filesystem::path& path) {
    return LoadLibraryW(path.c_str());
}
void* load_symbol(LibraryHandle library, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(library, name));
}
void close_library(LibraryHandle library) { FreeLibrary(library); }
#else
using LibraryHandle = void*;
LibraryHandle open_library(const std::filesystem::path& path) {
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
void* load_symbol(LibraryHandle library, const char* name) {
    return dlsym(library, name);
}
void close_library(LibraryHandle library) { dlclose(library); }
#endif

class OrtRuntime {
public:
    explicit OrtRuntime(const std::filesystem::path& dll_path) {
        module_ = open_library(dll_path);
        if (module_ == nullptr) {
            throw std::runtime_error("cannot load ONNX Runtime DLL");
        }
        using GetApiBase = const OrtApiBase*(ORT_API_CALL*)();
        const auto get_api_base = reinterpret_cast<GetApiBase>(
            load_symbol(module_, "OrtGetApiBase"));
        if (get_api_base == nullptr) {
            close_library(module_);
            module_ = nullptr;
            throw std::runtime_error("ONNX Runtime DLL does not export OrtGetApiBase");
        }
        api_ = get_api_base()->GetApi(ORT_API_VERSION);
        if (api_ == nullptr) {
            close_library(module_);
            module_ = nullptr;
            throw std::runtime_error("ONNX Runtime API version is incompatible");
        }
    }

    OrtRuntime(const OrtRuntime&) = delete;
    OrtRuntime& operator=(const OrtRuntime&) = delete;

    ~OrtRuntime() {
        if (module_ != nullptr) {
            close_library(module_);
        }
    }

    const OrtApi* api() const noexcept { return api_; }

    void check(OrtStatus* status, const char* operation) const {
        if (status == nullptr) {
            return;
        }
        const std::string message = api_->GetErrorMessage(status);
        api_->ReleaseStatus(status);
        throw std::runtime_error(std::string(operation) + ": " + message);
    }

private:
    LibraryHandle module_ = nullptr;
    const OrtApi* api_ = nullptr;
};

template <typename T>
using OrtPtr = std::unique_ptr<T, std::function<void(T*)>>;

struct Candidate {
    LayoutBlock block;
    int model_order{};
};

float area(const LayoutBlock& block) {
    return std::max(0.0F, block.right - block.left) *
           std::max(0.0F, block.bottom - block.top);
}

float intersection_area(const LayoutBlock& first, const LayoutBlock& second,
                        bool inclusive = false) {
    const float extra = inclusive ? 1.0F : 0.0F;
    const float width = std::max(
        0.0F, std::min(first.right, second.right) -
                  std::max(first.left, second.left) + extra);
    const float height = std::max(
        0.0F, std::min(first.bottom, second.bottom) -
                  std::max(first.top, second.top) + extra);
    return width * height;
}

float iou(const LayoutBlock& first, const LayoutBlock& second) {
    const float intersection = intersection_area(first, second, true);
    const float first_area = (first.right - first.left + 1.0F) *
                             (first.bottom - first.top + 1.0F);
    const float second_area = (second.right - second.left + 1.0F) *
                              (second.bottom - second.top + 1.0F);
    const float denominator = first_area + second_area - intersection;
    return denominator > 0.0F ? intersection / denominator : 0.0F;
}

bool is_contained(const LayoutBlock& inner, const LayoutBlock& outer) {
    const float inner_area = area(inner);
    return inner_area > 0.0F &&
           intersection_area(inner, outer) / inner_area >= 0.9F;
}

std::vector<Candidate> apply_layout_nms(std::vector<Candidate> candidates) {
    std::ranges::stable_sort(candidates, std::greater{},
                             [](const Candidate& value) { return value.block.score; });
    std::vector<Candidate> kept;
    for (Candidate& candidate : candidates) {
        bool suppressed = false;
        for (const Candidate& previous : kept) {
            const float threshold =
                candidate.block.class_id == previous.block.class_id ? 0.6F : 0.98F;
            if (iou(candidate.block, previous.block) >= threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(std::move(candidate));
        }
    }
    return kept;
}

void remove_contained_large_category_boxes(std::vector<Candidate>& candidates) {
    std::vector<bool> removed(candidates.size(), false);
    for (std::size_t outer = 0; outer < candidates.size(); ++outer) {
        if (!kLargeCategoryIds.contains(candidates[outer].block.class_id)) {
            continue;
        }
        for (std::size_t inner = 0; inner < candidates.size(); ++inner) {
            if (inner != outer && !removed[inner] &&
                is_contained(candidates[inner].block, candidates[outer].block)) {
                removed[inner] = true;
            }
        }
    }
    std::vector<Candidate> filtered;
    filtered.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (!removed[index]) {
            filtered.push_back(std::move(candidates[index]));
        }
    }
    candidates = std::move(filtered);
}

std::vector<LayoutBlock> make_parsing_blocks(
    const std::vector<LayoutBlock>& layout_blocks) {
    std::vector<bool> removed(layout_blocks.size(), false);
    for (std::size_t first = 0; first < layout_blocks.size(); ++first) {
        if (removed[first] || layout_blocks[first].label == "reference" ||
            layout_blocks[first].right - layout_blocks[first].left < 6.0F ||
            layout_blocks[first].bottom - layout_blocks[first].top < 6.0F) {
            removed[first] = true;
            continue;
        }
        for (std::size_t second = first + 1; second < layout_blocks.size(); ++second) {
            if (removed[second]) {
                continue;
            }
            const float minimum_area =
                std::min(area(layout_blocks[first]), area(layout_blocks[second]));
            if (minimum_area <= 0.0F) {
                continue;
            }
            const float overlap =
                intersection_area(layout_blocks[first], layout_blocks[second]) /
                minimum_area;
            if (overlap > 0.5F) {
                if (layout_blocks[first].label == "inline_formula") {
                    removed[first] = true;
                    break;
                }
                if (layout_blocks[second].label == "inline_formula") {
                    removed[second] = true;
                    continue;
                }
            }
            if (overlap <= 0.7F) {
                continue;
            }
            const bool visual_pair =
                kVisualLabels.contains(layout_blocks[first].label) ||
                kVisualLabels.contains(layout_blocks[second].label);
            if (visual_pair && layout_blocks[first].label != layout_blocks[second].label) {
                continue;
            }
            if (area(layout_blocks[first]) >= area(layout_blocks[second])) {
                removed[second] = true;
            } else {
                removed[first] = true;
                break;
            }
        }
    }

    std::vector<LayoutBlock> parsing_blocks;
    int ordered_index = 1;
    for (std::size_t index = 0; index < layout_blocks.size(); ++index) {
        if (removed[index]) {
            continue;
        }
        LayoutBlock block = layout_blocks[index];
        block.has_order = !kSkipOrderLabels.contains(block.label);
        block.order = block.has_order ? ordered_index++ : 0;
        parsing_blocks.push_back(std::move(block));
    }
    return parsing_blocks;
}

std::vector<LayoutBlock> run_layout_model(const ImageTensor& image,
                                          const std::filesystem::path& model_path,
                                          const OrtRuntime& runtime) {
    const OrtApi* api = runtime.api();

    OrtEnv* env_raw = nullptr;
    runtime.check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bibiocr", &env_raw),
                  "create ONNX Runtime environment");
    OrtPtr<OrtEnv> env(env_raw, [api](OrtEnv* value) { api->ReleaseEnv(value); });

    OrtSessionOptions* options_raw = nullptr;
    runtime.check(api->CreateSessionOptions(&options_raw), "create ONNX session options");
    OrtPtr<OrtSessionOptions> options(
        options_raw, [api](OrtSessionOptions* value) { api->ReleaseSessionOptions(value); });
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int inference_threads = static_cast<int>(std::max(1U, hardware_threads));
    runtime.check(api->SetIntraOpNumThreads(options.get(), inference_threads),
                  "configure ONNX threads");
    runtime.check(api->SetSessionGraphOptimizationLevel(options.get(), ORT_ENABLE_ALL),
                  "configure ONNX graph optimization");

    OrtSession* session_raw = nullptr;
    runtime.check(api->CreateSession(env.get(), model_path.c_str(), options.get(), &session_raw),
                  "load DocLayoutV3 model");
    OrtPtr<OrtSession> session(session_raw,
                               [api](OrtSession* value) { api->ReleaseSession(value); });

    OrtMemoryInfo* memory_raw = nullptr;
    runtime.check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_raw),
                  "create ONNX memory info");
    OrtPtr<OrtMemoryInfo> memory(
        memory_raw, [api](OrtMemoryInfo* value) { api->ReleaseMemoryInfo(value); });

    std::array<float, 2> image_shape = {static_cast<float>(kModelHeight),
                                        static_cast<float>(kModelWidth)};
    std::array<float, 2> scale_factor = {
        static_cast<float>(kModelHeight) / static_cast<float>(image.original_height),
        static_cast<float>(kModelWidth) / static_cast<float>(image.original_width)};
    const std::array<int64_t, 2> pair_shape = {1, 2};
    const std::array<int64_t, 4> image_dims = {1, 3, kModelHeight, kModelWidth};

    auto make_tensor = [&](void* data, std::size_t bytes, const int64_t* dimensions,
                           std::size_t dimension_count) {
        OrtValue* raw = nullptr;
        runtime.check(api->CreateTensorWithDataAsOrtValue(
                          memory.get(), data, bytes, dimensions, dimension_count,
                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw),
                      "create ONNX input tensor");
        return OrtPtr<OrtValue>(raw, [api](OrtValue* value) { api->ReleaseValue(value); });
    };

    auto shape_tensor = make_tensor(image_shape.data(), sizeof(image_shape), pair_shape.data(),
                                    pair_shape.size());
    auto image_tensor = make_tensor(const_cast<float*>(image.nchw.data()),
                                    image.nchw.size() * sizeof(float), image_dims.data(),
                                    image_dims.size());
    auto scale_tensor = make_tensor(scale_factor.data(), sizeof(scale_factor), pair_shape.data(),
                                    pair_shape.size());

    const std::array<const char*, 3> input_names = {"im_shape", "image", "scale_factor"};
    const std::array<const OrtValue*, 3> input_values = {
        shape_tensor.get(), image_tensor.get(), scale_tensor.get()};
    const std::array<const char*, 1> output_names = {"fetch_name_0"};
    OrtValue* output_raw = nullptr;
    runtime.check(api->Run(session.get(), nullptr, input_names.data(), input_values.data(),
                           input_values.size(), output_names.data(), output_names.size(),
                           &output_raw),
                  "run DocLayoutV3 inference");
    OrtPtr<OrtValue> output(output_raw, [api](OrtValue* value) { api->ReleaseValue(value); });

    OrtTensorTypeAndShapeInfo* output_info_raw = nullptr;
    runtime.check(api->GetTensorTypeAndShape(output.get(), &output_info_raw),
                  "inspect DocLayoutV3 output");
    OrtPtr<OrtTensorTypeAndShapeInfo> output_info(
        output_info_raw,
        [api](OrtTensorTypeAndShapeInfo* value) { api->ReleaseTensorTypeAndShapeInfo(value); });
    std::size_t element_count = 0;
    runtime.check(api->GetTensorShapeElementCount(output_info.get(), &element_count),
                  "read DocLayoutV3 output size");
    if (element_count % 7 != 0) {
        throw std::runtime_error("DocLayoutV3 returned an unexpected output shape");
    }
    void* output_data_raw = nullptr;
    runtime.check(api->GetTensorMutableData(output.get(), &output_data_raw),
                  "read DocLayoutV3 output");
    const auto* rows = static_cast<const float*>(output_data_raw);

    std::vector<Candidate> candidates;
    for (std::size_t offset = 0; offset < element_count; offset += 7) {
        const int class_id = static_cast<int>(rows[offset]);
        const float score = rows[offset + 1];
        if (score <= 0.3F || class_id < 0 ||
            class_id >= static_cast<int>(kLabels.size())) {
            continue;
        }
        const std::string label = kLabels[static_cast<std::size_t>(class_id)];
        LayoutBlock block{
            class_id,
            label,
            score,
            std::clamp(std::round(rows[offset + 2]), 0.0F,
                       static_cast<float>(image.original_width)),
            std::clamp(std::round(rows[offset + 3]), 0.0F,
                       static_cast<float>(image.original_height)),
            std::clamp(std::round(rows[offset + 4]), 0.0F,
                       static_cast<float>(image.original_width)),
            std::clamp(std::round(rows[offset + 5]), 0.0F,
                       static_cast<float>(image.original_height)),
            0,
            false};
        if (block.right > block.left && block.bottom > block.top) {
            candidates.push_back(
                Candidate{std::move(block), static_cast<int>(rows[offset + 6])});
        }
    }
    candidates = apply_layout_nms(std::move(candidates));
    remove_contained_large_category_boxes(candidates);
    std::ranges::stable_sort(candidates, {}, &Candidate::model_order);

    std::vector<LayoutBlock> blocks;
    blocks.reserve(candidates.size());
    int ordered_index = 1;
    for (Candidate& candidate : candidates) {
        candidate.block.has_order = !kSkipOrderLabels.contains(candidate.block.label);
        candidate.block.order = candidate.block.has_order ? ordered_index++ : 0;
        blocks.push_back(std::move(candidate.block));
    }
    return blocks;
}

}  // namespace

LayoutAnalyzer::LayoutAnalyzer(std::filesystem::path model_path,
                               std::filesystem::path runtime_dll)
    : model_path_(std::move(model_path)), runtime_dll_(std::move(runtime_dll)) {
    if (!std::filesystem::is_regular_file(model_path_)) {
        throw std::runtime_error("DocLayoutV3 model does not exist");
    }
    if (!std::filesystem::is_regular_file(runtime_dll_)) {
        throw std::runtime_error("ONNX Runtime DLL does not exist");
    }
}

std::vector<LayoutBlock> LayoutAnalyzer::analyze(
    const std::filesystem::path& image_path) const {
    LayoutAnalysis analysis = analyze_document(image_path);
    std::erase_if(analysis.parsing_blocks, [](const LayoutBlock& block) {
        return kMarkdownIgnoredLabels.contains(block.label) ||
               kImageLabels.contains(block.label);
    });
    return analysis.parsing_blocks;
}

LayoutAnalysis LayoutAnalyzer::analyze_document(
    const std::filesystem::path& image_path) const {
    if (!std::filesystem::is_regular_file(image_path)) {
        throw std::runtime_error("input image does not exist");
    }
    const ImageTensor image = load_image(image_path);
    const OrtRuntime runtime(runtime_dll_);
    LayoutAnalysis result;
    result.width = static_cast<int>(image.original_width);
    result.height = static_cast<int>(image.original_height);
    result.layout_blocks = run_layout_model(image, model_path_, runtime);
    result.parsing_blocks = make_parsing_blocks(result.layout_blocks);
    return result;
}

}  // namespace bibiocr
