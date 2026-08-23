#pragma once

#include <string>

namespace bibiocr {

struct PipelineResponse;

PipelineResponse run_pipeline(const std::string& image_path,
                              const std::string& output_dir,
                              const std::string& config_path);

}  // namespace bibiocr
