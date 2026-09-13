#include "bibiocr/bridge.hpp"
#include "bibiocr/src/ffi.rs.h"

#include <stdexcept>

namespace bibiocr {

PipelineResponse run_pipeline(const std::string&, const std::string&, const std::string&) {
    throw std::runtime_error(
        "The C++ backend currently depends on Windows WIC, WinHTTP, and process APIs; "
        "the egui Linux build is available, but backend inference is not yet portable.");
}

}  // namespace bibiocr
