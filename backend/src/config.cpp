#include "bibiocr/config.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace bibiocr {
namespace {

std::string trim(std::string value) {
    const auto whitespace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(),
                std::find_if_not(value.begin(), value.end(), whitespace));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), whitespace).base(),
                value.end());
    return value;
}

std::string remove_comment(std::string_view line) {
    char quote = 0;
    bool escaped = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char ch = line[index];
        if (quote == '"' && ch == '\\' && !escaped) {
            escaped = true;
            continue;
        }
        if ((ch == '\'' || ch == '"') && !escaped) {
            if (quote == 0) quote = ch;
            else if (quote == ch) quote = 0;
        } else if (ch == '#' && quote == 0) {
            return std::string(line.substr(0, index));
        }
        escaped = false;
    }
    return std::string(line);
}

std::string parse_string(std::string value, std::size_t line_number) {
    value = trim(std::move(value));
    if (value.size() < 2 ||
        !((value.front() == '\'' && value.back() == '\'') ||
          (value.front() == '"' && value.back() == '"'))) {
        throw std::runtime_error("bibiocr.toml line " + std::to_string(line_number) +
                                 ": dependency value must be a quoted string");
    }
    const char quote = value.front();
    std::string result;
    for (std::size_t index = 1; index + 1 < value.size(); ++index) {
        const char ch = value[index];
        if (ch == quote) {
            throw std::runtime_error("bibiocr.toml line " +
                                     std::to_string(line_number) +
                                     ": unexpected quote in dependency value");
        }
        if (quote == '"' && ch == '\\') {
            if (++index + 1 >= value.size()) {
                throw std::runtime_error("bibiocr.toml line " +
                                         std::to_string(line_number) +
                                         ": invalid string escape");
            }
            switch (value[index]) {
                case '\\': result.push_back('\\'); break;
                case '"': result.push_back('"'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default:
                    throw std::runtime_error("bibiocr.toml line " +
                                             std::to_string(line_number) +
                                             ": unsupported string escape");
            }
        } else {
            result.push_back(ch);
        }
    }
    if (result.empty()) {
        throw std::runtime_error("bibiocr.toml line " + std::to_string(line_number) +
                                 ": dependency path is empty");
    }
    return result;
}

std::filesystem::path utf8_path(const std::string& value) {
    return std::filesystem::path(
        std::u8string(reinterpret_cast<const char8_t*>(value.data()), value.size()));
}

std::filesystem::path resolve_path(const std::filesystem::path& runtime_directory,
                                   const std::string& value) {
    std::filesystem::path result = utf8_path(value);
    if (result.is_relative()) result = runtime_directory / result;
    return std::filesystem::absolute(result).lexically_normal();
}

}  // namespace

AppConfig load_config(const std::filesystem::path& config_path) {
    if (!std::filesystem::is_regular_file(config_path)) {
        throw std::runtime_error("configuration file does not exist: " +
                                 config_path.string());
    }
    std::ifstream input(config_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open bibiocr.toml");

    const std::array<std::string, 5> required = {
        "vlm_model", "mmproj", "layout_model", "ort_dll", "llama_server"};
    std::unordered_map<std::string, std::string> values;
    std::string section;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number == 1 && line.starts_with("\xEF\xBB\xBF")) line.erase(0, 3);
        line = trim(remove_comment(line));
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            if (section != "dependencies" && section != "tools") {
                throw std::runtime_error("bibiocr.toml line " +
                                         std::to_string(line_number) +
                                         ": unknown section [" + section + "]");
            }
            continue;
        }
        if (section == "tools") {
            const std::size_t equals = line.find('=');
            if (equals == std::string::npos) {
                throw std::runtime_error("bibiocr.toml line " +
                                         std::to_string(line_number) +
                                         ": expected key = 'path'");
            }
            const std::string key = trim(line.substr(0, equals));
            if (key != "pandoc") {
                throw std::runtime_error("bibiocr.toml line " +
                                         std::to_string(line_number) +
                                         ": unknown tool key " + key);
            }
            (void)parse_string(line.substr(equals + 1), line_number);
            continue;
        }
        if (section != "dependencies") {
            throw std::runtime_error("bibiocr.toml line " +
                                     std::to_string(line_number) +
                                     ": dependency must be inside [dependencies]");
        }
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("bibiocr.toml line " +
                                     std::to_string(line_number) +
                                     ": expected key = 'path'");
        }
        const std::string key = trim(line.substr(0, equals));
        if (std::ranges::find(required, key) == required.end()) {
            throw std::runtime_error("bibiocr.toml line " +
                                     std::to_string(line_number) +
                                     ": unknown dependency key " + key);
        }
        if (values.contains(key)) {
            throw std::runtime_error("bibiocr.toml line " +
                                     std::to_string(line_number) +
                                     ": duplicate dependency key " + key);
        }
        values.emplace(key, parse_string(line.substr(equals + 1), line_number));
    }

    for (const std::string& key : required) {
        if (!values.contains(key)) {
            throw std::runtime_error("bibiocr.toml is missing dependencies." + key);
        }
    }
    const std::filesystem::path runtime_directory =
        std::filesystem::absolute(config_path).parent_path();
    AppConfig config{
        resolve_path(runtime_directory, values.at("vlm_model")),
        resolve_path(runtime_directory, values.at("mmproj")),
        resolve_path(runtime_directory, values.at("layout_model")),
        resolve_path(runtime_directory, values.at("ort_dll")),
        resolve_path(runtime_directory, values.at("llama_server"))};
    const std::array<std::pair<std::string_view, const std::filesystem::path*>, 5> files = {{
        {"vlm_model", &config.vlm_model},
        {"mmproj", &config.mmproj},
        {"layout_model", &config.layout_model},
        {"ort_dll", &config.ort_dll},
        {"llama_server", &config.llama_server},
    }};
    for (const auto& [key, path] : files) {
        if (!std::filesystem::is_regular_file(*path)) {
            throw std::runtime_error("configured dependencies." + std::string(key) +
                                     " file does not exist: " + path->string());
        }
    }
    return config;
}

}  // namespace bibiocr
