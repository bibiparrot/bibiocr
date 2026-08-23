#include "bibiocr/llama_cpp.hpp"

#include <WinSock2.h>
#include <WS2tcpip.h>
#include <Windows.h>
#include <winhttp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace bibiocr {
namespace {

struct InternetHandleCloser {
    void operator()(void* handle) const noexcept {
        if (handle != nullptr) {
            WinHttpCloseHandle(handle);
        }
    }
};

using InternetHandle = std::unique_ptr<void, InternetHandleCloser>;

std::wstring utf8_to_wide(std::string_view value) {
    if (value.empty()) {
        return {};
    }
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                          static_cast<int>(value.size()), nullptr, 0);
    if (count <= 0) {
        throw std::runtime_error("endpoint is not valid UTF-8");
    }
    std::wstring result(static_cast<std::size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                        static_cast<int>(value.size()), result.data(), count);
    return result;
}

std::string base64(std::span<const std::byte> bytes) {
    static constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve(((bytes.size() + 2) / 3) * 4);
    for (std::size_t index = 0; index < bytes.size(); index += 3) {
        const auto first = std::to_integer<unsigned int>(bytes[index]);
        const auto second = index + 1 < bytes.size()
                                ? std::to_integer<unsigned int>(bytes[index + 1])
                                : 0U;
        const auto third = index + 2 < bytes.size()
                               ? std::to_integer<unsigned int>(bytes[index + 2])
                               : 0U;
        const unsigned int value = (first << 16U) | (second << 8U) | third;
        output.push_back(alphabet[(value >> 18U) & 0x3FU]);
        output.push_back(alphabet[(value >> 12U) & 0x3FU]);
        output.push_back(index + 1 < bytes.size() ? alphabet[(value >> 6U) & 0x3FU] : '=');
        output.push_back(index + 2 < bytes.size() ? alphabet[value & 0x3FU] : '=');
    }
    return output;
}

std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 8);
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (ch < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    output += "\\u00";
                    output.push_back(hex[(ch >> 4U) & 0xFU]);
                    output.push_back(hex[ch & 0xFU]);
                } else {
                    output.push_back(static_cast<char>(ch));
                }
        }
    }
    return output;
}

unsigned int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') return static_cast<unsigned int>(ch - '0');
    if (ch >= 'a' && ch <= 'f') return static_cast<unsigned int>(ch - 'a' + 10);
    if (ch >= 'A' && ch <= 'F') return static_cast<unsigned int>(ch - 'A' + 10);
    throw std::runtime_error("llama.cpp returned invalid JSON Unicode escape");
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7F) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

std::uint32_t parse_hex_quad(std::string_view json, std::size_t& index) {
    if (index + 4 > json.size()) {
        throw std::runtime_error("llama.cpp returned truncated JSON Unicode escape");
    }
    std::uint32_t value = 0;
    for (int digit = 0; digit < 4; ++digit) {
        value = (value << 4U) | hex_value(json[index++]);
    }
    return value;
}

std::string parse_json_string(std::string_view json, std::size_t index) {
    if (index >= json.size() || json[index] != '"') {
        throw std::runtime_error("llama.cpp response content is not a JSON string");
    }
    ++index;
    std::string output;
    while (index < json.size()) {
        const char ch = json[index++];
        if (ch == '"') {
            return output;
        }
        if (ch != '\\') {
            output.push_back(ch);
            continue;
        }
        if (index >= json.size()) {
            break;
        }
        const char escaped = json[index++];
        switch (escaped) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u': {
                std::uint32_t codepoint = parse_hex_quad(json, index);
                if (codepoint >= 0xD800 && codepoint <= 0xDBFF &&
                    index + 6 <= json.size() && json[index] == '\\' &&
                    json[index + 1] == 'u') {
                    index += 2;
                    const std::uint32_t low = parse_hex_quad(json, index);
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) +
                                    (low - 0xDC00U);
                    }
                }
                append_utf8(output, codepoint);
                break;
            }
            default: throw std::runtime_error("llama.cpp returned invalid JSON escape");
        }
    }
    throw std::runtime_error("llama.cpp returned an unterminated JSON string");
}

std::string extract_content(std::string_view json) {
    std::size_t key = json.find("\"content\"");
    while (key != std::string_view::npos) {
        std::size_t colon = json.find(':', key + 9);
        if (colon == std::string_view::npos) {
            break;
        }
        std::size_t value = json.find_first_not_of(" \t\r\n", colon + 1);
        if (value != std::string_view::npos && json[value] == '"') {
            return parse_json_string(json, value);
        }
        key = json.find("\"content\"", colon + 1);
    }
    throw std::runtime_error("llama.cpp response has no message content: " +
                             std::string(json.substr(0, 500)));
}

unsigned short reserve_loopback_port() {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        throw std::runtime_error("cannot initialize Winsock for llama.cpp server");
    }
    const SOCKET socket_handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_handle == INVALID_SOCKET) {
        WSACleanup();
        throw std::runtime_error("cannot allocate a llama.cpp server socket");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    int address_size = sizeof(address);
    const bool ok = bind(socket_handle, reinterpret_cast<sockaddr*>(&address),
                         sizeof(address)) == 0 &&
                    getsockname(socket_handle, reinterpret_cast<sockaddr*>(&address),
                                &address_size) == 0;
    closesocket(socket_handle);
    WSACleanup();
    if (!ok) {
        throw std::runtime_error("cannot choose a port for llama.cpp server");
    }
    return ntohs(address.sin_port);
}

std::wstring quote_windows_argument(std::wstring_view argument) {
    if (argument.find_first_of(L" \t\"") == std::wstring_view::npos) {
        return std::wstring(argument);
    }
    std::wstring output = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t ch : argument) {
        if (ch == L'\\') {
            ++backslashes;
        } else if (ch == L'\"') {
            output.append(backslashes * 2 + 1, L'\\');
            output.push_back(L'\"');
            backslashes = 0;
        } else {
            output.append(backslashes, L'\\');
            backslashes = 0;
            output.push_back(ch);
        }
    }
    output.append(backslashes * 2, L'\\');
    output.push_back(L'\"');
    return output;
}

bool health_ready(unsigned short port) {
    InternetHandle session(WinHttpOpen(L"bibiocr_pipeline/0.1",
                                       WINHTTP_ACCESS_TYPE_NO_PROXY,
                                       WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) return false;
    WinHttpSetTimeouts(session.get(), 500, 500, 500, 500);
    InternetHandle connection(WinHttpConnect(session.get(), L"127.0.0.1", port, 0));
    if (!connection) return false;
    InternetHandle request(WinHttpOpenRequest(connection.get(), L"GET", L"/health", nullptr,
                                              WINHTTP_NO_REFERER,
                                              WINHTTP_DEFAULT_ACCEPT_TYPES, 0));
    if (!request || !WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                        WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.get(), nullptr)) {
        return false;
    }
    DWORD status = 0;
    DWORD size = sizeof(status);
    return WinHttpQueryHeaders(request.get(),
                               WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &status, &size,
                               WINHTTP_NO_HEADER_INDEX) &&
           status == 200;
}

}  // namespace

struct LlamaServerProcess::Impl {
    HANDLE process = nullptr;
    HANDLE job = nullptr;
    std::string endpoint;

    ~Impl() {
        if (job != nullptr) {
            TerminateJobObject(job, 0);
        } else if (process != nullptr) {
            TerminateProcess(process, 0);
        }
        if (process != nullptr) {
            WaitForSingleObject(process, 5'000);
            CloseHandle(process);
        }
        if (job != nullptr) {
            CloseHandle(job);
        }
    }
};

LlamaServerProcess::LlamaServerProcess(const LlamaServerConfig& config)
    : impl_(std::make_unique<Impl>()) {
    if (!std::filesystem::is_regular_file(config.executable)) {
        throw std::runtime_error("llama-server.exe does not exist");
    }
    if (!std::filesystem::is_regular_file(config.model)) {
        throw std::runtime_error("PaddleOCR-VL GGUF model does not exist");
    }
    if (!std::filesystem::is_regular_file(config.mmproj)) {
        throw std::runtime_error("PaddleOCR-VL multimodal projector does not exist");
    }
    if (config.startup_timeout_seconds <= 0) {
        throw std::runtime_error("llama.cpp startup timeout must be positive");
    }

    const unsigned short port = reserve_loopback_port();
    impl_->endpoint = "http://127.0.0.1:" + std::to_string(port) + "/v1";
    const std::wstring command =
        quote_windows_argument(config.executable.wstring()) + L" -m " +
        quote_windows_argument(config.model.wstring()) + L" --mmproj " +
        quote_windows_argument(config.mmproj.wstring()) + L" --host 127.0.0.1 --port " +
        std::to_wstring(port) +
        L" --temp 0 --ctx-size 8192 --no-warmup --log-disable";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');

    impl_->job = CreateJobObjectW(nullptr, nullptr);
    if (impl_->job == nullptr) {
        throw std::runtime_error("cannot create llama.cpp process job");
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(impl_->job, JobObjectExtendedLimitInformation, &limits,
                                 sizeof(limits))) {
        throw std::runtime_error("cannot configure llama.cpp process cleanup");
    }

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process_info{};
    const std::wstring working_directory = config.executable.parent_path().wstring();
    if (!CreateProcessW(config.executable.c_str(), mutable_command.data(), nullptr, nullptr,
                        FALSE, CREATE_NO_WINDOW, nullptr, working_directory.c_str(), &startup,
                        &process_info)) {
        throw std::runtime_error("cannot launch llama-server.exe (Windows error " +
                                 std::to_string(GetLastError()) + ")");
    }
    CloseHandle(process_info.hThread);
    impl_->process = process_info.hProcess;
    if (!AssignProcessToJobObject(impl_->job, impl_->process)) {
        throw std::runtime_error("cannot attach llama.cpp server to cleanup job");
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(config.startup_timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        DWORD exit_code = STILL_ACTIVE;
        if (!GetExitCodeProcess(impl_->process, &exit_code) || exit_code != STILL_ACTIVE) {
            throw std::runtime_error("llama-server.exe exited during startup (code " +
                                     std::to_string(exit_code) + ")");
        }
        if (health_ready(port)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    throw std::runtime_error("timed out waiting for llama.cpp server to load the models");
}

LlamaServerProcess::~LlamaServerProcess() = default;
LlamaServerProcess::LlamaServerProcess(LlamaServerProcess&&) noexcept = default;
LlamaServerProcess& LlamaServerProcess::operator=(LlamaServerProcess&&) noexcept = default;

const std::string& LlamaServerProcess::endpoint() const noexcept {
    return impl_->endpoint;
}

struct LlamaCppRecognizer::Impl {
    std::wstring host;
    std::wstring path;
    INTERNET_PORT port{};
    bool secure{};
};

LlamaCppRecognizer::LlamaCppRecognizer(std::string endpoint) : impl_(std::make_unique<Impl>()) {
    const std::wstring wide_endpoint = utf8_to_wide(endpoint);
    URL_COMPONENTS components{};
    components.dwStructSize = sizeof(components);
    components.dwHostNameLength = static_cast<DWORD>(-1);
    components.dwUrlPathLength = static_cast<DWORD>(-1);
    if (!WinHttpCrackUrl(wide_endpoint.c_str(), static_cast<DWORD>(wide_endpoint.size()), 0,
                         &components)) {
        throw std::runtime_error("invalid llama.cpp endpoint URL");
    }
    impl_->host.assign(components.lpszHostName, components.dwHostNameLength);
    impl_->path.assign(components.lpszUrlPath, components.dwUrlPathLength);
    while (impl_->path.size() > 1 && impl_->path.back() == L'/') {
        impl_->path.pop_back();
    }
    impl_->path += L"/chat/completions";
    impl_->port = components.nPort;
    impl_->secure = components.nScheme == INTERNET_SCHEME_HTTPS;
}

LlamaCppRecognizer::~LlamaCppRecognizer() = default;
LlamaCppRecognizer::LlamaCppRecognizer(LlamaCppRecognizer&&) noexcept = default;
LlamaCppRecognizer& LlamaCppRecognizer::operator=(LlamaCppRecognizer&&) noexcept = default;

std::string LlamaCppRecognizer::recognize(std::span<const std::byte> png,
                                          std::string_view prompt) {
    if (png.empty()) {
        throw std::runtime_error("cannot recognize an empty image crop");
    }
    const std::string body =
        "{\"model\":\"PaddleOCR-VL-1.6\",\"temperature\":0,\"max_tokens\":4096,"
        "\"stream\":false,\"messages\":[{\"role\":"
        "\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":"
        "\"data:image/png;base64," +
        base64(png) + "\"}},{\"type\":\"text\",\"text\":\"" + json_escape(prompt) +
        "\"}]}]}";

    InternetHandle session(WinHttpOpen(L"bibiocr_pipeline/0.1",
                                       WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                       WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) throw std::runtime_error("WinHttpOpen failed");
    WinHttpSetTimeouts(session.get(), 30'000, 30'000, 30'000, 600'000);
    InternetHandle connection(WinHttpConnect(session.get(), impl_->host.c_str(), impl_->port, 0));
    if (!connection) throw std::runtime_error("cannot connect to llama.cpp server");
    const DWORD flags = impl_->secure ? WINHTTP_FLAG_SECURE : 0;
    InternetHandle request(WinHttpOpenRequest(connection.get(), L"POST", impl_->path.c_str(),
                                              nullptr, WINHTTP_NO_REFERER,
                                              WINHTTP_DEFAULT_ACCEPT_TYPES, flags));
    if (!request) throw std::runtime_error("cannot create llama.cpp request");
    constexpr wchar_t headers[] = L"Content-Type: application/json\r\n";
    if (!WinHttpSendRequest(request.get(), headers, static_cast<DWORD>(-1),
                            const_cast<char*>(body.data()), static_cast<DWORD>(body.size()),
                            static_cast<DWORD>(body.size()), 0) ||
        !WinHttpReceiveResponse(request.get(), nullptr)) {
        throw std::runtime_error("llama.cpp HTTP request failed");
    }

    DWORD status = 0;
    DWORD status_size = sizeof(status);
    WinHttpQueryHeaders(request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
                        WINHTTP_NO_HEADER_INDEX);
    std::string response;
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.get(), &available)) {
            throw std::runtime_error("cannot read llama.cpp response");
        }
        if (available == 0) break;
        const std::size_t offset = response.size();
        response.resize(offset + available);
        DWORD received = 0;
        if (!WinHttpReadData(request.get(), response.data() + offset, available, &received)) {
            throw std::runtime_error("cannot read llama.cpp response body");
        }
        response.resize(offset + received);
    }
    if (status < 200 || status >= 300) {
        throw std::runtime_error("llama.cpp returned HTTP " + std::to_string(status) + ": " +
                                 response.substr(0, 500));
    }
    return extract_content(response);
}

}  // namespace bibiocr
