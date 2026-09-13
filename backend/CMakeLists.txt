cmake_minimum_required(VERSION 3.20)
project(bibiocr_pipeline VERSION 0.1.0 LANGUAGES CXX)

add_library(bibiocr_core
    src/config.cpp
    src/document_pipeline.cpp
    src/document_result.cpp
    src/layout_analyzer.cpp
    src/llama_cpp.cpp
)
target_compile_features(bibiocr_core PUBLIC cxx_std_20)
target_include_directories(bibiocr_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime
)
target_link_libraries(bibiocr_core PRIVATE windowscodecs ole32 winhttp ws2_32 gdiplus)
if(MSVC)
    target_compile_options(bibiocr_core PRIVATE /W4 /permissive- /utf-8)
endif()

add_executable(bibiocr_pipeline src/main.cpp)
target_compile_features(bibiocr_pipeline PRIVATE cxx_std_20)
target_link_libraries(bibiocr_pipeline PRIVATE bibiocr_core)
configure_file(bibiocr.toml bibiocr.toml COPYONLY)
if(MSVC)
    target_compile_options(bibiocr_pipeline PRIVATE /W4 /permissive- /utf-8)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(config_test tests/config_test.cpp)
    target_compile_features(config_test PRIVATE cxx_std_20)
    target_compile_definitions(config_test PRIVATE
        TEST_CONFIG_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/bibiocr.toml"
    )
    target_link_libraries(config_test PRIVATE bibiocr_core)
    add_test(NAME config COMMAND config_test)

    add_test(
        NAME cli_help
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_help_test.ps1
                -Exe $<TARGET_FILE:bibiocr_pipeline>
    )
    add_test(
        NAME cli_rejects_missing_image
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_missing_image_test.ps1
                -Exe $<TARGET_FILE:bibiocr_pipeline>
    )

    add_executable(layout_inference_test tests/layout_inference_test.cpp)
    target_compile_features(layout_inference_test PRIVATE cxx_std_20)
    target_include_directories(layout_inference_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(layout_inference_test PRIVATE
        TEST_IMAGE_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/demo1.png"
        TEST_CONFIG_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/bibiocr.toml"
    )
    target_link_libraries(layout_inference_test PRIVATE bibiocr_core)
    add_test(NAME layout_inference COMMAND layout_inference_test)

    add_executable(document_pipeline_test tests/document_pipeline_test.cpp)
    target_compile_features(document_pipeline_test PRIVATE cxx_std_20)
    target_compile_definitions(document_pipeline_test PRIVATE
        TEST_IMAGE_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/demo1.png"
        TEST_CONFIG_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/bibiocr.toml"
    )
    target_link_libraries(document_pipeline_test PRIVATE bibiocr_core)
    add_test(NAME document_pipeline COMMAND document_pipeline_test)

    add_executable(artifact_output_test tests/artifact_output_test.cpp)
    target_compile_features(artifact_output_test PRIVATE cxx_std_20)
    target_compile_definitions(artifact_output_test PRIVATE
        TEST_IMAGE_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/demo2.png"
        TEST_IMAGE3_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/demo3.png"
        TEST_CONFIG_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/bibiocr.toml"
    )
    target_link_libraries(artifact_output_test PRIVATE bibiocr_core)
    add_test(NAME artifact_output COMMAND artifact_output_test)

    add_executable(llama_recognizer_test tests/llama_recognizer_test.cpp)
    target_compile_features(llama_recognizer_test PRIVATE cxx_std_20)
    target_link_libraries(llama_recognizer_test PRIVATE bibiocr_core ws2_32)
    add_test(NAME llama_recognizer COMMAND llama_recognizer_test)

    add_test(
        NAME cli_writes_markdown
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_writes_markdown_test.ps1
                -Exe $<TARGET_FILE:bibiocr_pipeline>
                -Image ${CMAKE_CURRENT_SOURCE_DIR}/demo1.png
                -FakeServer ${CMAKE_CURRENT_SOURCE_DIR}/tests/fake_vlm_server.py
    )

    add_executable(llama_server_process_test tests/llama_server_process_test.cpp)
    target_compile_features(llama_server_process_test PRIVATE cxx_std_20)
    target_compile_definitions(llama_server_process_test PRIVATE
        TEST_CONFIG_PATH=L"${CMAKE_CURRENT_SOURCE_DIR}/bibiocr.toml"
    )
    target_link_libraries(llama_server_process_test PRIVATE bibiocr_core)
    add_test(NAME llama_server_process COMMAND llama_server_process_test)
    set_tests_properties(llama_server_process PROPERTIES TIMEOUT 180)

    add_test(
        NAME cli_real_models
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_real_models_test.ps1
                -Exe $<TARGET_FILE:bibiocr_pipeline>
                -Image ${CMAKE_CURRENT_SOURCE_DIR}/demo1.png
    )
    set_tests_properties(cli_real_models PROPERTIES TIMEOUT 900 LABELS slow)
endif()
