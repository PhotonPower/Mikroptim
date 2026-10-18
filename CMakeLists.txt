cmake_minimum_required(VERSION 3.25)

project(Mikroptim
    VERSION 0.1.0
    DESCRIPTION "GPU-Raytracing und Systemsynthese für Mikroskopobjektive"
    LANGUAGES CXX)

option(MIKROPTIM_ENABLE_CUDA "CUDA-Backend bauen" OFF)
option(MIKROPTIM_BUILD_TESTS "Tests bauen" ON)
option(MIKROPTIM_WARNINGS_AS_ERRORS "Warnungen als Fehler behandeln" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------------------------------
# CUDA (siehe docs/adr/0002-cuda-12.md und docs/hardware.md)
# --------------------------------------------------------------------------
if(MIKROPTIM_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Tesla P100 = sm_60, RTX 3090 = sm_86, PTX für 86 als Rückfall
        set(CMAKE_CUDA_ARCHITECTURES "60-real;86-real;86-virtual")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
        message(FATAL_ERROR
            "CUDA ${CUDAToolkit_VERSION} unterstützt die Tesla P100 (sm_60) nicht mehr. "
            "Bitte CUDA 12.x verwenden (siehe docs/adr/0002-cuda-12.md).")
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# --------------------------------------------------------------------------
# Warnungen und Gleitkomma-Regeln (keine fast-math-Optionen, siehe ADR-0003)
# --------------------------------------------------------------------------
add_library(mikroptim_warnings INTERFACE)
if(MSVC)
    target_compile_options(mikroptim_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /fp:precise /utf-8>)
    if(MIKROPTIM_WARNINGS_AS_ERRORS)
        target_compile_options(mikroptim_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
else()
    target_compile_options(mikroptim_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow>)
    if(MIKROPTIM_WARNINGS_AS_ERRORS)
        target_compile_options(mikroptim_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
endif()

# --------------------------------------------------------------------------
# Externe Abhängigkeiten (feste Versionen, siehe ADR-0001)
# --------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

# --------------------------------------------------------------------------
# Kern (header-only, host + device)
# --------------------------------------------------------------------------
add_library(mikroptim_core INTERFACE)
add_library(Mikroptim::core ALIAS mikroptim_core)
target_include_directories(mikroptim_core INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mikroptim_core INTERFACE cxx_std_20)

# --------------------------------------------------------------------------
# Datenmodell und IO (nur Host; nlohmann/json bleibt privat, siehe docs/architektur.md)
# --------------------------------------------------------------------------
add_library(mikroptim_model STATIC
    src/model/system.cpp
    src/io/system_json.cpp)
add_library(Mikroptim::model ALIAS mikroptim_model)
target_link_libraries(mikroptim_model
    PUBLIC mikroptim_core
    PRIVATE nlohmann_json::nlohmann_json mikroptim_warnings)

# --------------------------------------------------------------------------
# Glasmodell (Host): Glaskataloge, Brechzahltabellen
# --------------------------------------------------------------------------
add_library(mikroptim_glass STATIC
    src/glass/catalog.cpp
    src/glass/index_table.cpp)
add_library(Mikroptim::glass ALIAS mikroptim_glass)
target_link_libraries(mikroptim_glass
    PUBLIC mikroptim_core
    PRIVATE nlohmann_json::nlohmann_json mikroptim_warnings)

# --------------------------------------------------------------------------
# CUDA-Backend
# --------------------------------------------------------------------------
if(MIKROPTIM_ENABLE_CUDA)
    add_library(mikroptim_cuda STATIC
        src/cuda/device_info.cu
        src/cuda/dispersion_eval.cu
        src/cuda/smoke_trace.cu)
    add_library(Mikroptim::cuda ALIAS mikroptim_cuda)
    target_link_libraries(mikroptim_cuda PUBLIC mikroptim_core PRIVATE CUDA::cudart mikroptim_warnings)
    target_compile_definitions(mikroptim_cuda PUBLIC MIKROPTIM_HAS_CUDA=1)
    target_compile_options(mikroptim_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror=cross-execution-space-call>)
endif()

# --------------------------------------------------------------------------
# Kommandozeilenprogramm
# --------------------------------------------------------------------------
add_executable(mikroptim src/cli/main.cpp)
target_link_libraries(mikroptim PRIVATE mikroptim_core mikroptim_model mikroptim_warnings)
if(MIKROPTIM_ENABLE_CUDA)
    target_link_libraries(mikroptim PRIVATE mikroptim_cuda)
endif()

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(MIKROPTIM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
