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
    src/model/variables.cpp
    src/io/system_json.cpp)
add_library(Mikroptim::model ALIAS mikroptim_model)
target_link_libraries(mikroptim_model
    PUBLIC mikroptim_core
    PRIVATE nlohmann_json::nlohmann_json mikroptim_warnings)

# --------------------------------------------------------------------------
# Glasmodell (Host): Glaskataloge, Brechzahltabellen, Buchdahl-Fit
# --------------------------------------------------------------------------
add_library(mikroptim_glass STATIC
    src/glass/buchdahl_fit.cpp
    src/glass/catalog.cpp
    src/glass/index_table.cpp)
add_library(Mikroptim::glass ALIAS mikroptim_glass)
target_link_libraries(mikroptim_glass
    PUBLIC mikroptim_core
    PRIVATE nlohmann_json::nlohmann_json mikroptim_warnings)

# --------------------------------------------------------------------------
# Paraxialoptik (Host): Brennweite, Schnittweite, Hauptebenen, Petzval, Pupillen
# --------------------------------------------------------------------------
add_library(mikroptim_paraxial STATIC
    src/paraxial/paraxial.cpp)
add_library(Mikroptim::paraxial ALIAS mikroptim_paraxial)
target_link_libraries(mikroptim_paraxial
    PUBLIC mikroptim_core mikroptim_model mikroptim_glass
    PRIVATE mikroptim_warnings)

# --------------------------------------------------------------------------
# CPU-Tracer (Host): Lader, Flächentabelle, CPU-Referenztracer über trace_ray
# --------------------------------------------------------------------------
add_library(mikroptim_cpu STATIC
    src/cpu/cpu_trace.cpp
    src/cpu/host_surface_table.cpp
    src/cpu/load_system.cpp)
add_library(Mikroptim::cpu ALIAS mikroptim_cpu)
target_link_libraries(mikroptim_cpu
    PUBLIC mikroptim_core mikroptim_model mikroptim_glass
    PRIVATE mikroptim_warnings)

# --------------------------------------------------------------------------
# Merit (Host): Wellenfrontauswertung (Referenzsphäre, Kolbenabzug, RMS) auf der CPU
# --------------------------------------------------------------------------
add_library(mikroptim_merit STATIC
    src/merit/wavefront.cpp)
add_library(Mikroptim::merit ALIAS mikroptim_merit)
target_link_libraries(mikroptim_merit
    PUBLIC mikroptim_core mikroptim_model mikroptim_glass mikroptim_paraxial mikroptim_cpu
    PRIVATE mikroptim_warnings)

# --------------------------------------------------------------------------
# SystemBatch (Host): SoA-Layout vieler Varianten, Speicherabschätzung, CPU-Trace und
# CPU-Referenz der Residuen pro Variante
# --------------------------------------------------------------------------
add_library(mikroptim_batch STATIC
    src/batch/batch_extrema.cpp
    src/batch/batch_memory.cpp
    src/batch/batch_merit.cpp
    src/batch/batch_residuals.cpp
    src/batch/batch_trace_cpu.cpp
    src/batch/system_batch.cpp)
add_library(Mikroptim::batch ALIAS mikroptim_batch)
target_link_libraries(mikroptim_batch
    PUBLIC mikroptim_core mikroptim_model mikroptim_glass mikroptim_cpu
    PRIVATE mikroptim_paraxial mikroptim_warnings)

# --------------------------------------------------------------------------
# Optimierer (Host): Variablenabbildung Vektor ↔ Batch, Variantengenerator (ab M3)
# --------------------------------------------------------------------------
add_library(mikroptim_optim STATIC
    src/optim/variable_set.cpp
    src/optim/variant_generator.cpp)
add_library(Mikroptim::optim ALIAS mikroptim_optim)
target_link_libraries(mikroptim_optim
    PUBLIC mikroptim_core mikroptim_model mikroptim_batch
    PRIVATE mikroptim_warnings)

# --------------------------------------------------------------------------
# CUDA-Backend
# --------------------------------------------------------------------------
if(MIKROPTIM_ENABLE_CUDA)
    add_library(mikroptim_cuda STATIC
        src/cuda/batch_trace_check.cu
        src/cuda/device_info.cu
        src/cuda/device_system_batch.cu
        src/cuda/dispersion_eval.cu
        src/cuda/merit.cu
        src/cuda/quadrature_check.cu
        src/cuda/residuals.cu
        src/cuda/smoke_trace.cu
        src/cuda/trace_check.cu
        src/cuda/trace_engine.cu)
    add_library(Mikroptim::cuda ALIAS mikroptim_cuda)
    target_link_libraries(mikroptim_cuda
        PUBLIC mikroptim_core mikroptim_batch
        PRIVATE CUDA::cudart mikroptim_warnings)
    target_compile_definitions(mikroptim_cuda PUBLIC MIKROPTIM_HAS_CUDA=1)
    target_compile_options(mikroptim_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--Werror=cross-execution-space-call>)
endif()

# --------------------------------------------------------------------------
# Kommandozeilenprogramm
# --------------------------------------------------------------------------
# Unterbefehle als Bibliothek, damit die Tests sie direkt aufrufen können (docs/cli.md)
add_library(mikroptim_cli STATIC
    src/cli/trace_command.cpp)
target_include_directories(mikroptim_cli PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mikroptim_cli
    PUBLIC mikroptim_core
    PRIVATE mikroptim_model mikroptim_glass mikroptim_paraxial mikroptim_cpu mikroptim_merit
            nlohmann_json::nlohmann_json mikroptim_warnings)
if(MIKROPTIM_ENABLE_CUDA)
    # `mikroptim trace --gpu` (M2-T5) über die TraceEngine; setzt MIKROPTIM_HAS_CUDA
    target_link_libraries(mikroptim_cli PRIVATE mikroptim_cuda)
endif()

add_executable(mikroptim src/cli/main.cpp)
target_link_libraries(mikroptim PRIVATE mikroptim_core mikroptim_cli mikroptim_warnings)
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
