# Benchmarks (nur mit MIKROPTIM_ENABLE_CUDA und MIKROPTIM_BUILD_BENCH, siehe bench/README.md).
# Die CI kompiliert sie, führt sie aber nicht aus (keine GPU); der Smoke-Lauf steht in
# tests/CMakeLists.txt.

add_executable(trace_throughput trace_throughput.cpp)
target_link_libraries(trace_throughput PRIVATE
    mikroptim_core mikroptim_model mikroptim_glass mikroptim_paraxial mikroptim_cpu
    mikroptim_batch mikroptim_cuda CUDA::cudart nlohmann_json::nlohmann_json mikroptim_warnings)
