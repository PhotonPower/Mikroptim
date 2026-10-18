include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(mikroptim_tests
    test_batch_memory.cpp
    test_batch_residuals.cpp
    test_buchdahl_fit.cpp
    test_cli_trace.cpp
    test_dispersion.cpp
    test_ghost_plate.cpp
    test_glass_catalog.cpp
    test_index_table.cpp
    test_load_system.cpp
    test_paraxial.cpp
    test_pupil_quadrature.cpp
    test_surface_ops.cpp
    test_system_batch.cpp
    test_system_io.cpp
    test_trace.cpp
    test_trace_engine_header.cpp
    test_wavefront.cpp)
target_link_libraries(mikroptim_tests PRIVATE
    mikroptim_core mikroptim_model mikroptim_glass mikroptim_paraxial mikroptim_cpu mikroptim_merit
    mikroptim_batch mikroptim_cli nlohmann_json::nlohmann_json mikroptim_warnings GTest::gtest_main)
# Testdaten unter data/ (z. B. data/systems/)
target_compile_definitions(mikroptim_tests PRIVATE MKO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if(MIKROPTIM_ENABLE_CUDA)
    target_sources(mikroptim_tests PRIVATE
        test_cuda_dispersion.cpp
        test_cuda_quadrature.cpp
        test_cuda_residuals.cpp
        test_cuda_smoke.cpp
        test_cuda_system_batch.cpp
        test_cuda_trace.cpp
        test_cuda_trace_engine.cpp)
    target_link_libraries(mikroptim_tests PRIVATE mikroptim_cuda)
endif()

# Programm `mikroptim` als Ganzes (Unterbefehl trace, docs/cli.md). PASS_REGULAR_EXPRESSION
# ersetzt die Prüfung des Exit-Codes; den prüft test_cli_trace.cpp direkt.
set(MKO_EINZELLINSE ${PROJECT_SOURCE_DIR}/data/systems/einzellinse_bk7.json)
add_test(NAME cli_trace_json
    COMMAND $<TARGET_FILE:mikroptim> trace ${MKO_EINZELLINSE} --json)
set_tests_properties(cli_trace_json PROPERTIES
    PASS_REGULAR_EXPRESSION "\"format\": \"mikroptim\\.trace/1\".*\"focal_length\": 101\\.049.*\"failed_rays\": 0,[\r\n]+  \"warnings\"")
add_test(NAME cli_trace_text
    COMMAND $<TARGET_FILE:mikroptim> trace ${MKO_EINZELLINSE})
set_tests_properties(cli_trace_text PROPERTIES
    PASS_REGULAR_EXPRESSION "System: Einzellinse N-BK7.*Versagte Strahlen: 0 von 510")
add_test(NAME cli_trace_missing_file
    COMMAND $<TARGET_FILE:mikroptim> trace ${PROJECT_SOURCE_DIR}/data/systems/gibt_es_nicht.json)
set_tests_properties(cli_trace_missing_file PROPERTIES WILL_FAIL TRUE)

include(GoogleTest)
# PRE_TEST: Testliste erst beim Ausführen ermitteln (CI ohne GPU baut nur).
gtest_discover_tests(mikroptim_tests DISCOVERY_MODE PRE_TEST)
