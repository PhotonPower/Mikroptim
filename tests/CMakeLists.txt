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
    test_batch_merit.cpp
    test_batch_residuals.cpp
    test_buchdahl_fit.cpp
    test_cli_eval.cpp
    test_cli_optimize.cpp
    test_cli_trace.cpp
    test_constraints_schema.cpp
    test_df64.cpp
    test_dispersion.cpp
    test_eval.cpp
    test_fd_jacobian.cpp
    test_ghost_plate.cpp
    test_ghost_diagnosis.cpp
    test_ghost_optimization.cpp
    test_ghost_penalty.cpp
    test_ghost_schema.cpp
    test_glass_catalog.cpp
    test_index_table.cpp
    test_levenberg_marquardt.cpp
    test_load_system.cpp
    test_multi_gpu_split.cpp
    test_paraxial.cpp
    test_paraxial_residuals.cpp
    test_petzval_optimization.cpp
    test_prefilter.cpp
    test_psf.cpp
    test_pupil_quadrature.cpp
    test_surface_ops.cpp
    test_system_batch.cpp
    test_system_io.cpp
    test_trace.cpp
    test_trace_engine_header.cpp
    test_variable_set.cpp
    test_variables_schema.cpp
    test_variant_generator.cpp
    test_wavefront.cpp)
target_link_libraries(mikroptim_tests PRIVATE
    mikroptim_core mikroptim_model mikroptim_glass mikroptim_paraxial mikroptim_cpu mikroptim_merit
    mikroptim_batch mikroptim_eval mikroptim_optim mikroptim_cli nlohmann_json::nlohmann_json mikroptim_warnings GTest::gtest_main)
# Testdaten unter data/ (z. B. data/systems/)
target_compile_definitions(mikroptim_tests PRIVATE MKO_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

if(MIKROPTIM_ENABLE_CUDA)
    target_sources(mikroptim_tests PRIVATE
        test_cuda_df64.cpp
        test_cuda_dispersion.cpp
        test_cuda_eval.cpp
        test_cuda_fd_jacobian.cpp
        test_cuda_ghost_optimization.cpp
        test_cuda_levenberg_marquardt.cpp
        test_cuda_merit.cpp
        test_cuda_multi_trace_engine.cpp
        test_cuda_petzval_optimization.cpp
        test_cuda_prefilter.cpp
        test_cuda_psf.cpp
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
# `mikroptim eval` (M8-T1)
add_test(NAME cli_eval_json
    COMMAND $<TARGET_FILE:mikroptim> eval ${MKO_EINZELLINSE} --json --grid 32)
set_tests_properties(cli_eval_json PROPERTIES
    PASS_REGULAR_EXPRESSION "\"format\": \"mikroptim\\.eval/1\".*\"failed_rays\": 0,[\r\n]+  \"warnings\"")
# `mikroptim eval --psf` (M8-T2)
add_test(NAME cli_eval_psf_json
    COMMAND $<TARGET_FILE:mikroptim> eval ${MKO_EINZELLINSE} --json --grid 32 --psf)
set_tests_properties(cli_eval_psf_json PROPERTIES
    PASS_REGULAR_EXPRESSION "\"psf\": {[\r\n]+    \"fft_size\": 128.*\"strehl_fft\": ")

# `mikroptim trace --gpu` (M2-T5). Exit-Code 3 = GPU-Backend nicht verfügbar (kExitNoGpu).
if(MIKROPTIM_ENABLE_CUDA)
    # Ohne CUDA-Gerät übersprungen (SKIP_RETURN_CODE); lokal auf der GPU ausführen.
    add_test(NAME cli_trace_gpu_json
        COMMAND $<TARGET_FILE:mikroptim> trace ${MKO_EINZELLINSE} --json --gpu)
    set_tests_properties(cli_trace_gpu_json PROPERTIES
        SKIP_RETURN_CODE 3
        PASS_REGULAR_EXPRESSION "\"backend\": \"gpu\".*\"failed_rays\": 0,[\r\n]+  \"warnings\"")
    add_test(NAME cli_eval_gpu_json
        COMMAND $<TARGET_FILE:mikroptim> eval ${MKO_EINZELLINSE} --json --grid 32 --gpu)
    set_tests_properties(cli_eval_gpu_json PROPERTIES
        SKIP_RETURN_CODE 3
        PASS_REGULAR_EXPRESSION "\"backend\": \"gpu\".*\"failed_rays\": 0,[\r\n]+  \"warnings\"")
    # Kein sichtbares Gerät (CUDA_VISIBLE_DEVICES=-1): Meldung statt Ausnahme.
    add_test(NAME cli_trace_gpu_no_device
        COMMAND $<TARGET_FILE:mikroptim> trace ${MKO_EINZELLINSE} --gpu)
    set_tests_properties(cli_trace_gpu_no_device PROPERTIES
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1"
        PASS_REGULAR_EXPRESSION "--gpu: kein CUDA-Ger.+t gefunden")
else()
    add_test(NAME cli_trace_gpu_unavailable
        COMMAND $<TARGET_FILE:mikroptim> trace ${MKO_EINZELLINSE} --gpu)
    set_tests_properties(cli_trace_gpu_unavailable PROPERTIES
        PASS_REGULAR_EXPRESSION "--gpu: ohne CUDA gebaut")
endif()

# `mikroptim optimize` (M3-T5): Biegung der Einzellinse mit Ergebnisdatei im Build-Verzeichnis,
# danach `mikroptim trace` auf der Ergebnisdatei (Reihenfolge über FIXTURES_SETUP/REQUIRED).
set(MKO_BIEGUNG ${PROJECT_SOURCE_DIR}/data/systems/einzellinse_bk7_biegung.json)
set(MKO_BIEGUNG_OPT ${CMAKE_CURRENT_BINARY_DIR}/einzellinse_bk7_biegung_opt.json)
add_test(NAME cli_optimize_text
    COMMAND $<TARGET_FILE:mikroptim> optimize ${MKO_BIEGUNG} -o ${MKO_BIEGUNG_OPT})
set_tests_properties(cli_optimize_text PROPERTIES
    FIXTURES_SETUP cli_optimize_result
    PASS_REGULAR_EXPRESSION "Iterationsprotokoll.*Status +converged.*Ergebnis geschrieben: ")
add_test(NAME cli_optimize_result_trace
    COMMAND $<TARGET_FILE:mikroptim> trace ${MKO_BIEGUNG_OPT})
set_tests_properties(cli_optimize_result_trace PROPERTIES
    FIXTURES_REQUIRED cli_optimize_result
    PASS_REGULAR_EXPRESSION "Brennweite f' +100\\.0000.*Versagte Strahlen: 0 von 10")
add_test(NAME cli_optimize_no_variables
    COMMAND $<TARGET_FILE:mikroptim> optimize ${MKO_EINZELLINSE})
set_tests_properties(cli_optimize_no_variables PROPERTIES
    PASS_REGULAR_EXPRESSION "keine Variablen")
if(MIKROPTIM_ENABLE_CUDA)
    # Ohne CUDA-Gerät übersprungen (SKIP_RETURN_CODE); lokal auf der GPU ausführen.
    add_test(NAME cli_optimize_gpu_json
        COMMAND $<TARGET_FILE:mikroptim> optimize ${MKO_BIEGUNG} --json --quiet --gpu)
    set_tests_properties(cli_optimize_gpu_json PROPERTIES
        SKIP_RETURN_CODE 3
        PASS_REGULAR_EXPRESSION "\"format\": \"mikroptim\\.optimize/1\".*\"backend\": \"gpu\".*\"converged\": true")
    add_test(NAME cli_optimize_gpu_no_device
        COMMAND $<TARGET_FILE:mikroptim> optimize ${MKO_BIEGUNG} --gpu)
    set_tests_properties(cli_optimize_gpu_no_device PROPERTIES
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1"
        PASS_REGULAR_EXPRESSION "--gpu: kein CUDA-Ger.+t gefunden")
else()
    add_test(NAME cli_optimize_gpu_unavailable
        COMMAND $<TARGET_FILE:mikroptim> optimize ${MKO_BIEGUNG} --gpu)
    set_tests_properties(cli_optimize_gpu_unavailable PROPERTIES
        PASS_REGULAR_EXPRESSION "--gpu: ohne CUDA gebaut")
endif()

# Benchmark trace_throughput (M2-T6) als kurzer Smoke-Lauf: 10 Varianten, ein Lauf. Exit-Code 3 =
# kein CUDA-Gerät (übersprungen); die Synthesesysteme dürfen keine versagten Strahlen haben.
if(TARGET trace_throughput)
    add_test(NAME bench_trace_throughput_smoke
        COMMAND $<TARGET_FILE:trace_throughput> --surfaces 10,40 --variants 10 --repeat 1
                --warmup 0 --mode all)
    set_tests_properties(bench_trace_throughput_smoke PROPERTIES
        SKIP_RETURN_CODE 3
        PASS_REGULAR_EXPRESSION "Versagte Strahlen insgesamt: 0 von 40800")
endif()

include(GoogleTest)
# PRE_TEST: Testliste erst beim Ausführen ermitteln (CI ohne GPU baut nur).
gtest_discover_tests(mikroptim_tests DISCOVERY_MODE PRE_TEST)
