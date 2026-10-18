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
    test_surface_ops.cpp)
target_link_libraries(mikroptim_tests PRIVATE mikroptim_core mikroptim_warnings GTest::gtest_main)

if(MIKROPTIM_ENABLE_CUDA)
    target_sources(mikroptim_tests PRIVATE test_cuda_smoke.cpp)
    target_link_libraries(mikroptim_tests PRIVATE mikroptim_cuda)
endif()

include(GoogleTest)
# PRE_TEST: Testliste erst beim Ausführen ermitteln (CI ohne GPU baut nur).
gtest_discover_tests(mikroptim_tests DISCOVERY_MODE PRE_TEST)
