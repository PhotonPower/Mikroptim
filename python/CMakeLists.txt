# Python-Bindings (M9-T1): Erweiterungsmodul mikroptim._mikroptim mit nanobind, das reine
# Python-Paket python/mikroptim und die pytest-Suite python/tests als ctest. Aufbau, Kopien und
# Sichten: python/README.md und docs/architektur.md, Abschnitt "Python-Bindings".

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind in fester Version (AGENTS.md, Abschnitt 9: erlaubte Abhängigkeit). Das Git-Archiv
# enthält die Untermodule (robin_map) nicht, daher Git mit Untermodulen.
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(_mikroptim
    NB_STATIC
    NB_SUPPRESS_WARNINGS
    src/bind_cli.cpp
    src/bind_eval.cpp
    src/bind_optimize.cpp
    src/bind_system.cpp
    src/bind_trace.cpp
    src/module.cpp)
target_link_libraries(_mikroptim PRIVATE
    mikroptim_core mikroptim_model mikroptim_glass mikroptim_paraxial mikroptim_cpu mikroptim_merit
    mikroptim_batch mikroptim_eval mikroptim_optim mikroptim_cli mikroptim_warnings)
target_include_directories(_mikroptim PRIVATE ${PROJECT_SOURCE_DIR}/src)
if(MIKROPTIM_ENABLE_CUDA)
    # setzt MIKROPTIM_HAS_CUDA: GPU-Pfade (trace, evaluate, optimize, TraceEngine)
    target_link_libraries(_mikroptim PRIVATE mikroptim_cuda)
endif()

# Paketverzeichnis im Build-Baum (für ctest und Entwicklung ohne Wheel): python/mikroptim/*.py
# und das Modul liegen zusammen unter <build>/python/mikroptim. "$<0:>" verhindert das
# Unterverzeichnis pro Konfiguration bei Multi-Config-Generatoren (Visual Studio).
set(MKO_PY_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mikroptim)
set_target_properties(_mikroptim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${MKO_PY_PACKAGE_DIR}$<0:>
    RUNTIME_OUTPUT_DIRECTORY ${MKO_PY_PACKAGE_DIR}$<0:>)
# Auch Unterpakete (z. B. mikroptim/design, M10-T6a) mit ihrer Verzeichnisstruktur
file(GLOB_RECURSE MKO_PY_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/mikroptim
    ${CMAKE_CURRENT_SOURCE_DIR}/mikroptim/*.py)
foreach(rel IN LISTS MKO_PY_SOURCES)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mikroptim/${rel} ${MKO_PY_PACKAGE_DIR}/${rel} COPYONLY)
endforeach()

# Angaben zum Build für das Paket: CUDA-Build und (Windows) Verzeichnis der CUDA-DLLs, das
# mikroptim/__init__.py vor dem Laden des Moduls mit os.add_dll_directory einträgt.
set(MKO_PY_HAS_CUDA False)
set(MKO_PY_CUDA_BIN_DIR "")
if(MIKROPTIM_ENABLE_CUDA)
    set(MKO_PY_HAS_CUDA True)
    set(MKO_PY_CUDA_BIN_DIR ${CUDAToolkit_BIN_DIR})
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/build_info.py.in ${MKO_PY_PACKAGE_DIR}/_build_info.py
    @ONLY)

# Wheel (scikit-build-core, pyproject.toml): Modul und Build-Angaben ins Paketverzeichnis
install(TARGETS _mikroptim LIBRARY DESTINATION mikroptim)
install(FILES ${MKO_PY_PACKAGE_DIR}/_build_info.py DESTINATION mikroptim)

# pytest-Suite als ctest. Übersprungen (Exit-Code 77), wenn pytest oder NumPy fehlen; die
# Vergleiche mit der CLI rufen das Programm `mikroptim` dieses Builds auf.
if(MIKROPTIM_BUILD_TESTS)
    add_test(NAME PythonBindings.pytest
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_pytest.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_tests_properties(PythonBindings.pytest PROPERTIES
        SKIP_RETURN_CODE 77
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};MIKROPTIM_CLI=$<TARGET_FILE:mikroptim>")
endif()
