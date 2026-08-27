cmake_minimum_required(VERSION 3.15)
project(qengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Locate the active Python environment (venv, if the caller pointed cmake
# at one via -DPython3_EXECUTABLE=...) and then pybind11's CMake config from
# that same interpreter (`python -m pybind11 --cmakedir`) rather than
# vendoring/FetchContent, so the build always matches whichever
# interpreter/venv is building it.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_result
    )
    if(_pybind11_result EQUAL 0 AND _pybind11_cmakedir)
        set(pybind11_DIR "${_pybind11_cmakedir}")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(qengine
    bindings.cpp
    src/bs_pricer.cpp
    src/svi.cpp
)

set_target_properties(qengine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../core"
)

# Optional standalone C++ unit test binary (no external test framework
# dependency; uses simple assert-based checks in tests/cpp_tests.cpp).
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp_tests.cpp")
    add_executable(qengine_cpp_tests
        tests/cpp_tests.cpp
        src/bs_pricer.cpp
        src/svi.cpp
    )
    add_test(NAME qengine_cpp_tests COMMAND qengine_cpp_tests)
endif()
