cmake_minimum_required(VERSION 3.16)

if(DEFINED ENV{VCPKG_ROOT})
    set(
        CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file"
    )
endif()

project(RoboFDM_py)

set(CMAKE_VERBOSE_MAKEFILE ON)

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
endif()

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(CGAL CONFIG REQUIRED)

set(
    CGAL_DONT_OVERRIDE_CMAKE_FLAGS
    TRUE
    CACHE BOOL "Force CGAL to maintain CMAKE flags"
)

include(FetchContent)

FetchContent_Declare(
    pybind11_sources
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0
)

FetchContent_GetProperties(pybind11_sources)

if(NOT pybind11_sources_POPULATED)
    FetchContent_Populate(pybind11_sources)

    add_subdirectory(
        ${pybind11_sources_SOURCE_DIR}
        ${pybind11_sources_BINARY_DIR}
    )
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# Python module
# ---------------------------------------------------------------------------

pybind11_add_module(RoboFDM
    src/GeometryTools.cpp
    src/GeometryTools.h
    src/FillHole.cpp
    src/FillHole.h
    src/FillHoleCDT.cpp
    src/FillHoleCDT.h
    src/PlaneCut.cpp
    src/PlaneCut.h
    src/MeshCutEval.cpp
    src/MeshCutEval.h
    src/MeshSupEval.cpp
    src/RoboFDM.cpp
    src/RoboFDM.h
    src/RoboFDMUtils.cpp
)

target_link_libraries(
    RoboFDM
    PRIVATE
    pybind11::module
    Eigen3::Eigen
    CGAL::CGAL
)

if(OPENMP_FOUND)
    target_link_libraries(
        RoboFDM
        PRIVATE
        OpenMP::OpenMP_CXX
    )
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install(
    TARGETS RoboFDM
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
