
cmake_minimum_required(VERSION 3.16)

project(RoboFDM_py)

set(CMAKE_VERBOSE_MAKEFILE ON)

IF (WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(MSVC_RUNTIME "dynamic")
ENDIF ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

find_package(CGAL CONFIG REQUIRED)
# Don't let CGAL override flags
set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS
    TRUE
    CACHE BOOL "Force CGAL to maintain CMAKE flags")
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

include(FetchContent)

FetchContent_Declare(
        pybind11_sources
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.2
)

FetchContent_GetProperties(pybind11_sources)

if (NOT pybind11_sources_POPULATED)
    FetchContent_Populate(pybind11_sources)

    add_subdirectory(
            ${pybind11_sources_SOURCE_DIR}
            ${pybind11_sources_BINARY_DIR}
    )
endif ()

 find_package(Eigen3 CONFIG REQUIRED)

# include helper file
include(${CGAL_USE_FILE})

# Boost and its components
find_package(Boost REQUIRED)

find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()


# create python module

add_library(RoboFDM
    MODULE
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


if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "7.0" OR
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7.0")
        target_link_libraries(RoboFDM PUBLIC
                pybind11::module
                Eigen3::Eigen PRIVATE ${OpenMP_libomp_LIBRARY})
    endif()
else()
    target_link_libraries(RoboFDM
            PUBLIC
            pybind11::module
            Eigen3::Eigen
            )
endif()

set_target_properties(RoboFDM
        PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        )
