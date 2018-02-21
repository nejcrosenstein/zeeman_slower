cmake_minimum_required(VERSION 3.8)
project(ZeemanSlower)

add_compile_options("/std:c++latest")
add_compile_options("/arch:AVX2")

set(CPP_FILES ${PROJECT_SOURCE_DIR}/source/main.cpp)

set(HEADER_FILES 
    ${PROJECT_SOURCE_DIR}/externals/randgen/xoroshiro128plus.hpp 
    ${PROJECT_SOURCE_DIR}/source/mathtools.hpp 
    ${PROJECT_SOURCE_DIR}/source/physics.hpp
    ${PROJECT_SOURCE_DIR}/source/coils.hpp
    ${PROJECT_SOURCE_DIR}/source/laser.hpp
    ${PROJECT_SOURCE_DIR}/source/beam.hpp
    ${PROJECT_SOURCE_DIR}/source/histogram.hpp)

add_executable(${PROJECT_NAME} ${CPP_FILES} ${HEADER_FILES})
