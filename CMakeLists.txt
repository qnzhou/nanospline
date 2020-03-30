cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 14)

project(nanospline)

option(HIGH_DEGREE_SUPPORT "With high degree polynomial solver support" OFF)

include(FetchContent)
include(cmake/Eigen3.cmake)
include(cmake/Catch2.cmake)
include(cmake/sanitizer-cmake.cmake)

file(GLOB INC_FILES "${PROJECT_SOURCE_DIR}/include/nanospline/*.h"
    "${PROJECT_SOURCE_DIR}/include/nanospline/internal/*.h")
file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_library(nanospline STATIC ${SRC_FILES} ${INC_FILES})
target_link_libraries(nanospline PUBLIC Eigen3::Eigen)
target_include_directories(nanospline PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

if (HIGH_DEGREE_SUPPORT)
    target_compile_definitions(nanospline PRIVATE
        -DHIGH_DEGREE_SUPPORT)
endif()

if(NOT MSVC)
    target_compile_options(nanospline PRIVATE
        -Wconversion -Wall -Werror)
else()
    target_compile_options(nanospline PRIVATE
        -D_USE_MATH_DEFINES)
endif()

add_sanitizers(nanospline)


file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
add_executable(nanospline_test ${TEST_FILES})
target_link_libraries(nanospline_test nanospline Catch2::Catch2)

if(NOT MSVC)
    target_compile_options(nanospline_test PRIVATE
        -Wconversion -Wall -Werror)
else()
    target_compile_options(nanospline_test PRIVATE
        -D_USE_MATH_DEFINES)
endif()

add_sanitizers(nanospline_test)

add_custom_target(run_unit_tests
    COMMAND nanospline_test
    DEPENDS nanospline_test)

