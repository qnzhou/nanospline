CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
SET(CMAKE_CXX_STANDARD 14)

PROJECT(nanospline)

OPTION(HIGH_DEGREE_SUPPORT, "With high degree polynomial solver support" OFF)

FIND_PACKAGE(Eigen3)
IF (NOT TARGET Eigen3::Eigen)
    ADD_LIBRARY(Eigen3::Eigen INTERFACE IMPORTED)
    TARGET_INCLUDE_DIRECTORIES(Eigen3::Eigen SYSTEM INTERFACE
        ${PROJECT_SOURCE_DIR}/third_party/eigen)
ENDIF()
ADD_SUBDIRECTORY(tests/Catch2)

FILE(GLOB INC_FILES "${PROJECT_SOURCE_DIR}/include/nanospline/*.h"
    "${PROJECT_SOURCE_DIR}/include/nanospline/internal/*.h")
FILE(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

ADD_LIBRARY(nanospline SHARED ${SRC_FILES} ${INC_FILES})
TARGET_LINK_LIBRARIES(nanospline Eigen3::Eigen)
TARGET_INCLUDE_DIRECTORIES(nanospline PUBLIC
    ${PROJECT_SOURCE_DIR}/include)

IF (HIGH_DEGREE_SUPPORT)
    TARGET_COMPILE_DEFINITIONS(nanospline PRIVATE
        -DHIGH_DEGREE_SUPPORT)
ENDIF()

IF(NOT MSVC)
    TARGET_COMPILE_OPTIONS(nanospline PRIVATE
        -Wconversion -Wall -Werror)
ELSE()
    TARGET_COMPILE_OPTIONS(nanospline PRIVATE
        -D_USE_MATH_DEFINES)
ENDIF()



FILE(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
ADD_EXECUTABLE(nanospline_test ${TEST_FILES})
TARGET_LINK_LIBRARIES(nanospline_test nanospline Catch2::Catch2)

ADD_CUSTOM_TARGET(run_unit_tests
    COMMAND nanospline_test
    DEPENDS nanospline_test)
