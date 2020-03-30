CMAKE_MINIMUM_REQUIRED(VERSION 3.11)
SET(CMAKE_CXX_STANDARD 14)

PROJECT(nanospline)

OPTION(HIGH_DEGREE_SUPPORT "With high degree polynomial solver support" OFF)
OPTION(WITH_ADDRESS_SANITIZER "Compile with address sanitizer" OFF)

INCLUDE(FetchContent)
INCLUDE(cmake/Eigen3.cmake)
INCLUDE(cmake/Catch2.cmake)

FILE(GLOB INC_FILES "${PROJECT_SOURCE_DIR}/include/nanospline/*.h"
    "${PROJECT_SOURCE_DIR}/include/nanospline/internal/*.h")
FILE(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

ADD_LIBRARY(nanospline STATIC ${SRC_FILES} ${INC_FILES})
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

IF (WITH_ADDRESS_SANITIZER)
    TARGET_COMPILE_OPTIONS(nanospline PUBLIC
        -fno-omit-frame-pointer -fsanitize=address)
    TARGET_LINK_OPTIONS(nanospline PUBLIC
        -fno-omit-frame-pointer -fsanitize=address)
ENDIF()



FILE(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
ADD_EXECUTABLE(nanospline_test ${TEST_FILES})
TARGET_LINK_LIBRARIES(nanospline_test nanospline Catch2::Catch2)

IF(NOT MSVC)
    TARGET_COMPILE_OPTIONS(nanospline_test PRIVATE
        -Wconversion -Wall -Werror)
ELSE()
    TARGET_COMPILE_OPTIONS(nanospline_test PRIVATE
        -D_USE_MATH_DEFINES)
ENDIF()

ADD_CUSTOM_TARGET(run_unit_tests
    COMMAND nanospline_test
    DEPENDS nanospline_test)
