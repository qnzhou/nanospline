if(TARGET Catch2::Catch2)
    return()
endif()

message(STATUS "Third-party (external): creating target 'Catch2::Catch2'")

option(CATCH_CONFIG_CPP17_STRING_VIEW "Enable support for std::string_view" ON)
option(CATCH_INSTALL_DOCS "Install documentation alongside library" OFF)
option(CATCH_INSTALL_EXTRAS "Install extras alongside library" OFF)

include(CPM)
CPMAddPackage(
    NAME catch2
    GITHUB_REPOSITORY catchorg/Catch2
    GIT_TAG v3.4.0
)

target_compile_features(Catch2 PUBLIC cxx_std_17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    # See https://github.com/catchorg/Catch2/issues/2654
    target_compile_options(Catch2 PUBLIC -Wno-parentheses)
endif()

set_target_properties(Catch2 PROPERTIES FOLDER third_party)
set_target_properties(Catch2WithMain PROPERTIES FOLDER third_party)
include(${catch2_SOURCE_DIR}/extras/Catch.cmake)
