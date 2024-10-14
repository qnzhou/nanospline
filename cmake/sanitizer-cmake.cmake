if(COMMAND add_sanitizers)
    return()
endif()

message(STATUS "Third-party (external): creating command 'add_sanitizers'")

include(CPM)
CPMAddPackage(
    NAME sanitizers
    GITHUB_REPOSITORY arsenm/sanitizers-cmake
    GIT_TAG 0573e2ea8651b9bb3083f193c41eb086497cc80a
    DOWNLOAD_ONLY Yes
)
set(CMAKE_MODULE_PATH "${sanitizers_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(Sanitizers REQUIRED)
