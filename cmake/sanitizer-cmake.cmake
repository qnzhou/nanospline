include_guard()

FetchContent_Declare(
    sanitizer
    GIT_REPOSITORY https://github.com/arsenm/sanitizers-cmake.git
    GIT_TAG        99e159ec9bc8dd362b08d18436bd40ff0648417b
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(sanitizer)
if(NOT sanitizer_POPULATED)
    FetchContent_Populate(sanitizer)
    set(CMAKE_MODULE_PATH
        "${sanitizer_SOURCE_DIR}/cmake"
        ${CMAKE_MODULE_PATH})
    find_package(Sanitizers REQUIRED)
endif()
