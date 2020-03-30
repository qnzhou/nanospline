include_guard()

if (NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.3.7
        GIT_SHALLOW TRUE
    )

    FetchContent_GetProperties(Eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen SYSTEM INTERFACE
            ${eigen_SOURCE_DIR})
    endif()
endif()
