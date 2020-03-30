include_guard()

FIND_PACKAGE(Eigen3)
IF (NOT TARGET Eigen3::Eigen)

    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY git@gitlab.com:libeigen/eigen.git
        GIT_TAG        3.3.7
        GIT_SHALLOW TRUE
    )

    FetchContent_GetProperties(Eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(Eigen)
        add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR})
    endif()

ENDIF()

