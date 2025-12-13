if(TARGET Eigen3::Eigen)
    return()
endif()

message(STATUS "Third-party (external): creating target 'Eigen3::Eigen'")

include(CPM)
CPMAddPackage(
    NAME eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    DOWNLOAD_ONLY ON
    SYSTEM ON
)
set(EIGEN_INCLUDE_DIRS ${eigen_SOURCE_DIR})

install(DIRECTORY ${EIGEN_INCLUDE_DIRS}/Eigen
    DESTINATION include
)

add_library(Eigen3_Eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)

include(GNUInstallDirs)
target_include_directories(Eigen3_Eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(Eigen3_Eigen INTERFACE EIGEN_MPL2_ONLY)

# Install rules
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME eigen)
set_target_properties(Eigen3_Eigen PROPERTIES EXPORT_NAME Eigen)
install(DIRECTORY ${EIGEN_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS Eigen3_Eigen EXPORT Eigen_Targets)
install(EXPORT Eigen_Targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/eigen NAMESPACE Eigen3::)
