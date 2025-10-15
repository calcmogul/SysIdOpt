cmake_minimum_required(VERSION 3.21)

# Set default build type to release with debug info
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

project(SysIdOpt)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BUILD_SHARED_LIBS ON CACHE INTERNAL "Build using shared libraries")

include(FetchContent)

set(EIGEN_BUILD_CMAKE_PACKAGE TRUE)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG master
)
FetchContent_MakeAvailable(Eigen3)

set(BUILD_TESTING OFF CACHE INTERNAL "Build testing")
FetchContent_Declare(
    Sleipnir
    GIT_REPOSITORY https://github.com/SleipnirGroup/Sleipnir.git
    GIT_TAG main
)
FetchContent_MakeAvailable(Sleipnir)

set(WITH_CSCORE OFF CACHE INTERNAL "With CSCore")
set(WITH_GUI OFF CACHE INTERNAL "With GUI")
set(WITH_JAVA OFF CACHE INTERNAL "With Java")
set(WITH_NTCORE OFF CACHE INTERNAL "With NTCore")
set(WITH_SIMULATION_MODULES OFF CACHE INTERNAL "With Simulation Modules")
set(WITH_TESTS OFF CACHE INTERNAL "With Tests")
set(WITH_WPIMATH ON CACHE INTERNAL "With WPIMath")
set(WITH_WPILIB OFF CACHE INTERNAL "With WPILib")
FetchContent_Declare(
    wpilib
    GIT_REPOSITORY https://github.com/wpilibsuite/allwpilib.git
    GIT_TAG main
    SYSTEM
)
FetchContent_MakeAvailable(wpilib)

file(GLOB_RECURSE SysIdOpt_src src/*.cpp)
add_executable(SysIdOpt ${SysIdOpt_src})

target_compile_options(SysIdOpt PRIVATE -Wall -Wextra -pedantic)
set_property(TARGET SysIdOpt PROPERTY COMPILE_WARNING_AS_ERROR ON)

target_compile_features(SysIdOpt PUBLIC cxx_std_23)
target_link_libraries(SysIdOpt PRIVATE Eigen3::Eigen Sleipnir::Sleipnir wpimath)
