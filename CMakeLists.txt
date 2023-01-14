# Disable in-source builds to prevent source tree corruption
if ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.21)

project(SysIdOpt)

# Set default build type to release with debug info (i.e. release mode
# optimizations are performed, but debug info still exists).
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

# Generate compile_commands.json by default
if (NOT CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS "YES" CACHE STRING "" FORCE)
endif()

find_package(Eigen3)
find_package(Sleipnir)
find_package(fmt)
find_package(wpimath)

file(GLOB_RECURSE SysIdOpt_src src/*.cpp)
add_executable(SysIdOpt ${SysIdOpt_src})

target_compile_options(SysIdOpt PRIVATE -Wall -pedantic -Wextra -Werror -Wno-unused-parameter)
target_compile_features(SysIdOpt PUBLIC cxx_std_20)
target_link_libraries(SysIdOpt PRIVATE Eigen3::Eigen Sleipnir::Sleipnir fmt wpimath)
