cmake_minimum_required(VERSION 2.6)
project(css)

#cmake .. -DCMAKE_BUILD_TYPE=DEBUG
SET(CMAKE_CXX_FLAGS_DEBUG "-ggdb -Wall -Werror -std=c++0x")

#cmake .. -DCMAKE_BUILD_TYPE=RELEASE
SET(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Werror -std=c++0x -O3 -fpic -mssse3 -mfpmath=sse -march=core2")

SET(CSS_BINARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})
# SET(APPLICATION_DIR "${CMAKE_BINARY_DIR}/app")
# MESSAGE("APP DIR: ${APPLICATION_DIR}")

SET(BUILD_SHARED_LIBS ON)

# Version Number
set (CSS_VERSION_MAJOR 1)
set (CSS_VERSION_MINOR 0)

configure_file (
       "${PROJECT_SOURCE_DIR}/css_config.h.in"
       "${PROJECT_BINARY_DIR}/css_config.h"
)

# add_subdirectory(css)
add_subdirectory(ods)
add_subdirectory(sos)
# add_subdirectory(doc)

