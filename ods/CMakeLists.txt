cmake_minimum_required(VERSION 2.6)
project(ods)

#cmake .. -DCMAKE_BUILD_TYPE=DEBUG
SET(CMAKE_CXX_FLAGS_DEBUG "-pg -ggdb -Wall -Werror -std=c++0x -DODS_DEBUG")
#cmake .. -DCMAKE_BUILD_TYPE=RELEASE
SET(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Werror -std=c++0x -O3 -fpic -mssse3 -mfpmath=sse -march=core2")

SET(ODS_BINARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_BUILD_TYPE})
SET(APPLICATION_DIR "${CMAKE_BINARY_DIR}/app")
MESSAGE("APP DIR: ${APPLICATION_DIR}")

SET(BUILD_SHARED_LIBS ON)

include_directories(
    src
)

add_subdirectory(src)
#add_subdirectory(swig)
