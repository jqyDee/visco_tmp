cmake_minimum_required(VERSION 3.10)
project(Assignment02)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV found at: ${OpenCV_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_executable(ass02 main.cpp rectification.cpp pyramid.cpp gabor.cpp canny.cpp)

target_include_directories(ass02 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    )
target_link_libraries(ass02 PRIVATE ${OpenCV_LIBRARIES})
