cmake_minimum_required(VERSION 3.10)

# set the project name
project(pano_generator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

include_directories(include)
add_subdirectory(src)

# add the executable
add_executable(pano_generator main.cpp)

target_link_libraries(pano_generator DataPreparationPipeline ${OpenCV_LIBS})