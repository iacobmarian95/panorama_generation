SET(OpenCV_DIR "D:/Kits/opencv/build")

find_package(OpenCV 4.5.2 REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

ADD_LIBRARY(DataPreparationPipeline "DataPreparationPipeline.cpp")

target_link_libraries(DataPreparationPipeline ${OpenCV_LIBS})