cmake_minimum_required(VERSION 3.5.0)
project(torchscriptExample)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/libtorch/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/libtorch/lib)

set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/3rdparty/libtorch)

find_package(Torch REQUIRED)
message(STATUS "lib is: " ${TORCH_LIBRARIES})

# opencv
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/opencv-3.2.0/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/opencv-3.2.0/lib)

list(APPEND BASE_LINK_LIBS PUBLIC ${TORCH_LIBRARIES})
list(APPEND BASE_LINK_LIBS PUBLIC opencv_core)
list(APPEND BASE_LINK_LIBS PUBLIC opencv_imgproc)
list(APPEND BASE_LINK_LIBS PUBLIC opencv_imgcodecs)
list(APPEND BASE_LINK_LIBS PUBLIC opencv_highgui)
list(APPEND BASE_LINK_LIBS PUBLIC opencv_ml)
list(APPEND BASE_LINK_LIBS PUBLIC opencv_video)
list(APPEND BASE_LINK_LIBS PUBLIC opencv_videoio)


add_library(TorchWrapper SHARED src/TorchWrapper.cpp)
target_link_libraries(TorchWrapper ${BASE_LINK_LIBS})

add_executable(TorchWrapperTest src/TorchWrapperTest.cpp)
target_link_libraries(TorchWrapperTest ${BASE_LINK_LIBS} TorchWrapper)

add_executable(torchscriptExample main.cpp)
target_link_libraries(torchscriptExample ${BASE_LINK_LIBS})