cmake_minimum_required(VERSION 2.8.12)
project(vision_trt_yolov3_detect)

set(ENV{AUTOWARE_COMPILE_WITH_CUDA} 1)

find_package(autoware_build_flags REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  autoware_config_msgs
  autoware_msgs
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

if(NVINFER AND NVPARSERS AND NVONNXPARSERS)
  message("TensorRT is available!")
  message("NVINFER: ${NVINFER}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVONNXPARSERS: ${NVONNXPARSERS}")
  set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()


catkin_package()

#set(CMAKE_CXX_FLAGS "-O3 -g -Wall ${CMAKE_CXX_FLAGS}")

AW_CHECK_CUDA()

if(USE_CUDA)
  list(APPEND CUDA_NVCC_FLAGS "--std=c++11 -I$${PROJECT_SOURCE_DIR}/include -DGPU -arch=sm_72")
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

  #ros node
  cuda_add_executable(vision_trt_yolov3_detect
    src/vision_trt_yolov3_detect_node.cpp
    src/vision_trt_yolov3_detect.cpp
    src/logger.cpp
    include/vision_trt_yolov3_detect.h
    include/rect_class_score.h
    include/logging.h
    include/logger.h
    include/buffers.h
    include/common.h
    include/half.h
  )

  target_compile_definitions(vision_trt_yolov3_detect PUBLIC -DGPU)

  target_include_directories(vision_trt_yolov3_detect PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
  )

  target_link_libraries(vision_trt_yolov3_detect
    ${NVINFER}
    ${NVONNXPARSERS}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    cuda
  )
  add_dependencies(vision_trt_yolov3_detect
    ${catkin_EXPORTED_TARGETS}
  )
  install(
    TARGETS vision_trt_yolov3_detect
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
else()
  message(ERROR "Cannot build vision_trt_yolov3_detect without CUDA")
endif()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
  PATTERN ".svn" EXCLUDE
)
