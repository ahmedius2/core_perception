cmake_minimum_required(VERSION 2.8.12)
project(vision_tkdnn_detect)

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

catkin_package()

#set(CMAKE_CXX_FLAGS "-O3 -g -Wall ${CMAKE_CXX_FLAGS}")

AW_CHECK_CUDA()

if(USE_CUDA)
#  list(APPEND CUDA_NVCC_FLAGS "--std=c++11 -I$${PROJECT_SOURCE_DIR}/include -DGPU -arch=sm_72")
#  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

  #ros node
  cuda_add_executable(vision_tkdnn_detect
    src/vision_tkdnn_detect_node.cpp
    src/vision_tkdnn_detect.cpp
    include/vision_tkdnn_detect.h
  )

  target_compile_definitions(vision_tkdnn_detect PUBLIC -DGPU)

  target_include_directories(vision_tkdnn_detect PRIVATE
    ${OPENCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    /shared/tkDNN/include
  )

  target_link_libraries(vision_tkdnn_detect
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    /shared/tkDNN/build/libkernels.so
    /shared/tkDNN/build/libtkDNN.so
  )
  add_dependencies(vision_tkdnn_detect
    ${catkin_EXPORTED_TARGETS}
  )
  install(
    TARGETS vision_tkdnn_detect
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
else()
  message(ERROR "Cannot build vision_tkdnn_detect without CUDA")
endif()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY serialized_models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/serialized_models
  PATTERN ".svn" EXCLUDE
)

#install(DIRECTORY names/
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/names
#  PATTERN ".svn" EXCLUDE
#)

