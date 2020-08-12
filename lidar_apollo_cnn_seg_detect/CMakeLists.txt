cmake_minimum_required(VERSION 2.8.3)
project(lidar_apollo_cnn_seg_detect)

list(APPEND CMAKE_PREFIX_PATH "/root/autoware.ai/install/ymc:/root/autoware.ai/install/xsens_driver:/root/autoware.ai/install/wf_simulator:/root/autoware.ai/install/lattice_planner:/root/autoware.ai/install/waypoint_planner:/root/autoware.ai/install/waypoint_maker:/root/autoware.ai/install/way_planner:/root/autoware.ai/install/vlg22c_cam:/root/autoware.ai/install/vision_ssd_detect:/root/autoware.ai/install/vision_segment_enet_detect:/root/autoware.ai/install/vision_lane_detect:/root/autoware.ai/install/vision_darknet_detect:/root/autoware.ai/install/vision_beyond_track:/root/autoware.ai/install/vel_pose_diff_checker:/root/autoware.ai/install/vehicle_socket:/root/autoware.ai/install/vehicle_sim_model:/root/autoware.ai/install/vehicle_model:/root/autoware.ai/install/vehicle_gazebo_simulation_launcher:/root/autoware.ai/install/vehicle_gazebo_simulation_interface:/root/autoware.ai/install/vehicle_engage_panel:/root/autoware.ai/install/vehicle_description:/root/autoware.ai/install/trafficlight_recognizer:/root/autoware.ai/install/op_utilities:/root/autoware.ai/install/op_simulation_package:/root/autoware.ai/install/op_local_planner:/root/autoware.ai/install/op_global_planner:/root/autoware.ai/install/lidar_kf_contour_track:/root/autoware.ai/install/op_ros_helpers:/root/autoware.ai/install/ff_waypoint_follower:/root/autoware.ai/install/dp_planner:/root/autoware.ai/install/op_simu:/root/autoware.ai/install/op_planner:/root/autoware.ai/install/op_utility:/root/autoware.ai/install/lidar_euclidean_cluster_detect:/root/autoware.ai/install/vector_map_server:/root/autoware.ai/install/road_occupancy_processor:/root/autoware.ai/install/costmap_generator:/root/autoware.ai/install/object_map:/root/autoware.ai/install/naive_motion_predict:/root/autoware.ai/install/lanelet_aisan_converter:/root/autoware.ai/install/map_file:/root/autoware.ai/install/libvectormap:/root/autoware.ai/install/lane_planner:/root/autoware.ai/install/imm_ukf_pda_track:/root/autoware.ai/install/decision_maker:/root/autoware.ai/install/vector_map:/root/autoware.ai/install/vector_map_msgs:/root/autoware.ai/install/vectacam:/root/autoware.ai/install/udon_socket:/root/autoware.ai/install/twist_generator:/root/autoware.ai/install/twist_gate:/root/autoware.ai/install/twist_filter:/root/autoware.ai/install/twist2odom:/root/autoware.ai/install/tablet_socket:/root/autoware.ai/install/runtime_manager:/root/autoware.ai/install/mqtt_socket:/root/autoware.ai/install/tablet_socket_msgs:/root/autoware.ai/install/system_monitor:/root/autoware.ai/install/state_machine_lib:/root/autoware.ai/install/sound_player:/root/autoware.ai/install/sick_lms5xx:/root/autoware.ai/install/sick_ldmrs_tools:/root/autoware.ai/install/sick_ldmrs_driver:/root/autoware.ai/install/sick_ldmrs_msgs:/root/autoware.ai/install/sick_ldmrs_description:/root/autoware.ai/install/points2image:/root/autoware.ai/install/rosinterface:/root/autoware.ai/install/rosbag_controller:/root/autoware.ai/install/pure_pursuit:/root/autoware.ai/install/points_preprocessor:/root/autoware.ai/install/mpc_follower:/root/autoware.ai/install/lidar_localizer:/root/autoware.ai/install/emergency_handler:/root/autoware.ai/install/autoware_health_checker:/root/autoware.ai/install/as:/root/autoware.ai/install/ros_observer:/root/autoware.ai/install/roi_object_filter:/root/autoware.ai/install/range_vision_fusion:/root/autoware.ai/install/pos_db:/root/autoware.ai/install/points_downsampler:/root/autoware.ai/install/pixel_cloud_fusion:/root/autoware.ai/install/pcl_omp_registration:/root/autoware.ai/install/pc2_downsampler:/root/autoware.ai/install/oculus_socket:/root/autoware.ai/install/obj_db:/root/autoware.ai/install/nmea_navsat:/root/autoware.ai/install/ndt_tku:/root/autoware.ai/install/ndt_gpu:/root/autoware.ai/install/ndt_cpu:/root/autoware.ai/install/multi_lidar_calibrator:/root/autoware.ai/install/microstrain_driver:/root/autoware.ai/install/memsic_imu:/root/autoware.ai/install/marker_downsampler:/root/autoware.ai/install/map_tools:/root/autoware.ai/install/map_tf_generator:/root/autoware.ai/install/log_tools:/root/autoware.ai/install/lidar_shape_estimation:/root/autoware.ai/install/lidar_point_pillars:/root/autoware.ai/install/lidar_naive_l_shape_detect:/root/autoware.ai/install/lidar_fake_perception:/root/autoware.ai/install/lidar_apollo_cnn_seg_detect:/root/autoware.ai/install/libwaypoint_follower:/root/autoware.ai/install/lgsvl_simulator_bridge:/root/autoware.ai/install/lanelet2_extension:/root/autoware.ai/install/kvaser:/root/autoware.ai/install/kitti_launch:/root/autoware.ai/install/kitti_player:/root/autoware.ai/install/kitti_box_publisher:/root/autoware.ai/install/javad_navsat_driver:/root/autoware.ai/install/integrated_viewer:/root/autoware.ai/install/image_processor:/root/autoware.ai/install/hokuyo:/root/autoware.ai/install/graph_tools:/root/autoware.ai/install/gnss_localizer:/root/autoware.ai/install/gnss:/root/autoware.ai/install/glviewer:/root/autoware.ai/install/gazebo_world_description:/root/autoware.ai/install/gazebo_imu_description:/root/autoware.ai/install/gazebo_camera_description:/root/autoware.ai/install/garmin:/root/autoware.ai/install/freespace_planner:/root/autoware.ai/install/fastvirtualscan:/root/autoware.ai/install/ekf_localizer:/root/autoware.ai/install/ds4_msgs:/root/autoware.ai/install/ds4_driver:/root/autoware.ai/install/detected_objects_visualizer:/root/autoware.ai/install/decision_maker_panel:/root/autoware.ai/install/data_preprocessor:/root/autoware.ai/install/custom_msgs:/root/autoware.ai/install/carla_autoware_bridge:/root/autoware.ai/install/calibration_publisher:/root/autoware.ai/install/autoware_system_msgs:/root/autoware.ai/install/autoware_rviz_plugins:/root/autoware.ai/install/autoware_quickstart_examples:/root/autoware.ai/install/autoware_pointgrey_drivers:/root/autoware.ai/install/autoware_driveworks_interface:/root/autoware.ai/install/autoware_connector:/root/autoware.ai/install/autoware_camera_lidar_calibrator:/root/autoware.ai/install/astar_search:/root/autoware.ai/install/amathutils_lib:/root/autoware.ai/install/autoware_msgs:/root/autoware.ai/install/autoware_map_msgs:/root/autoware.ai/install/autoware_launcher_rviz:/root/autoware.ai/install/autoware_launcher:/root/autoware.ai/install/autoware_lanelet2_msgs:/root/autoware.ai/install/autoware_external_msgs:/root/autoware.ai/install/autoware_driveworks_gmsl_interface:/root/autoware.ai/install/autoware_config_msgs:/root/autoware.ai/install/autoware_can_msgs:/root/autoware.ai/install/autoware_build_flags:/root/autoware.ai/install/autoware_bag_tools:/root/autoware.ai/install/adi_driver:/opt/ros/melodic")

find_package(autoware_build_flags REQUIRED PATHS /root/autoware.ai/install/autoware_build_flags/share)
find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  geometry_msgs
  pcl_ros
  roscpp
  sensor_msgs
  tf
  tf_conversions
)

find_package(CUDA)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
)

###CAFFE
set(CAFFE_PATH "$ENV{HOME}/caffe/distribute")

AW_CHECK_CUDA()

if(USE_CUDA AND EXISTS "${CAFFE_PATH}")
  include_directories(${CUDA_INCLUDE_DIRS})

  IF ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "^arm")
    link_directories(/usr/lib/arm-linux-gnueabihf/tegra)
  endif ()

  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CAFFE_PATH}/include
  )

  add_executable(lidar_apollo_cnn_seg_detect
    nodes/lidar_apollo_cnn_seg_detect_node.cpp
    nodes/cnn_segmentation.cpp
    nodes/cluster2d.cpp
    nodes/feature_generator.cpp
    nodes/logger.cpp
  )


  target_link_libraries(lidar_apollo_cnn_seg_detect
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CAFFE_PATH}/lib/libcaffe.so
    /usr/lib/libnvcaffe_parser.so
    /usr/lib/libnvinfer.so
    /usr/lib/libnvparsers.so
    glog
  )
  add_dependencies(lidar_apollo_cnn_seg_detect
    ${catkin_EXPORTED_TARGETS}
  )

  install(TARGETS
    lidar_apollo_cnn_seg_detect
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
  install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    PATTERN ".svn" EXCLUDE
  )
else()
  message("'Caffe' is not installed. 'lidar_apollo_cnn_seg_detect' will not be built.")
endif()
