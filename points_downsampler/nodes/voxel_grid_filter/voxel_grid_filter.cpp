/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include "autoware_config_msgs/ConfigVoxelGridFilter.h"

#include <points_downsampler/PointsDownsamplerInfo.h>

#include <chrono>
#include <sstream>

#include "points_downsampler.h"
#include "sched_server/time_profiling_spinner.h"
#include "sched_server/sched_client.hpp"

#define MAX_MEASUREMENT_RANGE 200.0

ros::Publisher filtered_points_pub;

// Leaf size of VoxelGrid filter.
static double voxel_leaf_size = 2.0;

static ros::Publisher points_downsampler_info_pub;
static points_downsampler::PointsDownsamplerInfo points_downsampler_info_msg;

static std::chrono::time_point<std::chrono::system_clock> filter_start, filter_end;

static bool _output_log = false;
static std::ofstream ofs;
static std::string filename;

static std::string POINTS_TOPIC;
static double measurement_range = MAX_MEASUREMENT_RANGE;

static pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;

static void config_callback(const autoware_config_msgs::ConfigVoxelGridFilter::ConstPtr& input)
{
  voxel_leaf_size = input->voxel_leaf_size;
  measurement_range = input->measurement_range;
}

static void scan_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  std::vector<float> times_ms;

  // 1
  filter_start = std::chrono::system_clock::now();
  pcl::PointCloud<pcl::PointXYZI> scan;
  pcl::fromROSMsg(*input, scan);
  filter_end = std::chrono::system_clock::now();
  times_ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0);

  // 2
  filter_start = std::chrono::system_clock::now();
  if(measurement_range != MAX_MEASUREMENT_RANGE){
    scan = removePointsByRange(scan, 0, measurement_range);
  }
  filter_end = std::chrono::system_clock::now();
  times_ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0);

  // 3
  filter_start = std::chrono::system_clock::now();
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());

  sensor_msgs::PointCloud2 filtered_msg;

  // if voxel_leaf_size < 0.1 voxel_grid_filter cannot down sample (It is specification in PCL)
  if (voxel_leaf_size >= 0.1)
  {
    // Downsampling the velodyne scan using VoxelGrid filter
    // pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_grid_filter.setInputCloud(scan_ptr);
    voxel_grid_filter.filter(*filtered_scan_ptr);
    pcl::toROSMsg(*filtered_scan_ptr, filtered_msg);
  }
  else
  {
    pcl::toROSMsg(*scan_ptr, filtered_msg);
  }

  filter_end = std::chrono::system_clock::now();
  times_ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0);

  // orig clock end point

  // 4
  filter_start = std::chrono::system_clock::now();

  filtered_msg.header = input->header;
  filtered_points_pub.publish(filtered_msg);

  filter_end = std::chrono::system_clock::now();
  times_ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0);

  // 5
  filter_start = std::chrono::system_clock::now();

  points_downsampler_info_msg.header = input->header;
  points_downsampler_info_msg.filter_name = "voxel_grid_filter";
  points_downsampler_info_msg.measurement_range = measurement_range;
  points_downsampler_info_msg.original_points_size = scan.size();
  if (voxel_leaf_size >= 0.1)
  {
    points_downsampler_info_msg.filtered_points_size = filtered_scan_ptr->size();
  }
  else
  {
    points_downsampler_info_msg.filtered_points_size = scan_ptr->size();
  }
  points_downsampler_info_msg.original_ring_size = 0;
  points_downsampler_info_msg.filtered_ring_size = 0;
  points_downsampler_info_msg.exe_time = 1;//std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0;
  points_downsampler_info_pub.publish(points_downsampler_info_msg);

  filter_end = std::chrono::system_clock::now();
  times_ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0);

  // 6
  filter_start = std::chrono::system_clock::now();

  if(_output_log == true){
    if(!ofs){
      std::cerr << "Could not open " << filename << "." << std::endl;
      exit(1);
    }
    ofs << points_downsampler_info_msg.header.seq << ","
      << points_downsampler_info_msg.header.stamp << ","
      << points_downsampler_info_msg.header.frame_id << ","
      << points_downsampler_info_msg.filter_name << ","
      << points_downsampler_info_msg.original_points_size << ","
      << points_downsampler_info_msg.filtered_points_size << ","
      << points_downsampler_info_msg.original_ring_size << ","
      << points_downsampler_info_msg.filtered_ring_size << ","
      << points_downsampler_info_msg.exe_time << ","
      << std::endl;
  }

  filter_end = std::chrono::system_clock::now();
  times_ms.push_back(std::chrono::duration_cast<std::chrono::microseconds>(filter_end - filter_start).count() / 1000.0);

  std::stringstream ss;
  for(float tm : times_ms){
      ss << std::setprecision(3) << tm << " ";
  }
  ROS_INFO("Elapsed time (ms) of six stages are:%s\n", ss.str().c_str());

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "voxel_grid_filter");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  private_nh.getParam("points_topic", POINTS_TOPIC);
  private_nh.getParam("output_log", _output_log);
  if(_output_log == true){
    char buffer[80];
    std::time_t now = std::time(NULL);
    std::tm *pnow = std::localtime(&now);
    std::strftime(buffer,80,"%Y%m%d_%H%M%S",pnow);
    filename = "voxel_grid_filter_" + std::string(buffer) + ".csv";
    ofs.open(filename.c_str(), std::ios::app);
  }
  private_nh.param<double>("measurement_range", measurement_range, MAX_MEASUREMENT_RANGE);
  voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

  // Publishers
  filtered_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_points", 1);
  points_downsampler_info_pub = nh.advertise<points_downsampler::PointsDownsamplerInfo>("/points_downsampler_info", 1);

  // Subscribers
  ros::Subscriber config_sub = nh.subscribe("config/voxel_grid_filter", 1, config_callback,
		  ros::TransportHints().tcpNoDelay());
  ros::Subscriber scan_sub = nh.subscribe(POINTS_TOPIC, 1, scan_callback,
		  ros::TransportHints().tcpNoDelay());

//  ros::spin();
  TimeProfilingSpinner spinner(TimeProfilingSpinner::OperationMode::CHAIN_HEAD);
  spinner.spinAndProfileUntilShutdown();
  spinner.saveProfilingData();

  return 0;
}
