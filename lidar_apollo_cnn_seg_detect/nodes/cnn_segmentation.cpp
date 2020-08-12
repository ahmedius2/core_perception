/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
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

#include "cnn_segmentation.h"



CNNSegmentation::CNNSegmentation() : nh_()
{
}

bool CNNSegmentation::init()
{
  std::string proto_file;
  std::string weight_file;

  ros::NodeHandle private_node_handle("~");//to receive args

  //deploy file .prototxt
  if (private_node_handle.getParam("network_definition_file", proto_file))
  {
    ROS_INFO("[%s] network_definition_file: %s", __APP_NAME__, proto_file.c_str());
  } else
  {
    ROS_INFO("[%s] No Network Definition File was received. Finishing execution.", __APP_NAME__);
    return false;
  }
  //modelFile .caffemodel
  if (private_node_handle.getParam("pretrained_model_file", weight_file))
  {
    ROS_INFO("[%s] Pretrained Model File: %s", __APP_NAME__, weight_file.c_str());
  } else
  {
    ROS_INFO("[%s] No Pretrained Model File was received. Finishing execution.", __APP_NAME__);
    return false;
  }


  private_node_handle.param<std::string>("points_src", topic_src_, "points_raw");
  ROS_INFO("[%s] points_src: %s", __APP_NAME__, topic_src_.c_str());

  private_node_handle.param<double>("range", range_, 60.);
  ROS_INFO("[%s] Pretrained Model File: %.2f", __APP_NAME__, range_);

  private_node_handle.param<double>("score_threshold", score_threshold_, 0.6);
  ROS_INFO("[%s] score_threshold: %.2f", __APP_NAME__, score_threshold_);

  private_node_handle.param<int>("width", width_, 512);
  ROS_INFO("[%s] width: %d", __APP_NAME__, width_);

  private_node_handle.param<int>("height", height_, 512);
  ROS_INFO("[%s] height: %d", __APP_NAME__, height_);

  private_node_handle.param<bool>("use_gpu", use_gpu_, true);
  ROS_INFO("[%s] use_gpu: %d", __APP_NAME__, use_gpu_);

  private_node_handle.param<int>("gpu_device_id", gpu_device_id_, 0);
  ROS_INFO("[%s] gpu_device_id: %d", __APP_NAME__, gpu_device_id_);

  private_node_handle.param<bool>("use_dla", use_dla_, false);
  ROS_INFO("[%s] use_dla: %d", __APP_NAME__, use_gpu_);

  private_node_handle.param<int>("dla_id", dla_id_, 0);
  ROS_INFO("[%s] dla_id: %d", __APP_NAME__, dla_id_);

  private_node_handle.param<std::string>("precision", precision_, "FP32");
  ROS_INFO("[%s] gpu/nvdla precision: %s", __APP_NAME__, precision_.c_str());

  private_node_handle.param<std::string>("last_layer_name", last_layer_name_, "class_score");
  ROS_INFO("[%s] last_layer_name: %s", __APP_NAME__, last_layer_name_.c_str());

  /// Instantiate TensorRT net from Caffe model
  //An engine can have multiple execution contexts, allowing one set of weights to be used for multiple overlapping
  // inference tasks. For example, you can process images in parallel CUDA streams using one engine and one context
  // per stream. Each context will be created on the same GPU as the engine.
  trt_runtime_ = nvinfer1::createInferRuntime(gLogger.getTRTLogger());

  bool use_builder=true;

  std::ifstream engine_file("serialized_lidar_apollo_cnn.engine", std::ios::ate | std::ios::binary);
  if(engine_file.good()){
    std::streamsize file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> buffer(file_size);
    if (engine_file.read(buffer.data(), file_size))
    {
      trt_cuda_engine_ = trt_runtime_->deserializeCudaEngine(buffer.data(), buffer.size());
      use_builder=false;
      ROS_INFO("Using serialized model.", __APP_NAME__);

    }
    else{
      ROS_INFO("Read error during deserialization, backing to builder.", __APP_NAME__);
    }
  }

  if(use_builder) {
    trt_builder_ = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    //1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    trt_network_ = trt_builder_->createNetworkV2(0U);
    trt_bconfig_ = trt_builder_->createBuilderConfig();
    trt_caffe_parser_ = nvcaffeparser1::createCaffeParser();

    trt_builder_->setMaxBatchSize(1); // lets keep it simple for now
    trt_builder_->setGpuAllocator(nullptr); // use default

    trt_bconfig_->setMaxWorkspaceSize(1ULL << 32); // Let's make it 1 GB
    trt_bconfig_->setEngineCapability(nvinfer1::EngineCapability::kDEFAULT); // Full tensor capability
    //trt_bconfig_->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

    nvinfer1::DataType dt; // I am manipulating this to save space, maybe keeping it FLOAT will provide max accuracy
    if (precision_.compare("INT8")) {
      trt_bconfig_->setFlag(nvinfer1::BuilderFlag::kFP16);// let it fallback to both FP32(default) and FP16
      trt_bconfig_->setFlag(nvinfer1::BuilderFlag::kINT8);// if INT8 does not work for a layer
      dt = nvinfer1::DataType::kINT8;
    } else if (precision_.compare("FP16")) {
      trt_bconfig_->setFlag(nvinfer1::BuilderFlag::kFP16);
      dt = nvinfer1::DataType::kHALF;
    } else { // FP32
      dt = nvinfer1::DataType::kFLOAT;
    }

    if (use_dla_) { // Use DLA if configured
      trt_bconfig_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
      trt_bconfig_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      trt_bconfig_->setDLACore(dla_id_);
    }


    // Parse caffe network file
    // Set output tensors, caffe:blob, TensorRT:tensor
    const std::vector<std::string> output_tensor_names{
      "instance_pt", "category_score", "confidence_score", "height_pt", "data", "class_score"
    };

    std::vector<nvinfer1::ITensor**> output_tensor_ptrs{
      &instance_pt_blob_, &category_pt_blob_, &confidence_pt_blob_, &height_pt_blob_, &feature_blob_, &class_pt_blob_
    };

    const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = trt_caffe_parser_->parse(
        proto_file.c_str(), weight_file.c_str(), *trt_network_, dt);

    for(int i=0; i < output_tensor_names.size() ; ++i){
      *(output_tensor_ptrs[i]) = blobNameToTensor->find(output_tensor_names[i].c_str());
      CHECK(*(output_tensor_ptrs[i]) != nullptr) << "`" << output_tensor_names[i] << "` layer required";
      trt_network_->markOutput(**(output_tensor_ptrs[i]));
    }

    // print layers and check their dla compability
    for(int i=0; i< trt_network_->getNbLayers(); ++i){
      nvinfer1::ILayer* layer = trt_network_->getLayer(i);
      std::cout << "Layer " << i << ": " << layer->getName() << ",\t\tDLA compatible: "
                << (trt_bconfig_->canRunOnDLA(layer) ? "YES\n" : "NO\n" );
    }

    trt_cuda_engine_ = trt_builder_->buildEngineWithConfig(*trt_network_, *trt_bconfig_);
    if (trt_cuda_engine_ == nullptr) {
      ROS_ERROR("Couldn't build the CUDA Engine!", __APP_NAME__);
      return false;
    }

    nvinfer1::IHostMemory *serializedModel = trt_cuda_engine_->serialize();
    std::ofstream engine_file_out("serialized_lidar_apollo_cnn.engine", std::ios::out | std::ios::binary);
    if (engine_file_out.bad()) {
      ROS_INFO("Couldn't open the file to serialize.", __APP_NAME__);
    }
    else{
      engine_file_out.write((char *) serializedModel->data(), serializedModel->size());
    }
    serializedModel->destroy();
    trt_caffe_parser_->destroy();
    trt_bconfig_->destroy();
    trt_network_->destroy();
    trt_builder_->destroy();
  }

  cluster2d_.reset(new Cluster2D());
  if (!cluster2d_->init(height_, width_, range_))
  {
    ROS_ERROR("[%s] Fail to Initialize cluster2d for CNNSegmentation", __APP_NAME__);
    return false;
  }

  feature_generator_.reset(new FeatureGenerator());
  if (!feature_generator_->init(feature_blob_))
  {
    ROS_ERROR("[%s] Fail to Initialize feature generator for CNNSegmentation", __APP_NAME__);
    return false;
  }

  return true;
}

bool CNNSegmentation::segment(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pc_ptr,
                              const pcl::PointIndices &valid_idx,
                              autoware_msgs::DetectedObjectArray &objects)
{
  int num_pts = static_cast<int>(pc_ptr->points.size());
  if (num_pts == 0)
  {
    ROS_INFO("[%s] Empty point cloud.", __APP_NAME__);
    return true;
  }

  feature_generator_->generate(pc_ptr);

// network forward process
  caffe_net_->Forward();
#ifndef USE_CAFFE_GPU
//  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
//  int gpu_id = 0;
//   caffe::Caffe::SetDevice(gpu_id);
//    caffe::Caffe::set_mode(caffe::Caffe::GPU);
//    caffe::Caffe::DeviceQuery();
#endif

  // clutser points and construct segments/objects
  float objectness_thresh = 0.5;
  bool use_all_grids_for_clustering = true;
  cluster2d_->cluster(*category_pt_blob_, *instance_pt_blob_, pc_ptr,
                      valid_idx, objectness_thresh,
                      use_all_grids_for_clustering);
  cluster2d_->filter(*confidence_pt_blob_, *height_pt_blob_);
  cluster2d_->classify(*class_pt_blob_);
  float confidence_thresh = score_threshold_;
  float height_thresh = 0.5;
  int min_pts_num = 3;
  cluster2d_->getObjects(confidence_thresh, height_thresh, min_pts_num,
                         objects, message_header_);
  return true;
}

void CNNSegmentation::test_run()
{
  std::string in_pcd_file = "uscar_12_1470770225_1470770492_1349.pcd";

  pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(in_pcd_file, *in_pc_ptr);


  pcl::PointIndices valid_idx;
  auto &indices = valid_idx.indices;
  indices.resize(in_pc_ptr->size());
  std::iota(indices.begin(), indices.end(), 0);

  autoware_msgs::DetectedObjectArray objects;
  init();
  segment(in_pc_ptr, valid_idx, objects);


}

void CNNSegmentation::run()
{
  init();

  points_sub_ = nh_.subscribe(topic_src_, 1, &CNNSegmentation::pointsCallback, this);
  points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/detection/lidar_detector/points_cluster", 1);
  objects_pub_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);

  ROS_INFO("[%s] Ready. Waiting for data...", __APP_NAME__);
}

void CNNSegmentation::pointsCallback(const sensor_msgs::PointCloud2 &msg)
{
  std::chrono::system_clock::time_point start, end;
  start = std::chrono::system_clock::now();

  pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(msg, *in_pc_ptr);
  pcl::PointIndices valid_idx;
  auto &indices = valid_idx.indices;
  indices.resize(in_pc_ptr->size());
  std::iota(indices.begin(), indices.end(), 0);
  message_header_ = msg.header;

  autoware_msgs::DetectedObjectArray objects;
  objects.header = message_header_;
  segment(in_pc_ptr, valid_idx, objects);

  pubColoredPoints(objects);

  objects_pub_.publish(objects);

  end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

void CNNSegmentation::pubColoredPoints(const autoware_msgs::DetectedObjectArray &objects_array)
{
  pcl::PointCloud<pcl::PointXYZRGB> colored_cloud;
  for (size_t object_i = 0; object_i < objects_array.objects.size(); object_i++)
  {
    // std::cout << "objct i" << object_i << std::endl;
    pcl::PointCloud<pcl::PointXYZI> object_cloud;
    pcl::fromROSMsg(objects_array.objects[object_i].pointcloud, object_cloud);
    int red = (object_i) % 256;
    int green = (object_i * 7) % 256;
    int blue = (object_i * 13) % 256;

    for (size_t i = 0; i < object_cloud.size(); i++)
    {
      // std::cout << "point i" << i << "/ size: "<<object_cloud.size()  << std::endl;
      pcl::PointXYZRGB colored_point;
      colored_point.x = object_cloud[i].x;
      colored_point.y = object_cloud[i].y;
      colored_point.z = object_cloud[i].z;
      colored_point.r = red;
      colored_point.g = green;
      colored_point.b = blue;
      colored_cloud.push_back(colored_point);
    }
  }
  sensor_msgs::PointCloud2 output_colored_cloud;
  pcl::toROSMsg(colored_cloud, output_colored_cloud);
  output_colored_cloud.header = message_header_;
  points_pub_.publish(output_colored_cloud);
}

CNNSegmentation::~CNNSegmentation() {
  trt_cuda_engine_->destroy();
  trt_runtime_->destroy();
}
