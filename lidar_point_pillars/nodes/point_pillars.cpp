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

// headers in STL
#include <chrono>
#include <iostream>
#include <type_traits>

// headers in local files
#include "lidar_point_pillars/point_pillars.h"

// clang-format off
PointPillars::PointPillars(const bool reproduce_result_mode, const float score_threshold,
                           const float nms_overlap_threshold, const std::string pfe_onnx_file,
                           const std::string rpn_onnx_file)
  : reproduce_result_mode_(reproduce_result_mode)
  , score_threshold_(score_threshold)
  , nms_overlap_threshold_(nms_overlap_threshold)
  , pfe_onnx_file_(pfe_onnx_file)
  , rpn_onnx_file_(rpn_onnx_file)
  , MAX_NUM_PILLARS_(12000)
  , MAX_NUM_POINTS_PER_PILLAR_(100)
  , PFE_OUTPUT_SIZE_(MAX_NUM_PILLARS_ * 64)
  , GRID_X_SIZE_(432)
  , GRID_Y_SIZE_(496)
  , GRID_Z_SIZE_(1)
  , RPN_INPUT_SIZE_(64 * GRID_X_SIZE_ * GRID_Y_SIZE_)
  , NUM_ANCHOR_X_INDS_(GRID_X_SIZE_ * 0.5)
  , NUM_ANCHOR_Y_INDS_(GRID_Y_SIZE_ * 0.5)
  , NUM_ANCHOR_R_INDS_(2)
  , NUM_ANCHOR_(NUM_ANCHOR_X_INDS_ * NUM_ANCHOR_Y_INDS_ * NUM_ANCHOR_R_INDS_)
  , RPN_BOX_OUTPUT_SIZE_(NUM_ANCHOR_ * 7)
  , RPN_CLS_OUTPUT_SIZE_(NUM_ANCHOR_)
  , RPN_DIR_OUTPUT_SIZE_(NUM_ANCHOR_ * 2)
  , PILLAR_X_SIZE_(0.16f)
  , PILLAR_Y_SIZE_(0.16f)
  , PILLAR_Z_SIZE_(4.0f)
  , MIN_X_RANGE_(0.0f)
  , MIN_Y_RANGE_(-39.68f)
  , MIN_Z_RANGE_(-3.0f)
  , MAX_X_RANGE_(69.12f)
  , MAX_Y_RANGE_(39.68f)
  , MAX_Z_RANGE_(1)
  , BATCH_SIZE_(1)
  , NUM_INDS_FOR_SCAN_(512)
  , NUM_THREADS_(64) // if you change NUM_THREADS_, need to modify NUM_THREADS_MACRO in common.h
  , SENSOR_HEIGHT_(1.73f)
  , ANCHOR_DX_SIZE_(1.6f)
  , ANCHOR_DY_SIZE_(3.9f)
  , ANCHOR_DZ_SIZE_(1.56f)
  , NUM_BOX_CORNERS_(4) // these seem to be 4 features of a point, x, y, z, intensity
  , NUM_OUTPUT_BOX_FEATURE_(7)
{
  if (reproduce_result_mode_)
  {
    preprocess_points_ptr_.reset(new PreprocessPoints(MAX_NUM_PILLARS_, MAX_NUM_POINTS_PER_PILLAR_, GRID_X_SIZE_,
                                                      GRID_Y_SIZE_, GRID_Z_SIZE_, PILLAR_X_SIZE_, PILLAR_Y_SIZE_,
                                                      PILLAR_Z_SIZE_, MIN_X_RANGE_, MIN_Y_RANGE_, MIN_Z_RANGE_,
                                                      NUM_INDS_FOR_SCAN_, NUM_BOX_CORNERS_));
  }
  else
  {
    preprocess_points_cuda_ptr_.reset(new PreprocessPointsCuda(
        NUM_THREADS_, MAX_NUM_PILLARS_, MAX_NUM_POINTS_PER_PILLAR_, NUM_INDS_FOR_SCAN_, GRID_X_SIZE_, GRID_Y_SIZE_,
        GRID_Z_SIZE_, PILLAR_X_SIZE_, PILLAR_Y_SIZE_, PILLAR_Z_SIZE_, MIN_X_RANGE_, MIN_Y_RANGE_, MIN_Z_RANGE_, NUM_BOX_CORNERS_));
  }

  anchor_mask_cuda_ptr_.reset(new AnchorMaskCuda(NUM_INDS_FOR_SCAN_, NUM_ANCHOR_X_INDS_, NUM_ANCHOR_Y_INDS_,
                                                 NUM_ANCHOR_R_INDS_, MIN_X_RANGE_, MIN_Y_RANGE_, PILLAR_X_SIZE_,
                                                 PILLAR_Y_SIZE_, GRID_X_SIZE_, GRID_Y_SIZE_));

  scatter_cuda_ptr_.reset(new ScatterCuda(NUM_THREADS_, MAX_NUM_PILLARS_, GRID_X_SIZE_, GRID_Y_SIZE_));

  const float FLOAT_MIN = std::numeric_limits<float>::lowest();
  const float FLOAT_MAX = std::numeric_limits<float>::max();
  postprocess_cuda_ptr_.reset(new PostprocessCuda(FLOAT_MIN, FLOAT_MAX, NUM_ANCHOR_X_INDS_, NUM_ANCHOR_Y_INDS_,
                                                  NUM_ANCHOR_R_INDS_, score_threshold_, NUM_THREADS_,
                                                  nms_overlap_threshold_, NUM_BOX_CORNERS_, NUM_OUTPUT_BOX_FEATURE_));

  deviceMemoryMalloc();
  initTRT();
  initAnchors();
}
// clang-format on

PointPillars::~PointPillars()
{
  delete[] anchors_px_;
  delete[] anchors_py_;
  delete[] anchors_pz_;
  delete[] anchors_dx_;
  delete[] anchors_dy_;
  delete[] anchors_dz_;
  delete[] anchors_ro_;
  delete[] box_anchors_min_x_;
  delete[] box_anchors_min_y_;
  delete[] box_anchors_max_x_;
  delete[] box_anchors_max_y_;

  GPU_CHECK(cudaFree(dev_pillar_x_in_coors_));
  GPU_CHECK(cudaFree(dev_pillar_y_in_coors_));
  GPU_CHECK(cudaFree(dev_pillar_z_in_coors_));
  GPU_CHECK(cudaFree(dev_pillar_i_in_coors_));
  GPU_CHECK(cudaFree(dev_pillar_count_histo_));

  GPU_CHECK(cudaFree(dev_x_coors_));
  GPU_CHECK(cudaFree(dev_y_coors_));
  GPU_CHECK(cudaFree(dev_num_points_per_pillar_));
  GPU_CHECK(cudaFree(dev_sparse_pillar_map_));

  GPU_CHECK(cudaFree(dev_pillar_x_));
  GPU_CHECK(cudaFree(dev_pillar_y_));
  GPU_CHECK(cudaFree(dev_pillar_z_));
  GPU_CHECK(cudaFree(dev_pillar_i_));

  GPU_CHECK(cudaFree(dev_x_coors_for_sub_shaped_));
  GPU_CHECK(cudaFree(dev_y_coors_for_sub_shaped_));
  GPU_CHECK(cudaFree(dev_pillar_feature_mask_));

  GPU_CHECK(cudaFree(dev_cumsum_along_x_));
  GPU_CHECK(cudaFree(dev_cumsum_along_y_));

  GPU_CHECK(cudaFree(dev_box_anchors_min_x_));
  GPU_CHECK(cudaFree(dev_box_anchors_min_y_));
  GPU_CHECK(cudaFree(dev_box_anchors_max_x_));
  GPU_CHECK(cudaFree(dev_box_anchors_max_y_));
  GPU_CHECK(cudaFree(dev_anchor_mask_));

  GPU_CHECK(cudaFree(pfe_buffers_[0]));
  GPU_CHECK(cudaFree(pfe_buffers_[1]));
  GPU_CHECK(cudaFree(pfe_buffers_[2]));
  GPU_CHECK(cudaFree(pfe_buffers_[3]));
  GPU_CHECK(cudaFree(pfe_buffers_[4]));
  GPU_CHECK(cudaFree(pfe_buffers_[5]));
  GPU_CHECK(cudaFree(pfe_buffers_[6]));
  GPU_CHECK(cudaFree(pfe_buffers_[7]));
  GPU_CHECK(cudaFree(pfe_buffers_[8]));

  GPU_CHECK(cudaFree(rpn_buffers_[0]));
  GPU_CHECK(cudaFree(rpn_buffers_[1]));
  GPU_CHECK(cudaFree(rpn_buffers_[2]));
  GPU_CHECK(cudaFree(rpn_buffers_[3]));

  GPU_CHECK(cudaFree(dev_scattered_feature_));

  GPU_CHECK(cudaFree(dev_anchors_px_));
  GPU_CHECK(cudaFree(dev_anchors_py_));
  GPU_CHECK(cudaFree(dev_anchors_pz_));
  GPU_CHECK(cudaFree(dev_anchors_dx_));
  GPU_CHECK(cudaFree(dev_anchors_dy_));
  GPU_CHECK(cudaFree(dev_anchors_dz_));
  GPU_CHECK(cudaFree(dev_anchors_ro_));
  GPU_CHECK(cudaFree(dev_filtered_box_));
  GPU_CHECK(cudaFree(dev_filtered_score_));
  GPU_CHECK(cudaFree(dev_filtered_dir_));
  GPU_CHECK(cudaFree(dev_box_for_nms_));
  GPU_CHECK(cudaFree(dev_filter_count_));

  pfe_context_->destroy();
  rpn_context_->destroy();

  pfe_runtime_->destroy();
  rpn_runtime_->destroy();
  pfe_engine_->destroy();
  rpn_engine_->destroy();

}

void PointPillars::deviceMemoryMalloc()
{
  // clang-format off
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_x_in_coors_,GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_y_in_coors_,GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_z_in_coors_,GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_i_in_coors_,GRID_Y_SIZE_ * GRID_X_SIZE_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_count_histo_, GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int)));
  // clang-format on

  GPU_CHECK(cudaMalloc((void**)&dev_x_coors_, MAX_NUM_PILLARS_ * sizeof(int)));
  GPU_CHECK(cudaMalloc((void**)&dev_y_coors_, MAX_NUM_PILLARS_ * sizeof(int)));
  GPU_CHECK(cudaMalloc((void**)&dev_num_points_per_pillar_, MAX_NUM_PILLARS_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_sparse_pillar_map_, NUM_INDS_FOR_SCAN_ * NUM_INDS_FOR_SCAN_ * sizeof(int)));

  GPU_CHECK(cudaMalloc((void**)&dev_pillar_x_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_y_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_z_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_i_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));

  // clang-format off
  GPU_CHECK(cudaMalloc((void**)&dev_x_coors_for_sub_shaped_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_y_coors_for_sub_shaped_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_pillar_feature_mask_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  // clang-format on

  // cumsum kernel
  GPU_CHECK(cudaMalloc((void**)&dev_cumsum_along_x_, NUM_INDS_FOR_SCAN_ * NUM_INDS_FOR_SCAN_ * sizeof(int)));
  GPU_CHECK(cudaMalloc((void**)&dev_cumsum_along_y_, NUM_INDS_FOR_SCAN_ * NUM_INDS_FOR_SCAN_ * sizeof(int)));

  // for make anchor mask kernel
  GPU_CHECK(cudaMalloc((void**)&dev_box_anchors_min_x_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_box_anchors_min_y_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_box_anchors_max_x_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_box_anchors_max_y_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_anchor_mask_, NUM_ANCHOR_ * sizeof(int)));

  // for trt inference
  // create GPU buffers and a stream
  GPU_CHECK(cudaMalloc(&pfe_buffers_[0], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[1], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[2], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[3], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[4], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[5], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[6], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[7], MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[8], PFE_OUTPUT_SIZE_ * sizeof(float)));

  GPU_CHECK(cudaMalloc(&rpn_buffers_[0], RPN_INPUT_SIZE_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&rpn_buffers_[1], RPN_BOX_OUTPUT_SIZE_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&rpn_buffers_[2], RPN_CLS_OUTPUT_SIZE_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(&rpn_buffers_[3], RPN_DIR_OUTPUT_SIZE_ * sizeof(float)));

  // for scatter kernel
  GPU_CHECK(cudaMalloc((void**)&dev_scattered_feature_, NUM_THREADS_ * GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(float)));

  // for filter
  GPU_CHECK(cudaMalloc((void**)&dev_anchors_px_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_anchors_py_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_anchors_pz_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_anchors_dx_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_anchors_dy_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_anchors_dz_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_anchors_ro_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_filtered_box_, NUM_ANCHOR_ * NUM_OUTPUT_BOX_FEATURE_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_filtered_score_, NUM_ANCHOR_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_filtered_dir_, NUM_ANCHOR_ * sizeof(int)));
  GPU_CHECK(cudaMalloc((void**)&dev_box_for_nms_, NUM_ANCHOR_ * NUM_BOX_CORNERS_ * sizeof(float)));
  GPU_CHECK(cudaMalloc((void**)&dev_filter_count_, sizeof(int)));
}

void PointPillars::initAnchors()
{
  // allocate memory for anchors
  anchors_px_ = new float[NUM_ANCHOR_];
  anchors_py_ = new float[NUM_ANCHOR_];
  anchors_pz_ = new float[NUM_ANCHOR_];
  anchors_dx_ = new float[NUM_ANCHOR_];
  anchors_dy_ = new float[NUM_ANCHOR_];
  anchors_dz_ = new float[NUM_ANCHOR_];
  anchors_ro_ = new float[NUM_ANCHOR_];
  box_anchors_min_x_ = new float[NUM_ANCHOR_];
  box_anchors_min_y_ = new float[NUM_ANCHOR_];
  box_anchors_max_x_ = new float[NUM_ANCHOR_];
  box_anchors_max_y_ = new float[NUM_ANCHOR_];
  // deallocate these memories in deconstructor

  generateAnchors(anchors_px_, anchors_py_, anchors_pz_, anchors_dx_, anchors_dy_, anchors_dz_, anchors_ro_);

  convertAnchors2BoxAnchors(anchors_px_, anchors_py_, anchors_dx_, anchors_dy_, box_anchors_min_x_, box_anchors_min_y_,
                            box_anchors_max_x_, box_anchors_max_y_);

  putAnchorsInDeviceMemory();
}

void PointPillars::generateAnchors(float* anchors_px_, float* anchors_py_, float* anchors_pz_, float* anchors_dx_,
                                   float* anchors_dy_, float* anchors_dz_, float* anchors_ro_)
{
  // zero clear
  for (size_t i = 0; i < NUM_ANCHOR_; i++)
  {
    anchors_px_[i] = 0;
    anchors_py_[i] = 0;
    anchors_pz_[i] = 0;
    anchors_dx_[i] = 0;
    anchors_dy_[i] = 0;
    anchors_dz_[i] = 0;
    anchors_ro_[i] = 0;
    box_anchors_min_x_[i] = 0;
    box_anchors_min_y_[i] = 0;
    box_anchors_max_x_[i] = 0;
    box_anchors_max_y_[i] = 0;
  }

  float x_stride = PILLAR_X_SIZE_ * 2.0f;
  float y_stride = PILLAR_Y_SIZE_ * 2.0f;
  float x_offset = MIN_X_RANGE_ + PILLAR_X_SIZE_;
  float y_offset = MIN_Y_RANGE_ + PILLAR_Y_SIZE_;

  float anchor_x_count[NUM_ANCHOR_X_INDS_] = { 0 };
  for (size_t i = 0; i < NUM_ANCHOR_X_INDS_; i++)
  {
    anchor_x_count[i] = float(i) * x_stride + x_offset;
  }
  float anchor_y_count[NUM_ANCHOR_Y_INDS_] = { 0 };
  for (size_t i = 0; i < NUM_ANCHOR_Y_INDS_; i++)
  {
    anchor_y_count[i] = float(i) * y_stride + y_offset;
  }

  float anchor_r_count[NUM_ANCHOR_R_INDS_] = { 0, M_PI / 2 };

  // np.meshgrid
  for (size_t y = 0; y < NUM_ANCHOR_Y_INDS_; y++)
  {
    for (size_t x = 0; x < NUM_ANCHOR_X_INDS_; x++)
    {
      for (size_t r = 0; r < NUM_ANCHOR_R_INDS_; r++)
      {
        int ind = y * NUM_ANCHOR_X_INDS_ * NUM_ANCHOR_R_INDS_ + x * NUM_ANCHOR_R_INDS_ + r;
        anchors_px_[ind] = anchor_x_count[x];
        anchors_py_[ind] = anchor_y_count[y];
        anchors_ro_[ind] = anchor_r_count[r];
        anchors_pz_[ind] = -1 * SENSOR_HEIGHT_;
        anchors_dx_[ind] = ANCHOR_DX_SIZE_;
        anchors_dy_[ind] = ANCHOR_DY_SIZE_;
        anchors_dz_[ind] = ANCHOR_DZ_SIZE_;
      }
    }
  }
}

void PointPillars::putAnchorsInDeviceMemory()
{
  // clang-format off
  GPU_CHECK(cudaMemcpy(dev_box_anchors_min_x_, box_anchors_min_x_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_box_anchors_min_y_, box_anchors_min_y_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_box_anchors_max_x_, box_anchors_max_x_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_box_anchors_max_y_, box_anchors_max_y_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  // clang-format on

  GPU_CHECK(cudaMemcpy(dev_anchors_px_, anchors_px_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_py_, anchors_py_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_pz_, anchors_pz_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_dx_, anchors_dx_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_dy_, anchors_dy_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_dz_, anchors_dz_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_ro_, anchors_ro_, NUM_ANCHOR_ * sizeof(float), cudaMemcpyHostToDevice));
}


void PointPillars::convertAnchors2BoxAnchors(float* anchors_px, float* anchors_py, float* anchors_dx, float* anchors_dy,
                                             float* box_anchors_min_x_, float* box_anchors_min_y_,
                                             float* box_anchors_max_x_, float* box_anchors_max_y_)
{
  // flipping box's dimension
  float flipped_anchors_dx[NUM_ANCHOR_] = { 0 };
  float flipped_anchors_dy[NUM_ANCHOR_] = { 0 };
  for (size_t x = 0; x < NUM_ANCHOR_X_INDS_; x++)
  {
    for (size_t y = 0; y < NUM_ANCHOR_Y_INDS_; y++)
    {
      int base_ind = x * NUM_ANCHOR_Y_INDS_ * NUM_ANCHOR_R_INDS_ + y * NUM_ANCHOR_R_INDS_;
      flipped_anchors_dx[base_ind + 0] = ANCHOR_DX_SIZE_;
      flipped_anchors_dy[base_ind + 0] = ANCHOR_DY_SIZE_;
      flipped_anchors_dx[base_ind + 1] = ANCHOR_DY_SIZE_;
      flipped_anchors_dy[base_ind + 1] = ANCHOR_DX_SIZE_;
    }
  }
  for (size_t x = 0; x < NUM_ANCHOR_X_INDS_; x++)
  {
    for (size_t y = 0; y < NUM_ANCHOR_Y_INDS_; y++)
    {
      for (size_t r = 0; r < NUM_ANCHOR_R_INDS_; r++)
      {
        int ind = x * NUM_ANCHOR_Y_INDS_ * NUM_ANCHOR_R_INDS_ + y * NUM_ANCHOR_R_INDS_ + r;
        box_anchors_min_x_[ind] = anchors_px[ind] - flipped_anchors_dx[ind] / 2.0f;
        box_anchors_min_y_[ind] = anchors_py[ind] - flipped_anchors_dy[ind] / 2.0f;
        box_anchors_max_x_[ind] = anchors_px[ind] + flipped_anchors_dx[ind] / 2.0f;
        box_anchors_max_y_[ind] = anchors_py[ind] + flipped_anchors_dy[ind] / 2.0f;
      }
    }
  }
}

void PointPillars::initTRT()
{
  // create a TensorRT model from the onnx model and serialize it to a stream
  nvinfer1::IHostMemory* pfe_trt_model_stream{ nullptr };
  nvinfer1::IHostMemory* rpn_trt_model_stream{ nullptr };
  onnxToTRTModel(pfe_onnx_file_, pfe_trt_model_stream);
  onnxToTRTModel(rpn_onnx_file_, rpn_trt_model_stream);
  if (pfe_trt_model_stream == nullptr || rpn_trt_model_stream == nullptr)
  {//use std:cerr instead of ROS_ERROR because want to keep this fille ros-agnostics
    std::cerr<< "Failed to load ONNX file " << std::endl;
  }

  // deserialize the engine
  pfe_runtime_ = nvinfer1::createInferRuntime(g_logger_);
  rpn_runtime_ = nvinfer1::createInferRuntime(g_logger_);
  if (pfe_runtime_ == nullptr || rpn_runtime_ == nullptr)
  {
    std::cerr<<"Failed to create TensorRT Runtime object."<<std::endl;
  }

  pfe_engine_ =
      pfe_runtime_->deserializeCudaEngine(pfe_trt_model_stream->data(), pfe_trt_model_stream->size(), nullptr);
  rpn_engine_ =
      rpn_runtime_->deserializeCudaEngine(rpn_trt_model_stream->data(), rpn_trt_model_stream->size(), nullptr);
  if (pfe_engine_ == nullptr || rpn_engine_ == nullptr)
  {
    std::cerr << "Failed to create TensorRT Engine." << std::endl;
  }

  nvinfer1::INetworkDefinition* network;

  pfe_trt_model_stream->destroy();
  rpn_trt_model_stream->destroy();
  pfe_context_ = pfe_engine_->createExecutionContext();
  rpn_context_ = rpn_engine_->createExecutionContext();

  if (pfe_context_ == nullptr || rpn_context_ == nullptr)
  {
    std::cerr << "Failed to create TensorRT Execution Context." << std::endl;;
  }
}

void PointPillars::onnxToTRTModel(const std::string& model_file,             // name of the onnx model
                                  nvinfer1::IHostMemory*& trt_model_stream)  // output buffer for the TensorRT model
{
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

  // create the builder
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(g_logger_);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
		  1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto* config = builder->createBuilderConfig();

//#ifdef FP16_MODE
//  if(builder->platformHasFastFp16()) {
//      builder->setHalf2Mode(true);
//      config->setFlag(nvinfer1::BuilderFlag::kFP16);
//  }
//#endif

  auto parser = nvonnxparser::createParser(*network, g_logger_);

  if (!parser->parseFromFile(model_file.c_str(), verbosity))
  {
    std::string msg("failed to parse onnx file");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }

  // Build the engine
  builder->setMaxBatchSize(BATCH_SIZE_);
  config->setMaxWorkspaceSize(1 << 20);

  // Get concat layer of RPN, modify it
//  auto* clayer = network->getLayer(60);
//  if(clayer!=nullptr && clayer->getType() == nvinfer1::LayerType::kCONCATENATION){
//    auto *concat_l = dynamic_cast<nvinfer1::IConcatenationLayer*>(clayer);
//    nvinfer1::ITensor* second_input = concat_l->getInput(1);
//    concat_l->setInput(2,*second_input);

//    nvinfer1::ITensor* first_input = concat_l->getInput(0);
//    concat_l->setInput(1,*first_input);
//    concat_l->setInput(2,*first_input);
//    std::cout << "\nMODIFIED CONCATANATE LAYER 60\n";
//    concat_l->setInput(2,*first_input);
    // 1x128x248x216
//    nvinfer1::Dims dims;
//    dims.nbDims=4;
//    int arr[4] = {1,128,248,216};
//    std::memcpy(dims.d, arr, sizeof(int)*4);
//    nvinfer1::Weights weights;
//    weights.type = concat_l->getInput(0)->getType();//nvinfer1::DataType::kFLOAT;
//    weights.count = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3];
//    weights.values = static_cast<void*>(new float[weights.count]());
//    nvinfer1::IConstantLayer* const_layer = network->addConstant(dims, weights);
//    concat_l->setInput(1,*(const_layer->getOutput(0)));
//    concat_l->setInput(2,*(const_layer->getOutput(0)));
//  }

//  unsigned layernums[3] = {61,62,65}; // bounding box, class, dir
//  for(auto i=0; i<3; ++i){
//      auto* conv_l = dynamic_cast<nvinfer1::IConvolutionLayer*>(
//                  network->getLayer(layernums[i]));
//      if(conv_l!=nullptr && conv_l->getType() == nvinfer1::LayerType::kCONVOLUTION){
//        std::cout << "\nSSD Convolution " << i << ":";
//        std::cout << "\nNbOutputMaps: " << conv_l->getNbOutputMaps();
//        std::cout << "\nNbOutputs: " << conv_l->getNbOutputs();
//        std::cout << "\nNbGroups: " << conv_l->getNbGroups();
//        std::cout << "\nPaddingMode: " << static_cast<int>(conv_l->getPaddingMode());
//        std::cout << "\nPaddingNd(Dims): ";
//        nvinfer1::Dims dm = conv_l->getPaddingNd();
//        printDims(&dm);
//        std::cout << "\nPrePadding(Dims): ";
//        dm = conv_l->getPrePadding();
//        printDims(&dm);
//        std::cout << "\nPostPadding(Dims): ";
//        dm = conv_l->getPostPadding();
//        printDims(&dm);
//        std::cout << "\nStrideNd(Dims): ";
//        dm = conv_l->getStrideNd();
//        printDims(&dm);
//        std::cout << "\nDilationNd(Dims): ";
//        dm = conv_l->getDilationNd();
//        printDims(&dm);
//        std::cout << "\nKernelSizeNd(Dims): ";
//        dm = conv_l->getKernelSizeNd();
//        printDims(&dm);
//        std::cout << "\nKernelWeights: ";
//        nvinfer1::Weights w = conv_l->getKernelWeights();
//        printWeights(&w);
//        std::cout << "\nBiasWeights: ";
//        w = conv_l->getBiasWeights();
//        printWeights(&w);
//        std::cout << std::endl;
//      }
//  }



  // Get batch norm of middle in RPN
//  auto* layer = network->getLayer(58);
//  if(layer!=nullptr && layer->getType() == nvinfer1::LayerType::kSCALE){
//      auto *batch_norm_l = dynamic_cast<nvinfer1::IScaleLayer*>(layer);
//      std::cout << "\n\n\nRPN Layer 58, BatchNorm\nChannel Axis:" <<  batch_norm_l->getChannelAxis()
//                << "\nMode:" << static_cast<int>(batch_norm_l->getMode())
//                << "\nPower weights (size=";
//      nvinfer1::Weights weights = batch_norm_l->getPower();
//      std::cout << weights.count << "): ";
//      printWeights(&weights);

//      std::cout << "\nScale weights (size=";
//      weights = batch_norm_l->getScale();
//      std::cout << weights.count << "): ";
//      printWeights(&weights);

//      std::cout << "\nShift weights (size=";
//      weights = batch_norm_l->getShift();
//      std::cout << weights.count << "): ";
//      printWeights(&weights);

//      std::cout << "\nMODIFICATION TIME! SET EVERYTHING TO 0!";
//      if(batch_norm_l->getPower().count > 0){
//          nvinfer1::Weights weights_new;
//          weights_new.type = batch_norm_l->getPower().type;
//          weights_new.count = batch_norm_l->getPower().count;
//          weights_new.values = new float[weights_new.count]();
//          batch_norm_l->setPower(weights_new);
//      }

//      if(batch_norm_l->getScale().count > 0){
//          nvinfer1::Weights weights_new2;
//          weights_new2.type = batch_norm_l->getScale().type;
//          weights_new2.count = batch_norm_l->getScale().count;
//          weights_new2.values = new float[weights_new2.count]();
//          batch_norm_l->setScale(weights_new2);
//      }

//  }

  printNetwork(network);

  nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);
  parser->destroy();

  // serialize the engine, then close everything down
  trt_model_stream = engine->serialize();
  engine->destroy();
  config->destroy();
  network->destroy();
  builder->destroy();
}



void PointPillars::printNetwork(nvinfer1::INetworkDefinition *net_def)
{

    std::cout << "**Network " << net_def->getName() << std::endl;
    std::cout << "**Input tensors of network: " << std::endl;
    for(auto i = 0 ; i < net_def->getNbInputs(); ++i){
        printTensor(net_def->getInput(i));
    }
    std::cout << "**Layers: " << std::endl;
    for(auto i = 0 ; i < net_def->getNbLayers(); ++i){
        std::cout << i << "-";
        printLayer(net_def->getLayer(i));
    }
    std::cout << "**Output tensors of network: " << std::endl;
    for(auto i = 0 ; i < net_def->getNbOutputs(); ++i){
        printTensor(net_def->getOutput(i));
    }
}

void PointPillars::printTensor(nvinfer1::ITensor *tensor)
{
    std::cout << tensor->getName();
    std::cout << " Dims: ";
    auto d = tensor->getDimensions();
    printDims(&d);

    // these are Type:float32, AllowedFormats:63, BroadcastAcrossBatch:False by default
//    std::cout << " Type:" << static_cast<int>(tensor->getType())
//              << " AllowedFormats:" << tensor->getAllowedFormats()
//              << " BroadcastAcrossBatch:" << tensor->getBroadcastAcrossBatch();
    if(tensor->dynamicRangeIsSet())
        std::cout << " DynamicRangeMinMax:" << tensor->getDynamicRangeMin()
                  << " " << tensor->getDynamicRangeMax();
    // all of them are execution tensor by default
//    std::cout << " ExecTensor:" << (tensor->isExecutionTensor() ? "Y" : "N") << std::endl;
}

void PointPillars::printLayer(nvinfer1::ILayer *layer)
{
    std::cout << layer->getName() << " Type: " << static_cast<int>(layer->getType())  << std::endl;
//              << " Precision: " << static_cast<int>(layer->getPrecision()) << std::endl;
    std::cout << "* Input tensors of layer: ";
    for(auto i = 0 ; i < layer->getNbInputs(); ++i){
        printTensor(layer->getInput(i));
        std::cout << " | ";
    }
    //0 2 5 7 8 13 14
    std::cout << "\n*Output tensors of layer: ";
    for(auto i = 0 ; i < layer->getNbOutputs(); ++i){
        printTensor(layer->getOutput(i));
        std::cout << " | ";
    }
    std::cout << std::endl;
}

void PointPillars::printWeights(nvinfer1::Weights *weights)
{
    if(weights->type != nvinfer1::DataType::kFLOAT){
        std::cout << "WARNING! Can't print weights!";
        return;
    }
    else if(sizeof(float) != 4){
        std::cout << "ERROR! float is not 32 bits in this system!";
        return;
    }

    std::cout << " (size:" << weights->count << ") ";

    auto *vals = static_cast<const float*>(weights->values);
    for(auto i=0; i<weights->count; ++i){
      std::cout << vals[i] << " ";
    }

}

void PointPillars::printDims(nvinfer1::Dims *dims)
{
    std::cout << dims->d[0];
    for(auto i = 1 ; i < dims->nbDims; ++i)
        std::cout << "x" << dims->d[i];
}

void PointPillars::preprocessCPU(const float* in_points_array, const int in_num_points)
{
  int x_coors[MAX_NUM_PILLARS_] = { 0 };
  int y_coors[MAX_NUM_PILLARS_] = { 0 };
  float num_points_per_pillar[MAX_NUM_PILLARS_] = { 0 };
  float* pillar_x = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_];
  float* pillar_y = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_];
  float* pillar_z = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_];
  float* pillar_i = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_];

  float* x_coors_for_sub_shaped = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_];
  float* y_coors_for_sub_shaped = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_];
  float* pillar_feature_mask = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_];

  float* sparse_pillar_map = new float[NUM_INDS_FOR_SCAN_ * NUM_INDS_FOR_SCAN_];

  // everything given in here is to be filled expect in_points_array and in_num_points
  preprocess_points_ptr_->preprocess(in_points_array, in_num_points, x_coors, y_coors, num_points_per_pillar, pillar_x,
                                     pillar_y, pillar_z, pillar_i, x_coors_for_sub_shaped, y_coors_for_sub_shaped,
                                     pillar_feature_mask, sparse_pillar_map, host_pillar_count_);

  GPU_CHECK(cudaMemset(dev_x_coors_, 0, MAX_NUM_PILLARS_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_y_coors_, 0, MAX_NUM_PILLARS_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_pillar_x_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_y_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_z_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_i_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_x_coors_for_sub_shaped_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_y_coors_for_sub_shaped_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_num_points_per_pillar_, 0, MAX_NUM_PILLARS_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_sparse_pillar_map_, 0, NUM_INDS_FOR_SCAN_ * NUM_INDS_FOR_SCAN_ * sizeof(int)));

  // clang-format off
  GPU_CHECK(cudaMemcpy(dev_x_coors_, x_coors, MAX_NUM_PILLARS_ * sizeof(int), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_y_coors_, y_coors, MAX_NUM_PILLARS_ * sizeof(int), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_pillar_x_, pillar_x, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_pillar_y_, pillar_y, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_pillar_z_, pillar_z, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_pillar_i_, pillar_i, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float),cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_x_coors_for_sub_shaped_, x_coors_for_sub_shaped, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_y_coors_for_sub_shaped_, y_coors_for_sub_shaped,MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_num_points_per_pillar_, num_points_per_pillar, MAX_NUM_PILLARS_ * sizeof(float),cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_pillar_feature_mask_, pillar_feature_mask,MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_sparse_pillar_map_, sparse_pillar_map,NUM_INDS_FOR_SCAN_ * NUM_INDS_FOR_SCAN_ * sizeof(float), cudaMemcpyHostToDevice));
  // clang-format on

  delete[] pillar_x;
  delete[] pillar_y;
  delete[] pillar_z;
  delete[] pillar_i;
  delete[] x_coors_for_sub_shaped;
  delete[] y_coors_for_sub_shaped;
  delete[] pillar_feature_mask;
  delete[] sparse_pillar_map;
}

void PointPillars::preprocessGPU(const float* in_points_array, const int in_num_points)
{
  float* dev_points;
  // clang-format off
  GPU_CHECK(cudaMalloc((void**)&dev_points, in_num_points * NUM_BOX_CORNERS_ * sizeof(float)));
  GPU_CHECK(cudaMemcpy(dev_points, in_points_array, in_num_points * NUM_BOX_CORNERS_ * sizeof(float), cudaMemcpyHostToDevice));
  // clang-format on

  GPU_CHECK(cudaMemset(dev_pillar_count_histo_, 0, GRID_Y_SIZE_ * GRID_X_SIZE_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_sparse_pillar_map_, 0, NUM_INDS_FOR_SCAN_ * NUM_INDS_FOR_SCAN_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_pillar_x_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_y_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_z_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_i_, 0, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_x_coors_, 0, MAX_NUM_PILLARS_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_y_coors_, 0, MAX_NUM_PILLARS_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_num_points_per_pillar_, 0, MAX_NUM_PILLARS_ * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_anchor_mask_, 0, NUM_ANCHOR_ * sizeof(int)));

  preprocess_points_cuda_ptr_->doPreprocessPointsCuda(
      dev_points, in_num_points, dev_x_coors_, dev_y_coors_, dev_num_points_per_pillar_, dev_pillar_x_, dev_pillar_y_,
      dev_pillar_z_, dev_pillar_i_, dev_x_coors_for_sub_shaped_, dev_y_coors_for_sub_shaped_, dev_pillar_feature_mask_,
      dev_sparse_pillar_map_, host_pillar_count_);

  GPU_CHECK(cudaFree(dev_points));
}

void PointPillars::preprocess(const float* in_points_array, const int in_num_points)
{
  if (reproduce_result_mode_)
  {
    preprocessCPU(in_points_array, in_num_points);
  }
  else
  {
    preprocessGPU(in_points_array, in_num_points);
  }
}

void PointPillars::doInference(const float* in_points_array, const int in_num_points, std::vector<float>& out_detections)
{
  stats_.push_back(std::vector<double>());
  auto& stat = stats_.back();
  {
      LPP_TSTART
      preprocess(in_points_array, in_num_points);

      anchor_mask_cuda_ptr_->doAnchorMaskCuda(dev_sparse_pillar_map_, dev_cumsum_along_x_, dev_cumsum_along_y_,
                                              dev_box_anchors_min_x_, dev_box_anchors_min_y_, dev_box_anchors_max_x_,
                                              dev_box_anchors_max_y_, dev_anchor_mask_);

      LPP_TQUERY(stat)
  }
  cudaStream_t stream;
  {
      LPP_TSTART
      GPU_CHECK(cudaStreamCreate(&stream));
      // clang-format off
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[0], dev_pillar_x_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[1], dev_pillar_y_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[2], dev_pillar_z_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[3], dev_pillar_i_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[4], dev_num_points_per_pillar_, MAX_NUM_PILLARS_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[5], dev_x_coors_for_sub_shaped_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[6], dev_y_coors_for_sub_shaped_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[7], dev_pillar_feature_mask_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));

      // clang-format on
      pfe_context_->enqueue(BATCH_SIZE_, pfe_buffers_, stream, nullptr);
      // output is (float*)pfe_buffers_[8]

      //cudaStreamSynchronize(stream); // test
      GPU_CHECK(cudaMemset(dev_scattered_feature_, 0, RPN_INPUT_SIZE_ * sizeof(float)));
      scatter_cuda_ptr_->doScatterCuda(host_pillar_count_[0], dev_x_coors_, dev_y_coors_, (float*)pfe_buffers_[8],
                                       dev_scattered_feature_);
      LPP_TQUERY(stat)
  }
  {
      LPP_TSTART
      GPU_CHECK(cudaMemcpyAsync(rpn_buffers_[0], dev_scattered_feature_, BATCH_SIZE_ * RPN_INPUT_SIZE_ * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));

      rpn_context_->enqueue(BATCH_SIZE_, rpn_buffers_, stream, nullptr);
      //cudaStreamSynchronize(stream);// test

      GPU_CHECK(cudaMemset(dev_filter_count_, 0, sizeof(int)));
      LPP_TQUERY(stat)
  }
  {
      LPP_TSTART
      postprocess_cuda_ptr_->doPostprocessCuda(
          (float*)rpn_buffers_[1], (float*)rpn_buffers_[2], (float*)rpn_buffers_[3], dev_anchor_mask_, dev_anchors_px_,
          dev_anchors_py_, dev_anchors_pz_, dev_anchors_dx_, dev_anchors_dy_, dev_anchors_dz_, dev_anchors_ro_,
          dev_filtered_box_, dev_filtered_score_, dev_filtered_dir_, dev_box_for_nms_, dev_filter_count_, out_detections);

      // release the stream and the buffers
      //cudaStreamDestroy(stream); //test
      LPP_TQUERY(stat)
  }

  static auto processed_messages=0;
  if(++processed_messages % MESSAGES_TO_PRINT_STATS == 1){
      printStats();
#ifdef LIMIT_EXEC
      ros::shutdown();

#endif

  }
}

void PointPillars::printStats()
{
    //print stats
    std::cout<<"\n\nTime stats:\n";
    auto sum_stats = std::vector<double>();
    for(auto stat : stats_){
        auto sum=0;
        for(auto stat_elem : stat)
            sum += stat_elem;
        sum_stats.push_back(sum);
    }

    std::cout<<"Min: "<<*std::min_element(sum_stats.begin(), sum_stats.end())/BATCH_SIZE_<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(sum_stats.begin(), sum_stats.end())/BATCH_SIZE_<<" ms\n";
    unsigned mean=0;
    for(int i=0; i<sum_stats.size(); i++)
        mean += sum_stats[i];
    mean /= sum_stats.size();
    std::cout<<"Avg: "<<mean/BATCH_SIZE_<<" ms\t"<<1000/(mean/BATCH_SIZE_)<<" FPS\n";

    // Leave the quantization as unknown for now
    std::ofstream wf("lidar-point-pillars_unknown_timing.dat", std::ios::out);
    for(auto stat : stats_){
        for(auto stat_elem : stat)
            wf << stat_elem << ' ';
        wf << std::endl;
    }
}
