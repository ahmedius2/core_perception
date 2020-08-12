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
 ********************
 *  v1.0: amc-nu (abrahammonrroy@yahoo.com)
 *
 * yolo3_node.cpp
 *
 *  Created on: April 4th, 2018
 */
#ifndef DARKNET_YOLO3_H
#define DARKNET_YOLO3_H

#define __APP_NAME__ "vision_trt_yolov3_detect"

#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>

#include <autoware_config_msgs/ConfigSSD.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>

#include <rect_class_score.h>

#include <opencv2/opencv.hpp>

#include <cuda.h>
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"

#define YOLOV3_OUT_BOXES 3 // anchor boxes
#define YOLOV3_OUTPUT_DIMS 4

namespace darknet {
    // This will output the proper CUDA error strings in the event that a CUDA host
    // call returns an error
    #ifndef checkCudaErrors
    #define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

    // These are the inline versions for all of the SDK helper functions
    inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
      if (CUDA_SUCCESS != err) {
        const char *errorStr = NULL;
        cuGetErrorString(err, &errorStr);
        fprintf(stderr,
                "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
                "line %i.\n",
                err, errorStr, file, line);
        exit(EXIT_FAILURE);
      }
    }
    #endif


    class Yolo3Detector {
    private:
        double min_confidence_, nms_threshold_;
        int orig_img_w_, orig_img_h_;
        //network* darknet_network_;
        //std::vector<box> darknet_boxes_;
        //std::vector<RectClassScore<float> > forward(image &in_darknet_image);

        //TensorRT variables
        int gpu_device_id_, dla_id_; // GPU and NVDLA (for jetson platforms)
        bool use_gpu_, use_dla_;
        std::string precision_, inp_tensor_name_;
        std::vector<std::string> outp_tensor_names_;
        //uint32_t network_input_width_, network_input_height_;
        Dims input_dims; // [batch_size, channel, height, width]

        CUcontext cuContext;
        IBuilder* trt_builder_;
        INetworkDefinition* trt_network_;
        nvonnxparser::IParser* trt_onnx_parser_;
        IBuilderConfig* trt_bconfig_;
        nvinfer1::ICudaEngine* trt_cuda_engine_;
        nvinfer1::IRuntime* trt_runtime_;
        nvinfer1::IExecutionContext* trt_exec_context_;
    public:
        Yolo3Detector() {}

        void configure(bool use_dla, int dla_id, std::string precision,
                       uint32_t inp_width, uint32_t inp_height);

        void load(std::string &in_trained_file, double in_min_confidence,
                  double in_nms_threshold);

        cv::cuda::GpuMat preprocess(const sensor_msgs::ImageConstPtr &in_image_msg);

        std::vector<RectClassScore<float> > detect(cv::cuda::GpuMat input_image);

        std::vector< RectClassScore<float> > postprocess(
                samplesCommon::BufferManager &bufs, float nms_thres);

        void process_yoloV3_output();

        ~Yolo3Detector();

    };
}  // namespace darknet

class Yolo3DetectorNode {
    ros::Subscriber                 subscriber_image_raw_;
    ros::Subscriber                 subscriber_yolo_config_;
    ros::Publisher                  publisher_objects_;
    ros::NodeHandle                 node_handle_;

    darknet::Yolo3Detector          yolo_detector_;

    float                           score_threshold_;
    float                           nms_threshold_;
    int network_input_width_, network_input_height_;
    std::vector<cv::Scalar>         colors_;

    std::vector<std::string>        custom_names_;
    bool                            use_coco_names_;

    //void                            convert_rect_to_image_obj(std::vector< RectClassScore<float> >& in_objects,
    //                                 autoware_msgs::DetectedObjectArray& out_message);
    void                            image_callback(const sensor_msgs::ImageConstPtr& in_image_message);
    void                            config_cb(const autoware_config_msgs::ConfigSSD::ConstPtr& param);
    std::vector<std::string>        read_custom_names_file(const std::string& in_path);
public:
    void    Run();
};

#endif  // DARKNET_YOLO3_H
