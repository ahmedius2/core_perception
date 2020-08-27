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

#define __APP_NAME__ "vision_tkdnn_detect"

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

#include <opencv2/opencv.hpp>

#include <tkDNN/CenternetDetection.h>
#include <tkDNN/MobilenetDetection.h>
#include <tkDNN/Yolo3Detection.h>

//#define LIMIT_EXEC
#ifdef LIMIT_EXEC
#define MESSAGES_TO_PROCESS 3639u
#endif

namespace tkdnn {

class BoundingBoxDetector {
    ros::Subscriber                 subscriber_image_raw_;
    ros::Subscriber                 subscriber_yolo_config_;
    ros::Publisher                  publisher_objects_;
    ros::NodeHandle                 node_handle_;

    int num_batches_;
    std::string pretrained_model_file_;
#ifdef LIMIT_EXEC
    unsigned processed_messages_=0;
#endif

    // Network variables
    tk::dnn::Yolo3Detection yolo_;

    //int dla_id_; // GPU and NVDLA (for jetson platforms)
    //bool use_dla_;
    //std::string precision_;

    void  image_callback(const sensor_msgs::ImageConstPtr& msg);

public:
    void    Run();
    void    printStats();
};

}

#endif  // DARKNET_YOLO3_H
