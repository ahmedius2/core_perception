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
#include "vision_tkdnn_detect.h"


namespace tkdnn
{

void BoundingBoxDetector::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    static std::vector<cv::Mat> batch_dnn_input;
    static auto batch_count=0;

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(msg, "bgr8");
    cv::Mat input_image = cv_image->image;

    batch_dnn_input.push_back(input_image);
    if(++batch_count < num_batches_){
        return;
    }
    batch_count=0;

    yolo_.update(batch_dnn_input, num_batches_); // inference
    batch_dnn_input.clear(); //not needed anymore

    autoware_msgs::DetectedObjectArray output_message;
    output_message.header = msg->header;

    for(auto img_bboxes : yolo_.batchDetected){
        for(auto bbox : img_bboxes){
            autoware_msgs::DetectedObject obj;
            obj.x = bbox.x;
            obj.y = bbox.y;
            obj.width = bbox.w;
            obj.height = bbox.h;

            obj.score = bbox.prob;
            if (true)//(use_coco_names_)
            {
                obj.label = yolo_.classesNames[bbox.cl];
            }
            else
            {
                obj.label = "unknown";
            }
            obj.valid = true;

            output_message.objects.push_back(obj);
        }
    }

    publisher_objects_.publish(output_message);
#ifdef LIMIT_EXEC
    if(++processed_messages_ == MESSAGES_TO_PROCESS){
        printStats();
    	ros::shutdown();
    }
#endif


}



void BoundingBoxDetector::Run()
{
    //ROS STUFF
    ros::NodeHandle private_node_handle("~");//to receive args

    //RECEIVE IMAGE TOPIC NAME
    std::string image_raw_topic_str;
    if (private_node_handle.getParam("image_raw_node", image_raw_topic_str))
    {
        ROS_INFO("Setting image node to %s", image_raw_topic_str.c_str());
    }
    else
    {
        ROS_INFO("No image node received, defaulting to /image_raw, you can use _image_raw_node:=YOUR_TOPIC");
        image_raw_topic_str = "/image_raw";
    }


    if (private_node_handle.getParam("pretrained_model_file", pretrained_model_file_))
    {
        ROS_INFO("Pretrained Model File (Weights): %s", pretrained_model_file_.c_str());
    }
    else
    {
        ROS_ERROR("No Pretrained Model File was received. Finishing execution.");
        return;
    }

    double min_confidence;
    int num_classes;

    private_node_handle.param<double>("score_threshold", min_confidence, 0.5);
    ROS_INFO("[%s] score_threshold: %f",__APP_NAME__, min_confidence);

    private_node_handle.param<int>("num_batches", num_batches_, 1);
    ROS_INFO("[%s] num_bacthes: %d",__APP_NAME__, num_batches_);

    private_node_handle.param<int>("num_classes", num_classes, 80); // COCO
    ROS_INFO("[%s] num_classes: %d",__APP_NAME__, num_classes);

    // not useful at the moment
//    int dla_id;
//    bool use_dla; std::string precision;

//    private_node_handle.param<bool>("use_dla", use_dla, false);
//    ROS_INFO("[%s] use_dla: %d", __APP_NAME__, use_dla);

//    private_node_handle.param<int>("dla_id", dla_id, 0);
//    ROS_INFO("[%s] dla_id: %d", __APP_NAME__, dla_id);

//    private_node_handle.param<std::string>("precision", precision, "FP32");
//    ROS_INFO("[%s] gpu/quantization precision: %s", __APP_NAME__, precision.c_str());

    ROS_INFO("Initializing tkDNN network...");
    yolo_.init(pretrained_model_file_,num_classes,num_batches_,min_confidence);
    ROS_INFO("Initialization complete.");

    //generateColors(colors_, 80);

    publisher_objects_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("/detection/image_detector/objects", 1);

    ROS_INFO("Subscribing to... %s", image_raw_topic_str.c_str());
    subscriber_image_raw_ = node_handle_.subscribe(image_raw_topic_str, 1, &BoundingBoxDetector::image_callback, this);

    ROS_INFO_STREAM( __APP_NAME__ << "" );

    ros::spin();

    ROS_INFO("END Yolo");

}

void BoundingBoxDetector::printStats()
{
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(yolo_.stats.begin(), yolo_.stats.end())/num_batches_<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(yolo_.stats.begin(), yolo_.stats.end())/num_batches_<<" ms\n";
    unsigned mean=0;
    for(int i=0; i<yolo_.stats.size(); i++)
        mean += yolo_.stats[i];
    mean /= yolo_.stats.size();
    std::cout<<"Avg: "<<mean/num_batches_<<" ms\t"<<1000/(mean/num_batches_)<<" FPS\n"<<COL_END;

    auto str = pretrained_model_file_.substr(0,pretrained_model_file_.length()-3);
    std::stringstream ss(str);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, '/'))
        tokens.push_back(token);
    std::ofstream wf(tokens.back()+"_timing.dat", std::ios::out | std::ios::app);
    for(auto tdiff : yolo_.stats)
        wf << tdiff << std::endl;
}

}
