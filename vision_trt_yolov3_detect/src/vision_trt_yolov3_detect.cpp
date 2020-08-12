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
#include "vision_trt_yolov3_detect.h"

#if (CV_MAJOR_VERSION <= 2)
#include <opencv2/contrib/contrib.hpp>
#else
#include "gencolors.cpp"
#endif

namespace darknet
{

void Yolo3Detector::configure(bool use_dla, int dla_id, std::string precision,
                              uint32_t inp_width, uint32_t inp_height)
{
    use_dla_=use_dla;
    dla_id_=dla_id;
    precision_=precision;
    input_dims.d[0]=1;//batch size
    input_dims.d[1]=3;//channels
    input_dims.d[3]=inp_width;
    input_dims.d[2]=inp_height;
}

void Yolo3Detector::load(std::string& in_trained_file,
                         double in_min_confidence, double in_nms_threshold)
{
    // It is better to create the CUDA context first
    checkCudaErrors(cuInit(0));
    CUdevice cuDevice;
    int devID = 0;
    cuDeviceGet(&cuDevice, devID);
    // Create context as suggested in the TensorRT document
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

    bool use_builder=true;

    std::ifstream engine_file("yolov3.trt", std::ios::ate | std::ios::binary);
    std::vector<char> engine_buffer;
    if(engine_file.good()){
        std::streamsize file_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        engine_buffer.resize(file_size);
        if (engine_file.read(engine_buffer.data(), file_size))
        {
            trt_runtime_ = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
            use_builder=false;
            ROS_INFO("[%s] Using serialized model.", __APP_NAME__);

        }
        else{
            ROS_ERROR("[%s] Read error during deserialization, backing to builder."
                      , __APP_NAME__);
        }
    }

    if(use_builder) {
        trt_builder_ = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
        trt_network_ = trt_builder_->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        trt_bconfig_ = trt_builder_->createBuilderConfig();
        trt_onnx_parser_ = nvonnxparser::createParser(*trt_network_, gLogger.getTRTLogger());

        trt_builder_->setMaxBatchSize(1); // lets keep it simple for now

        // use default which uses cudaMalloc, cudaFree
        trt_builder_->setGpuAllocator(nullptr);

        trt_bconfig_->setMaxWorkspaceSize(1ULL << 30); // Let's make it 1024 MiB
        trt_bconfig_->setEngineCapability(nvinfer1::EngineCapability::kDEFAULT); // Full tensor capability
        //trt_bconfig_->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

        //nvinfer1::DataType dt; // I am manipulating this to save space, maybe keeping it FLOAT will provide max accuracy
        if (precision_.compare("INT8") == 0) {
            trt_bconfig_->setFlag(BuilderFlag::kFP16);// let it fallback to both FP32(default) and FP16
            trt_bconfig_->setFlag(BuilderFlag::kINT8);// if INT8 does not work for a layer
            //dt = nvinfer1::DataType::kINT8;
        } else if (precision_.compare("FP16") == 0) {
            trt_bconfig_->setFlag(BuilderFlag::kFP16);
            //dt = nvinfer1::DataType::kHALF;
        } else { // FP32
            //dt = nvinfer1::DataType::kFLOAT;
        }


        if (use_dla_) { // Use DLA if configured
            trt_bconfig_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            trt_bconfig_->setFlag(BuilderFlag::kGPU_FALLBACK);
            trt_bconfig_->setDLACore(dla_id_);
            // report compatibility of the layers with DLA
            for(int i=0; i< trt_network_->getNbLayers(); ++i){
                ILayer* layer = trt_network_->getLayer(i);
                std::cout << "Layer " << i << ": " << layer->getName()
                          << ",\t\tDLA compatible: "
                          << (trt_bconfig_->canRunOnDLA(layer) ? "YES\n" : "NO\n" );
            }
            std::cout << std::endl << std::flush;
        }

        trt_onnx_parser_->parseFromFile(in_trained_file.c_str(),
                                        (int)nvinfer1::ILogger::Severity::kWARNING);
        for (int i = 0; i < trt_onnx_parser_->getNbErrors(); ++i)
        {
            std::cout << trt_onnx_parser_->getError(i)->desc() << std::endl;
        }

        auto *inp0=trt_network_->getInput(0);
        inp_tensor_name_ = std::string(inp0->getName());
        auto dims0=inp0->getDimensions();
        dims0.d[0]=1;//Change batch size to 1
        input_dims = dims0;
        inp0->setDimensions(dims0);
        std::cout << "Inputs:\n";
        for( int i=0; i < trt_network_->getNbInputs(); ++i){
            auto *inp=trt_network_->getInput(i);
            auto dim = inp->getDimensions();
            std::cout << i << " -> Name: " << inp->getName() << " Dimension: ";
            for(int j=0; j < dim.nbDims ; ++j)
                std::cout << dim.d[i] << ",";
            std::cout << std::endl;
            //std::cout << " Type: " << inp->getType() << std::endl;
        }
        std::cout << std::endl << std::flush;

        std::cout << "Outputs:\n";
        for( int i=0; i < trt_network_->getNbOutputs(); ++i){
            auto *outp=trt_network_->getOutput(i);
            outp_tensor_names_.push_back(outp->getName());
            auto dim = outp->getDimensions();
            std::cout << i << " -> Name: " << outp->getName() << " Dimension: ";
            for(int j=0; j < dim.nbDims ; ++j)
                std::cout << dim.d[i] << ",";
            std::cout << std::endl;
            //std::cout << " Type: " << outp->getType() << std::endl;
        }
        std::cout << std::endl << std::flush;

        trt_cuda_engine_ = trt_builder_->buildEngineWithConfig(*trt_network_, *trt_bconfig_);
        if (trt_cuda_engine_ == nullptr) {
            ROS_ERROR("[%s] Couldn't build the CUDA Engine!", __APP_NAME__);
            return;
        }

        // Serialize for later use
        nvinfer1::IHostMemory *serializedModel = trt_cuda_engine_->serialize();
        std::ofstream engine_file_out("yolov3.trt", std::ios::out | std::ios::binary);
        if (engine_file_out.bad()) {
            ROS_ERROR("[%s] Couldn't open the file to serialize.", __APP_NAME__);
        }
        else{
            engine_file_out.write((char *) serializedModel->data(), serializedModel->size());
            ROS_INFO("[%s] Serialized the engine file.", __APP_NAME__);

        }
        serializedModel->destroy();
        trt_onnx_parser_->destroy();
        trt_bconfig_->destroy();
        trt_network_->destroy();
        trt_builder_->destroy();
    }
    else{ // use serialized engine
        inp_tensor_name_="000_net";
        // outputs should be [u'016_convolutional', u'023_convolutional']) for tiny yolo
        outp_tensor_names_.push_back("082_convolutional");
        outp_tensor_names_.push_back("094_convolutional");
        outp_tensor_names_.push_back("106_convolutional");
        trt_runtime_->setGpuAllocator(nullptr);
        if (use_dla_) {
            trt_runtime_->setDLACore(dla_id_);
        }
        trt_cuda_engine_ = trt_runtime_->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size());
        if (trt_cuda_engine_ == nullptr) {
            ROS_ERROR("[%s] Couldn't deserialize the CUDA Engine!", __APP_NAME__);
            return;
        }
        trt_runtime_->destroy(); // won't need it anymore, I think...
    }

    std::cout << "CUDA engine bindings:\n";
    for( int i=0; i < trt_cuda_engine_->getNbBindings(); ++i){
        auto *name= trt_cuda_engine_->getBindingName(i);
        auto dim = trt_cuda_engine_->getBindingDimensions(i);
        std::cout << i << " -> Name: " << name << " Dimension: ";
        for(int j=0; j < dim.nbDims ; ++j)
            std::cout << dim.d[i] << ",";
        std::cout << std::endl;
        //std::cout << " Type: " << outp->getType() << std::endl;
    }
    std::cout << std::endl << std::flush;

    min_confidence_ = in_min_confidence;
    nms_threshold_ = in_nms_threshold;
    //darknet_network_ = parse_network_cfg(&in_model_file[0]);
    //load_weights(darknet_network_, &in_trained_file[0]);
    //set_batch_network(darknet_network_, 1);

    //layer output_layer = darknet_network_->layers[darknet_network_->n - 1];
    //darknet_boxes_.resize(output_layer.w * output_layer.h * output_layer.n);
}

Yolo3Detector::~Yolo3Detector()
{
    trt_cuda_engine_->destroy();
    cuCtxDestroy(cuContext);
    //free_network(darknet_network_);
}

cv::cuda::GpuMat Yolo3Detector::preprocess(const sensor_msgs::ImageConstPtr& in_image_msg)
{
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(in_image_msg, "rgb8");

    // got help from tkDNN
    cv::cuda::GpuMat orig_img, img_resized, imagePreproc;
#ifdef DEBUG
    std:cout << "cv_image->image:\n" << cv_image->image << std::endl;
#endif
    orig_img = cv::cuda::GpuMat(cv_image->image);
    orig_img_w_=orig_img.rows;
    orig_img_h_=orig_img.cols;
    int w=input_dims.d[3], h=input_dims.d[2];
    cv::cuda::resize(orig_img, img_resized, cv::Size(w, h));
    img_resized.convertTo(imagePreproc, CV_32FC3, 1.0/255.0);

    return imagePreproc;
}

std::vector< RectClassScore<float> > Yolo3Detector::detect(cv::cuda::GpuMat input_image)
{
    samplesCommon::BufferManager buffers(trt_cuda_engine_,1);

    trt_exec_context_ = trt_cuda_engine_->createExecutionContext();

    float* hostDataBuffer = static_cast<float*>(
                buffers.getHostBuffer(inp_tensor_name_));

    //split channels
    cv::cuda::GpuMat rgb[3];
    cv::cuda::split(input_image,rgb);//split source
    int channels=input_dims.d[1], w=input_dims.d[3], h=input_dims.d[2];
    // It could be coded to avoid this copy, but nevermind for now
    for(int i=0; i< channels; i++) {
        cv::Mat hdb(1,w*h,CV_32F,hostDataBuffer + i*(w*h));
        rgb[i].download(hdb);
    }

    buffers.copyInputToDevice();

    bool status = trt_exec_context_->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        ROS_ERROR("[%s] Error during inference!\n", __APP_NAME__);
    }
    
    buffers.copyOutputToHost();

    auto detections = postprocess(buffers,nms_threshold_);

    for(auto tname : outp_tensor_names_){
        std::cout << tname << ": ";
        buffers.dumpBuffer(std::cout, tname);
        std::cout << std::endl;
    }
    
    trt_exec_context_->destroy();

    return detections;
}

std::vector< RectClassScore<float> > Yolo3Detector::postprocess(
        samplesCommon::BufferManager &bufs, float nms_thres)
{
    int w=input_dims.d[3], h=input_dims.d[2];
    int category_num=80; // if use_coco_names_ then 80 else size of custom_names_
    int filters= (4+1+category_num) * YOLOV3_OUT_BOXES;

    // yolov3 uses different strides of 32, 16, and 8
    int output_shapes[YOLOV3_OUT_BOXES][YOLOV3_OUTPUT_DIMS] = {
                               {1, filters, h / 32, w / 32},
                               {1, filters, h / 16, w / 16},
                               {1, filters, h / 8,  w / 8}};

    //int output_shapes_tiny[2][4] = {{1, filters, h / 32, w / 32},
    //                              {1, filters, h / 16, w / 16}};

    const int yolo_masks[YOLOV3_OUT_BOXES][3] = {{6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
    //int yolo_masks_tiny[2][3] = {{3, 4, 5}, {0, 1, 2}};

    // these are pre-defined default bounding-boxes
    // yolo calculates the offset and scale factor to them
    // not all of them will be used necessariliy, yolo_masks
    // define which ones will be used
    const int yolo_anchors[9][2] = {{10, 13}, {16, 30}, {33, 23},
                                    {30, 61}, {62, 45}, {59, 119},
                                    {116, 90}, {156, 198}, {373, 326}};

    //const int yolo_anchors_tiny[9][2] = {{10, 14}, {23, 27}, {37, 58},
    //                                {81, 82}, {135, 169}, {344, 319}};



    //float* hostDataOutpBuffer = static_cast<float*>(
   //             buffers.getHostBuffer(inp_tensor_name_));

    std::vector<cv::Mat> outputs;
    for(int i=0 ; i< outp_tensor_names_.size(); ++i){
        auto tname = outp_tensor_names_[i];
        // not sure that is correct
        outputs.push_back(cv::Mat(YOLOV3_OUTPUT_DIMS,output_shapes[i]
                                  ,CV_32F,bufs.getHostBuffer(tname)));

    }



//    float * in_data = in_darknet_image.data;
//    float *prediction = network_predict(darknet_network_, in_data);
//    layer output_layer = darknet_network_->layers[darknet_network_->n - 1];

//    output_layer.output = prediction;
//    int nboxes = 0;
//    int num_classes = output_layer.classes;
//    detection *darknet_detections = get_network_boxes(darknet_network_,
//              darknet_network_->w, darknet_network_->h, min_confidence_,
//              .5, NULL, 0, &nboxes);

//    do_nms_sort(darknet_detections, nboxes, num_classes, nms_threshold_);

    std::vector< RectClassScore<float> > detections;

//    for (int i = 0; i < nboxes; i++)
//    {
//        int class_id = -1;
//        float score = 0.f;
//        //find the class
//        for(int j = 0; j < num_classes; ++j){
//            if (darknet_detections[i].prob[j] >= min_confidence_){
//                if (class_id < 0) {
//                    class_id = j;
//                    score = darknet_detections[i].prob[j];
//                }
//            }
//        }
//        //if class found
//        if (class_id >= 0)
//        {
//            RectClassScore<float> detection;

//            detection.x = darknet_detections[i].bbox.x - darknet_detections[i].bbox.w/2;
//            detection.y = darknet_detections[i].bbox.y - darknet_detections[i].bbox.h/2;
//            detection.w = darknet_detections[i].bbox.w;
//            detection.h = darknet_detections[i].bbox.h;
//            detection.score = score;
//            detection.class_type = class_id;
//            //std::cout << detection.toString() << std::endl;

//            detections.push_back(detection);
//        }
//    }
//    //std::cout << std::endl;
    return detections;
}
}  // namespace darknet

///////////////////

//void Yolo3DetectorNode::convert_rect_to_image_obj(std::vector< RectClassScore<float> >& in_objects, autoware_msgs::DetectedObjectArray& out_message)
//{
//    for (unsigned int i = 0; i < in_objects.size(); ++i)
//    {
//        {
//            autoware_msgs::DetectedObject obj;

//            obj.x = (in_objects[i].x /image_ratio_) - image_left_right_border_/image_ratio_;
//            obj.y = (in_objects[i].y /image_ratio_) - image_top_bottom_border_/image_ratio_;
//            obj.width = in_objects[i].w /image_ratio_;
//            obj.height = in_objects[i].h /image_ratio_;
//            if (in_objects[i].x < 0)
//                obj.x = 0;
//            if (in_objects[i].y < 0)
//                obj.y = 0;
//            if (in_objects[i].w < 0)
//                obj.width = 0;
//            if (in_objects[i].h < 0)
//                obj.height = 0;

//            obj.score = in_objects[i].score;
//            if (use_coco_names_)
//            {
//                obj.label = in_objects[i].GetClassString();
//            }
//            else
//            {
//                if (in_objects[i].class_type < custom_names_.size())
//                    obj.label = custom_names_[in_objects[i].class_type];
//                else
//                    obj.label = "unknown";
//            }
//            obj.valid = true;

//            out_message.objects.push_back(obj);

//        }
//    }
//}




void Yolo3DetectorNode::image_callback(const sensor_msgs::ImageConstPtr& in_image_message)
{
    std::vector< RectClassScore<float> > detections;

    auto img = yolo_detector_.preprocess(in_image_message);

    detections = yolo_detector_.detect(img);

    //Prepare Output message
    autoware_msgs::DetectedObjectArray output_message;
    output_message.header = in_image_message->header;



    //convert_rect_to_image_obj(detections, output_message);

    //publisher_objects_.publish(output_message);
}

void Yolo3DetectorNode::config_cb(const autoware_config_msgs::ConfigSSD::ConstPtr& param)
{
    score_threshold_ = param->score_threshold;
}

std::vector<std::string> Yolo3DetectorNode::read_custom_names_file(const std::string& in_names_path)
{
    std::ifstream file(in_names_path);
    std::string str;
    std::vector<std::string> names;
    while (std::getline(file, str))
    {
        names.push_back(str);
        std::cout << str <<  std::endl;
    }
    return names;
}

void Yolo3DetectorNode::Run()
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


    std::string pretrained_model_file, names_file;
    if (private_node_handle.getParam("pretrained_model_file", pretrained_model_file))
    {
        ROS_INFO("Pretrained Model File (Weights): %s", pretrained_model_file.c_str());
    }
    else
    {
        ROS_INFO("No Pretrained Model File was received. Finishing execution.");
        return;
    }

    if (private_node_handle.getParam("names_file", names_file))
    {
        ROS_INFO("Names File: %s", names_file.c_str());
        use_coco_names_ = false;
        custom_names_ = read_custom_names_file(names_file);
    }
    else
    {
        ROS_INFO("No Names file was received. Using default COCO names.");
        use_coco_names_ = true;
    }

    private_node_handle.param<float>("score_threshold", score_threshold_, 0.5);
    ROS_INFO("[%s] score_threshold: %f",__APP_NAME__, score_threshold_);

    private_node_handle.param<float>("nms_threshold", nms_threshold_, 0.45);
    ROS_INFO("[%s] nms_threshold: %f",__APP_NAME__, nms_threshold_);

    int dla_id; bool use_dla; std::string precision;

    private_node_handle.param<bool>("use_dla", use_dla, false);
    ROS_INFO("[%s] use_dla: %d", __APP_NAME__, use_dla);

    private_node_handle.param<int>("dla_id", dla_id, 0);
    ROS_INFO("[%s] dla_id: %d", __APP_NAME__, dla_id);

    private_node_handle.param<std::string>("precision", precision, "FP32");
    ROS_INFO("[%s] gpu/nvdla precision: %s", __APP_NAME__, precision.c_str());

    private_node_handle.param<int>("network_input_width", network_input_width_, 416);
    ROS_INFO("[%s] network input width: %d", __APP_NAME__, network_input_width_);

    private_node_handle.param<int>("network_input_height", network_input_height_, 416);
    ROS_INFO("[%s] network input height: %d", __APP_NAME__, network_input_height_);

    yolo_detector_.configure(use_dla,dla_id,precision,
                             network_input_width_,network_input_height_);

    ROS_INFO("Initializing YOLOv3 on TensorRT...");
    yolo_detector_.load(pretrained_model_file, score_threshold_, nms_threshold_);
    ROS_INFO("Initialization complete.");


#if (CV_MAJOR_VERSION <= 2)
    cv::generateColors(colors_, 80);
#else
    generateColors(colors_, 80);
#endif

    publisher_objects_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("/detection/image_detector/objects", 1);

    ROS_INFO("Subscribing to... %s", image_raw_topic_str.c_str());
    subscriber_image_raw_ = node_handle_.subscribe(image_raw_topic_str, 1, &Yolo3DetectorNode::image_callback, this);

    std::string config_topic("/config");
    config_topic += "/Yolov3";
    subscriber_yolo_config_ = node_handle_.subscribe(config_topic, 1, &Yolo3DetectorNode::config_cb, this);

    ROS_INFO_STREAM( __APP_NAME__ << "" );

    ros::spin();
    ROS_INFO("END Yolo");

}
