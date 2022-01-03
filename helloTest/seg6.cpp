// https://code.visualstudio.com/docs/cpp/cmake-linux

/*
    Usage:
---------
compiling
--------- 
	g++ -c seg6.cpp -I /opt/intel/openvino_2021/inference_engine/include -I /opt/intel/openvino_2021.4.689/opencv/include -I /opt/intel/openvino_2021.4.689/deployment_tools/inference_engine/samples/cpp/common/utils/include/


--------
linking: 
--------
	g++ seg6.o -L/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64 -L/opt/intel/openvino_2021.4.689/deployment_tools/ngraph/lib -L /opt/intel/openvino_2021.4.689/opencv/lib  -linference_engine -linference_engine_legacy -linference_engine_transformations -lngraph -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn


- executing: ./a.out

*/

#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <samples/ocv_common.hpp>
#include <iostream>
#include <utility>
#include <ctime>
#include <cmath>
using namespace std;
using std::cout;
using std::endl;
using namespace InferenceEngine; 

InferenceEngine::ExecutableNetwork load_model(std::string model_path, std::string device="CPU") {
    // build the path to the weights from the model path
    std::string weights_path = model_path.substr(0, model_path.size() - 4) + ".bin";
    
    // load the network from the generated IR files
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork net = ie.ReadNetwork(model_path, weights_path);

    // prepare the inputs
    std::string input_name = net.getInputsInfo().begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = net.getInputsInfo().begin()->second;

    // create the executable network
    InferenceEngine::ExecutableNetwork exec_net = ie.LoadNetwork(net, device);

    // return the executable network
    return exec_net;    
}

int pred(cv::Mat img, InferenceEngine::ExecutableNetwork exec_net, std::pair<int, int> img_size) 

{
    // resize the input image and convert it into a blob for inference

    unsigned long w = img_size.first, h = img_size.second;

    cv::resize(img, img, cv::Size(w, h));
    cv::Mat img1 = img.clone();
    cout<<"img height: "<<img1.size[0]<<endl;
    cout<<"img width: "<<img1.size[1]<<endl;
    cout<<"img channel: "<<img1.size[2]<<endl;    

    img = cv::dnn::blobFromImage(img, 1.0, cv::Size(w, h), cv::Scalar(0));
    // create an inference request
    InferenceEngine::InferRequest request = exec_net.CreateInferRequest();

    // extract the input information from the network
    InferenceEngine::ConstInputsDataMap input_info = exec_net.GetInputsInfo();

    // create a blob for inference and set it to the image data
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, w, h}, InferenceEngine::Layout::NCHW);
    InferenceEngine::Blob::Ptr img_blob = InferenceEngine::make_shared_blob<float>(desc, (float *)img.data);

    // set the blob to the infer request
    for(auto &item : input_info)
        request.SetBlob(item.first, img_blob);

    // run the inference
    request.Infer();

    // get the output information
    InferenceEngine::ConstOutputsDataMap output_info = exec_net.GetOutputsInfo();

    // declare output variables
    int label = -1;
    cv::Mat out;

    // process the output
    for(auto &item : output_info) {
        // grab the output
        InferenceEngine::Blob::Ptr output = request.GetBlob(item.first);
        float* buffer = output->cbuffer().as<float*>();
        // cout<<"first buffer: "<<buffer[0]<<endl;
        // create the output mask
        cv::Mat msk = cv::Mat(256, 256, CV_32F, buffer);  //retruns a masked image in black an white pixel 
        cv::imshow("msk1", msk);
        // cv::waitKey(0);
        // process the mask
        cv::threshold(msk, msk, 0.5, 1, cv::THRESH_BINARY);
        cv::imshow("msk", msk);
        // cv::imwrite("img.jpg",msk);
        cv::waitKey(0);
        // compute the output label
        cout<<"count the non zero: "<< countNonZero(msk)<<endl;
        label = (cv::countNonZero(msk) > 10) ? 1 : 0;

        // create the output image
        float alpha = 0.4;
        cv::cvtColor(msk, msk, cv::COLOR_GRAY2BGR);
        msk.convertTo(msk, CV_8U);
        cv::multiply(msk,cv::Scalar(0,0,255), msk, 1.0, CV_8U);

        // cout<<"msk height: "<<msk.size[0]<<endl;
        // cout<<"msk width: "<<msk.size[1]<<endl;
        // cout<<"msk channel: "<<msk.size[2]<<endl;

        // cout<<"img height: "<<img.size[0]<<endl;
        // cout<<"img width: "<<img.size[1]<<endl;
        // cout<<"img channel: "<<img.size[2]<<endl;
        cv::addWeighted(msk, alpha, img1, alpha, 0, out);
        cv::imshow("out", out);
        cv::waitKey(0);
        // cv::imwrite("output.png", out);
    }  

    // return the label
    return label;
    // return 0;
}

int main() {
    // const std::string model_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/segmentation/urvi/urvi.xml";
    const std::string model_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/segmentation/250_0/model/cam3/250_O_Cam3.xml";
    const std::string device = "CPU";

    // load the model
    InferenceEngine::ExecutableNetwork net = load_model(model_path);   

    // read the input image
    cv::Mat img = cv::imread("/home/sensovision/Desktop/openvino_inferencing code/Openvino/segmentation/250_0/train/cam3/20-09-2021_03_42_54img_39.jpg");

    // call the prediction function
    std::clock_t start = std::clock();
    int lbl = pred(img, net, std::make_pair(256, 256));  // model image size  
    cout<<"label: "<<lbl<<endl;
    std::clock_t end = std::clock();
}