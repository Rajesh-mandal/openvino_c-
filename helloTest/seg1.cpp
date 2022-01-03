/*
    Usage:
---------
compiling
--------- 
	g++ -c seg.cpp -I /opt/intel/openvino_2021/inference_engine/include -I /opt/intel/openvino_2021.4.689/opencv/include -I /opt/intel/openvino_2021.4.689/deployment_tools/inference_engine/samples/cpp/common/utils/include/


--------
linking: 
--------
	g++ seg.o -L/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64 -L/opt/intel/openvino_2021.4.689/deployment_tools/ngraph/lib -L /opt/intel/openvino_2021.4.689/opencv/lib  -linference_engine -linference_engine_legacy -linference_engine_transformations -lngraph -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn


- executing: ./seg.out

*/

#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <samples/ocv_common.hpp>
#include <iostream>
#include <ctime>
using namespace InferenceEngine;    //# removed the error of layout has not been declared
using std::cout;
using std::endl;
using namespace std;

int main() {
    const std::string model_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/segmentation/urvi/urvi.xml";
    const std::string weights_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/segmentation/urvi/urvi.bin";
    const std::string device = "CPU";

    // load the network from the generated IR files
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork net = ie.ReadNetwork(model_path, weights_path);

    // prepare the inputs
    std::string input_name = net.getInputsInfo().begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = net.getInputsInfo().begin()->second;
    
    // set the input precision
    input_info->setPrecision(InferenceEngine::Precision::U8);
    input_info->getInputData()->setLayout(Layout::NCHW);

    // prepare the outputs
    std::string output_name = net.getOutputsInfo().begin()->first;

    // create the executable network
    InferenceEngine::ExecutableNetwork exec_net = ie.LoadNetwork(net, device);

    // create an inference request
    InferenceEngine::InferRequest request = exec_net.CreateInferRequest();

    std::clock_t start = std::clock();

    // read the input image
    cv::Mat img = cv::imread("/home/sensovision/Desktop/Rajesh/Rajesh/datasets/exibitation Jamnagar/ok/14-12-2021_10:42:02img_148.jpg");

    // resize the input image
    cv::resize(img, img, cv::Size(256, 256));

    // convert the input image into a blob
    InferenceEngine::Blob::Ptr img_blob = wrapMat2Blob(img);

    // set the blob to the infer request
    request.SetBlob(input_name, img_blob);

    // run the inference
    request.Infer();

    // grab the output
    InferenceEngine::Blob::Ptr output = request.GetBlob(output_name);
    float* buffer = output->cbuffer().as<float*>();

    // create the output mask
    cv::Mat msk = cv::Mat(256, 256, CV_32F, buffer);

    // process the mask
    cv::threshold(msk, msk, 0.5, 1, cv::THRESH_BINARY);
    cv::cvtColor(msk, msk, cv::COLOR_GRAY2BGR);
    msk.convertTo(msk, CV_8U);
    cv::multiply(msk, 255, msk, 1.0, CV_8U);

    // create the output image
    cv::Mat out;
    float alpha = 0.4;
    cv::addWeighted(msk, alpha, img, alpha, 0, out);

    std::clock_t end = std::clock();
    std::cout << "[INFO] total inference time = " << (end - start) / (double) CLOCKS_PER_SEC << " seconds";

    // cv::imshow("image", out);
    // cv::waitKey(0);
}