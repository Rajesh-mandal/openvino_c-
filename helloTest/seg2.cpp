#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <samples/ocv_common.hpp>
#include <iostream>
#include <ctime>

using std::cout;
using std::endl;

InferenceEngine::ExecutableNetwork load_model(std::string model_path, std::string device="CPU") {
    // build the path to the weights from the model path
    std::string weights_path = model_path.substr(0, model_path.size() - 4) + ".bin";
    
    // load the network from the generated IR files
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork net = ie.ReadNetwork(model_path, weights_path);

    // prepare the inputs
    std::string input_name = net.getInputsInfo().begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = net.getInputsInfo().begin()->second;

    // set the input precision
    input_info->setPrecision(InferenceEngine::Precision::U8);

    // create the executable network
    InferenceEngine::ExecutableNetwork exec_net = ie.LoadNetwork(net, device);

    // return the executable network
    return exec_net;
}

int pred(cv::Mat img, InferenceEngine::ExecutableNetwork exec_net) {
    // create an inference request
    InferenceEngine::InferRequest request = exec_net.CreateInferRequest();
    cvtColor(img,img,cv::COLOR_BGR2RGB);
    // extract the input information from the network
    InferenceEngine::ConstInputsDataMap input_info = exec_net.GetInputsInfo();

    // convert the input image into a blob
    InferenceEngine::Blob::Ptr img_blob = wrapMat2Blob(img);
    cout<< "image blob::" <<img_blob->size()<<endl;
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

        // create the output mask
        cv::Mat msk = cv::Mat(256, 256, CV_32F, buffer);

        // process the mask
        cv::threshold(msk, msk, 0.5, 1, cv::THRESH_BINARY);

        // compute the output label
        label = (cv::countNonZero(msk) > 10) ? 1 : 0;

        // create the output image
        float alpha = 0.4;
        cv::cvtColor(msk, msk, cv::COLOR_GRAY2BGR);
        msk.convertTo(msk, CV_8U);
        cv::multiply(msk, 255, msk, 1.0, CV_8U);
        cv::addWeighted(msk, alpha, img, alpha, 0, out);
        // cv::imwrite("output.png", out);
    }  

    // return the label
    return label;
}

int main() {
    const std::string model_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/segmentation/urvi/urvi.xml";
    const std::string weights_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/segmentation/urvi/urvi.bin";
    const std::string device = "CPU";

    // load the model
    InferenceEngine::ExecutableNetwork net = load_model(model_path);   

    // read the input image
    cv::Mat img = cv::imread("/home/sensovision/Desktop/Rajesh/Rajesh/datasets/exibitation Jamnagar/ok/14-12-2021_10:42:02img_148.jpg");

    // resize the input image
    cv::resize(img, img, cv::Size(256, 256));

    // call the prediction function
    std::clock_t start = std::clock();
    int lbl = pred(img, net);
    std::clock_t end = std::clock();
    
    std::cout << "[INFO] total inference time = " << (end - start) / (double) CLOCKS_PER_SEC << " seconds";
}