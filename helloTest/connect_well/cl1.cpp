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

int pred(cv::Mat img, InferenceEngine::ExecutableNetwork exec_net, std::pair<int, int> img_size) {
    // calculate dimensions for center crop
    // const int crop_size = 300;
    // const int off_w = (img.cols - crop_size) / 2;
    // const int off_h = (img.rows - crop_size) / 2;

    // // perform center crop
    // const cv::Rect roi(off_w, off_h, crop_size, crop_size);
    // img = img(roi).clone();

    // resize the input image and convert it into a blob for inference
    unsigned long w = img_size.first, h = img_size.second;
    cv::resize(img, img, cv::Size(w, h));
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

    // initialise the output label
    int lbl = -1;

    // process the output
    for(auto &item : output_info) {
        // grab the output
        InferenceEngine::Blob::Ptr output = request.GetBlob(item.first);
        float* buffer = output->cbuffer().as<float*>();

        // use the buffer value to compute the output label
        std::cout << buffer[0] << std::endl;
        lbl = (buffer[0] < 0.5) ? 0 : 1;
    }

    // return the output label
    return lbl;
}

int main() {
    const std::string model_path = "/home/sensovision/Downloads/openvino_connect_well/openvino_connect_well/classification.xml";
    const std::string device = "CPU";

    // load the model
    InferenceEngine::ExecutableNetwork net = load_model(model_path);   

    // read the input image
    cv::Mat img = cv::imread("/home/sensovision/Desktop/Rajesh/Rajesh/datasets/connect_well/combined_unseen_data/bad033.jpg");

    // call the prediction function
    std::clock_t start = std::clock();
    int lbl = pred(img, net, std::make_pair(300, 300));  // model image size  
    std::clock_t end = std::clock();

    if(lbl == 1)
        std::cout << "[INFO] given image does not contain any defect" << std::endl;
        
    else if(lbl == 0)
        std::cout << "[INFO] given image has defects" << std::endl;
    else
        std::cout << "[ERROR] unable to process" << std::endl;

    std::cout << "[INFO] total inference time = " << (end - start) / (double) CLOCKS_PER_SEC << " seconds"<<endl;
}