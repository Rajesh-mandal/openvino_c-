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
using namespace std;

#include <ctime>
#include <cmath>

InferenceEngine::ExecutableNetwork load_model(std::string model_path, std::string device="CPU") 
    {
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

int pred(cv::Mat img, InferenceEngine::ExecutableNetwork exec_net, double threshold) {

    // create an inference request
    InferenceEngine::InferRequest request = exec_net.CreateInferRequest();
    cout<<"image size:" <<img.size;
    // extract the input information from the network
    InferenceEngine::ConstInputsDataMap input_info = exec_net.GetInputsInfo();

// convert the input image into a blob
    InferenceEngine::Blob::Ptr img_blob = wrapMat2Blob(img);
    cout<<img_blob->size()<<endl;
    // set the blob to the infer request
    for(auto &item : input_info)
    request.SetBlob(item.first, img_blob);

    // run the inference
    cout<<"reached v1"<<endl;
    request.Infer();

    // get the output information
    InferenceEngine::ConstOutputsDataMap output_info = exec_net.GetOutputsInfo();

    // declare output variables
    int label = -1;
    int res=-1;
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

        msk.convertTo(msk, CV_8UC1);

        cv::multiply(msk, 255, msk, 1.0, CV_8UC1);

        cv::addWeighted(msk, alpha, img, alpha, 0, out);

        // String names = "/home/optical/Desktop/im/ne"+to_string(count_img)+".jpg";
       imshow("resizedImg after",out);
       out.convertTo(out, CV_8UC1);
        // int pred_size;

        // res =   IFB_DL_blob(out,130,20,pred_size);
        // putText(out,to_string(res), Point(100,100),1,2,Scalar(255),1);
        // imwrite(names,out);
//        cv::imwrite("/home/optical/Desktop/QualViz/pythonOut/output_openvino.png", out);
    }

    // return the label
    return 0;
}


int main() {
    const std::string model_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/openvino python inference/segmentation/swaatik/best.xml";
    const std::string device = "CPU";

    // load the model
    InferenceEngine::ExecutableNetwork net = load_model(model_path);   

    // read the input image
    cv::Mat img = cv::imread("/home/sensovision/Desktop/Rajesh/Rajesh/swaatik/images/seg/018.jpg");

    // call the prediction function
    std::clock_t start = std::clock();
    // int lbl = pred(img, net, 0.0198);
    int lbl = pred(img, net, 0.1);
    std::clock_t end = std::clock();

    // if(lbl == 0)
    //     std::cout << "[INFO] given image does not contain any defect" << std::endl;
    // else if(lbl == 1)
    //     std::cout << "[INFO] given image has defects" << std::endl;
    // else
    //     std::cout << "[ERROR] unable to process" << std::endl;

    // std::cout << "[INFO] total inference time = " << (end - start) / (double) CLOCKS_PER_SEC << " seconds";
}