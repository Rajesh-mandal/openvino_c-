// g++ -c cl1.cpp -I /opt/intel/openvino_2021/inference_engine/include -I /opt/intel/openvino_2021.4.689/opencv/include -I /opt/intel/openvino_2021.4.689/deployment_tools/inference_engine/samples/cpp/common/utils/include/
// g++ cl1.o -L/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64 -L/opt/intel/openvino_2021.4.689/deployment_tools/ngraph/lib -L /opt/intel/openvino_2021.4.689/opencv/lib  -linference_engine -linference_engine_legacy -linference_engine_transformations -lngraph -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn
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
using namespace cv;
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
    // img = cv::imread("/media/sensovision/RAJESH M/connect well/Good11.jpg");
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

    // std::cout<<"exec_net: "<<exec_net.request[0]<<std::endl;
    // create an inference request
    InferenceEngine::InferRequest request = exec_net.CreateInferRequest();

    // extract the input information from the network
    InferenceEngine::ConstInputsDataMap input_info = exec_net.GetInputsInfo();
   
    // create a blob for inference and set it to the image data
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, w, h}, InferenceEngine::Layout::NCHW);
    InferenceEngine::Blob::Ptr img_blob = InferenceEngine::make_shared_blob<float>(desc, (float *)img.data);
    std::cout<<"img_blob:: "<<img_blob<<std::endl;

    // set the blob to the infer request
    for(auto &item : input_info)
        request.SetBlob(item.first, img_blob);

    // run the inference
    request.Infer();

    // get the output information
    InferenceEngine::ConstOutputsDataMap output_info = exec_net.GetOutputsInfo();
    // std::cout<< "output_info" <<output_info<<std::endl;


    // initialise the output label
    int lbl = -1;
    bool defect_found = 0;

    // process the output
    for(auto &item : output_info) {
        // grab the output
        InferenceEngine::Blob::Ptr output = request.GetBlob(item.first);
        float* buffer = output->cbuffer().as<float*>();
        std::cout<< "buffer" <<buffer<<std::endl;

        // use the buffer value to compute the output label
        std::cout <<"buffer 0:" <<buffer[0] << std::endl;
        lbl = (buffer[0] < 0.5) ? 0 : 1;
        if(lbl < 1){
            defect_found = true;
        }
    }

    if(defect_found){
        cout << "defect found" << endl;
    } else {
        cout << "defect not found" << endl;
    }

    // return the output label
    return lbl;
}

int main() {
    const std::string model_path = "/home/sensovision/Downloads/openvino_connect_well/openvino_connect_well/classification.xml";
    const std::string device = "CPU";

    // load the model
    InferenceEngine::ExecutableNetwork net = load_model(model_path);   
    CNNNetwork cnet = net.GetExecGraphInfo();
    // net.Export("t2");
    // net.GetExecGraphInfo().serialize("t.xml","t.bin");
    // net.Export()
    // vector<string> names = net.getLayerNames();
    // for (string n : names)
    // cout << n << endl;


    // read the input image
    // cv::Mat img = cv::imread("/media/sensovision/RAJESH M/connect well/Bad30.jpg");
    cv::Mat img = cv::imread("/media/sensovision/RAJESH M/img_screenshot_27.12.2021.png");
    // Mat img = imread("/media/sensovision/RAJESH M/connect well/Good1.jpg", 0);

    // call the prediction function
    std::clock_t start = std::clock();
    int lbl = pred(img, net, std::make_pair(300, 300));  // model image size
    std::clock_t end = std::clock();
    cv::resize(img, img, cv::Size(300, 300));
    string str = "OK";
    string str1 = "NOT_OK";
    if(lbl == 1)
    {
        std::cout << "[INFO] given image does not contain any defect" << std::endl;
        // cv::putText(img,str,cv::Point(100,100),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        cv::imshow("img", img);
        cv::waitKey(0);
    }
        
    else if(lbl == 0)
    {
        std::cout << "[INFO] given image has defects" << std::endl;
        // cv::putText(img,str1,cv::Point(100,100),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,0,255),2,false);
        cv::imshow("img", img);
        cv::waitKey(0); 
    }       
    else
        std::cout << "[ERROR] unable to process" << std::endl;

    std::cout << "[INFO] total inference time = " << (end - start) / (double) CLOCKS_PER_SEC << " seconds"<<endl;
}