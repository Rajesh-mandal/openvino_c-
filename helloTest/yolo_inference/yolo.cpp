//https://github.com/fb029ed/yolov5_cpp_openvino/blob/master/README.md
//https://blog.fireheart.in/a?ID=f09be881-767e-4dd3-8524-5c0a3d69eb5e
//https://github.com/itsnine/yolov5-onnxruntime
/*
    Usage:
        - compiling: $ g++ -c yolo_inference.cpp -I /opt/intel/openvino_2021/deployment_tools/inference_engine/include -I /opt/intel/openvino_2021.1.110/opencv/include -I /opt/intel/openvino_2021.1.110/deployment_tools/inference_engine/samples/cpp/common

        - linking: g++ yolo_inference.o -L/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64 -L/opt/intel/openvino_2021.1.110/deployment_tools/ngraph/lib/ -L /opt/intel/openvino_2021.1.110/opencv/lib  -linference_engine -linference_engine_legacy -linference_engine_transformations -lngraph -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

        - executing: ./a.out
*/

#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <samples/ocv_common.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

struct Object {
    float prob;
    std::string name;
    cv::Rect rect;
};

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

std::vector<int> get_anchors(int scale) {
    // initialise the list of anchors
    std::vector<int> anchors(6);

    // initialise the grid coordinates
    int a80[6] = {10, 13, 16, 30, 33, 23};
    int a40[6] = {30, 61, 62, 45, 59, 119};
    int a20[6] = {116, 90, 156, 198, 373, 326};


    // compute the anchors
    if (scale == 80)
        anchors.insert(anchors.begin(), a80, a80 + 6);
    else if (scale == 40)
        anchors.insert(anchors.begin(), a40, a40 + 6);
    else  if (scale == 20)
        anchors.insert(anchors.begin(), a20, a20 + 6);
    
    // return the compute anchors
    return anchors;
}

double sigmoid(double x) {
    if (x < 0)
        return exp(x) / (1 + exp(x));
    else
        return (1 / (1 + exp(-x)));
}

// Note that the threshold here is the threshold of the product of the box and the object prob

bool parse_output(const InferenceEngine::Blob::Ptr &blob, int scale, float conf_threshold, 
std::vector<cv::Rect> &o_rect,std::vector<float> &o_rect_conf, std::vector<int> &classId) 
{
    // form the anchors for the current scale
    std::vector<int> anchors = get_anchors(scale);

    // initialise the output blob
    InferenceEngine::LockedMemory<const void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();

    int item_size = 7;   // total_class + 5
    size_t anchor_n = 3;
    for (int n = 0; n < anchor_n; ++n)
        for (int i = 0; i < scale; ++i)
            for (int j = 0 ; j < scale; ++j) {

                // extract the confidence score
                double box_prob = output_blob[n * scale * scale * item_size + i * scale * item_size + j * item_size + 4];
                // std::cout<<"box_prob: "<<box_prob<<std::endl;
                // std::cout<<"conf_threshold: "<<conf_threshold<<std::endl;
                box_prob = sigmoid(box_prob);
                // std::cout<<"box_prob:::: "<<box_prob<<std::endl;

                // check if the confidence is greater than the threshold
                if (box_prob < conf_threshold)
                    continue;
                
                // extract the coordinates of the bbox and convert them to xywh format
                double x = output_blob[n * scale * scale * item_size + i * scale * item_size + j * item_size + 0];
                double y = output_blob[n * scale * scale * item_size + i * scale * item_size + j * item_size + 1];
                double w = output_blob[n * scale * scale * item_size + i * scale * item_size + j * item_size + 2];
                double h = output_blob[n * scale * scale * item_size + i * scale * item_size + j * item_size + 3];


                // extract class probabilities
                double max_prob = 0;
                int idx = 0;
                for (int t = 5 ; t < item_size; ++t) {
                    // extract the score and sigmoid it
                    double tp = output_blob[n * scale * scale * item_size + i * scale * item_size + j * item_size + t];
                    tp = sigmoid(tp);
                    // std::cout<<"tp:: "<<tp<<std::endl;
                    // check if the current class prob is the max prob
                    if (tp > max_prob) {
                        max_prob = tp;
                        idx = t;
                        // std::cout<<"idx:: "<<idx<<std::endl;
                    }
                }

                float conf = box_prob * max_prob;                
                // For borders whose border confidence is less than the threshold, do not care about other values, do not perform calculations to reduce the amount of calculation
                if (conf < conf_threshold)
                    continue;

                // compute the final coordinates
                x = (sigmoid(x) * 2 - 0.5 + j) * 640.0F / scale;
                y = (sigmoid(y) * 2 - 0.5 + i) * 640.0F / scale;
                w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
                h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

                // compute the opencv boxes
                double r_x = x - w / 2 ;
                double r_y = y - h / 2 ;
                cv::Rect rect = cv::Rect(round(r_x), round(r_y), round(w), round(h));

                // use the class index to determine class label

                std::cout<<"idx BEFORE:: "<<idx<<std::endl;
                idx = item_size - idx - 1;
              
                std::cout<<"idx:: "<<idx<<std::endl;

                o_rect.push_back(rect);
                o_rect_conf.push_back(box_prob);
                classId.push_back(idx);
            }

    // return whether a detection was made
    if (o_rect.size() == 0)
        return false;
    else  
        return true;
}

void pred(cv::Mat img, InferenceEngine::ExecutableNetwork exec_net, double conf_threshold, double nms_threshold, std::vector<Object> &detectedObjects) {
    // convert the image from BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // resize the input image and convert it into a blob for inference
    cv::resize(img, img, cv::Size(640, 640));

    // img = img/255.0;  

    img = cv::dnn::blobFromImage(img, 1.0, cv::Size(640, 640), cv::Scalar(0));
    img = img/255.0;  

    // create an inference request
    InferenceEngine::InferRequest request = exec_net.CreateInferRequest();

    // extract the input information from the network
    InferenceEngine::ConstInputsDataMap input_info = exec_net.GetInputsInfo();

    // create a blob for inference and set it to the image data
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, 640, 640}, InferenceEngine::Layout::NCHW);
    InferenceEngine::Blob::Ptr img_blob = InferenceEngine::make_shared_blob<float>(desc, (float *)img.data);

    // set the blob to the infer request
    for(auto &item : input_info)
        request.SetBlob(item.first, img_blob);

    // run the inference
    request.Infer();

    // get the output information
    InferenceEngine::ConstOutputsDataMap output_info = exec_net.GetOutputsInfo();

    // extract the results for each layer
    std::vector<cv::Rect> origin_rect;
    std::vector<float> origin_rect_conf;
    std::vector<int> classId;
    
    int scales[3] = {40, 20, 80};
  
    int i = 0;

    int count = 0;
    for (auto &output: output_info) {
        // extract the output data
        std::vector<cv::Rect> origin_rect_temp;
		std::vector<float> origin_rect_cof_temp;
        auto output_name = output.first;
        InferenceEngine::Blob::Ptr blob = request.GetBlob(output_name);

        float* buffer = blob->cbuffer().as<float*>();

        // parse the extracted output
        // parse_output(blob,scales[i],conf_threshold,origin_rect,origin_rect_conf);
        parse_output(blob, scales[i], conf_threshold, origin_rect, origin_rect_conf, classId);
        // origin_rect.insert(origin_rect.end(), origin_rect_temp.begin(), origin_rect_temp.end());
		// origin_rect_conf.insert(origin_rect_conf.end(), origin_rect_cof_temp.begin(), origin_rect_cof_temp.end());
        ++i;
    }

    // Post-processing to obtain the final test result
    std::vector<int> final_id;
    // std::cout<<"origin_rect: "<<origin_rect[30]<<std::endl;
    // std::cout<<"origin_rect_conf: "<<origin_rect_conf.size()<<std::endl;
    // std::cout<<"conf_threshold: "<<conf_threshold<<std::endl;
    // std::cout<<"nms_threshold: "<<nms_threshold<<std::endl;
    // std::cout<<"final_id: "<<final_id.size()<<std::endl;
    cv::dnn::NMSBoxes(origin_rect, origin_rect_conf, conf_threshold, nms_threshold, final_id);

    // Get the final result according to final_id
    // std::string className[6] = {"MP15N","C9783","MP15","MG4730","UP15CS0090","DL2CAW9699" };
    // std::string className[6] = {"DL2CAW9699","UP15CS0090","MG4730","MP15","C9783","MP15N"};
    // std::string className[3] = {"AAN","HERO","434"};
    std::string className[2] = {"CM","CP"};

    std::cout<<"final_id.size(): "<<final_id.size()<<std::endl;
    // std::cout<<"final_id.size(0)"<<classId[final_id[0]]<<std::endl;
    // std::cout<<"final_id.size(name)"<<className[classId[final_id[0]]]<<std::endl;
    // std::cout<<"final_id.size(6)"<<final_id[6]<<std::endl;
    // std::cout<<"final_id.size(10)"<<final_id[10]<<std::endl;
    // std::cout<<"final_id.size(25)"<<final_id[25]<<std::endl;
    // std::cout<<"final_id.size(name)"<<className[classId[final_id[2]]]<<std::endl;
    // std::cout<<"type of final_id: "<<typeid(final_id).name()<<std::endl;
    
    for (int i = 0; i < final_id.size(); ++i){
        std::cout<<"final_id: "<<final_id[i]<<std::endl;
    }

    for (int i = 0; i < final_id.size(); ++i) {
        // cv::Rect resize_rect= origin_rect[final_id[i]];
        detectedObjects.push_back(Object {
            origin_rect_conf[final_id[i]],
            // "",resize_rect
            // className[classId[final_id[i]]],resize_rect
            className[classId[final_id[i]]],
            origin_rect[final_id[i]]
        });
        std::cout << "origin_rect_conf[final_id[i]]::" <<origin_rect_conf[final_id[i]] << std::endl;

        std::cout << "class_name::" <<className[classId[final_id[i]]] << std::endl;
    }
}

int main (int argc, char const *argv[]) {
    const std::string model_path = "/home/sensovision/Desktop/openvino_inferencing code/Openvino/openvino python inference/detection/tablet/best_tablet.xml";
    const std::string device = "CPU";

    // load the model
    InferenceEngine::ExecutableNetwork exec_net = load_model(model_path);

    // read the input image
    cv::Mat img = cv::imread("/home/sensovision/Desktop/openvino_inferencing code/Openvino/openvino python inference/detection/tablet/5.jpg");

    // clone the input image and resize it to match model input size (for plotting)
    cv::Mat clone = img.clone();
    cv::resize(clone, clone, cv::Size(640, 640));

    // initialise a vector to store the detected objects
    std::vector<Object> detectedObjects;

    // run the inference
    std::clock_t start = std::clock();
    pred(img, exec_net, 0.4, 0.5, detectedObjects);  //confidence_score and iou threshold
    std::clock_t end = std::clock();


    // loop through the detections and determine the final class label

    for (int i = 0; i < detectedObjects.size(); ++i) {
        std::string lbl = detectedObjects[i].name;        
        std::cout<<"label: " << lbl << std::endl;

        // extract the rectangle's coordinates (for plotting)
        int xmin = detectedObjects[i].rect.x;
        int ymin = detectedObjects[i].rect.y;
        int width = detectedObjects[i].rect.width;
        int height = detectedObjects[i].rect.height;

        std::cout<<"xmin: "<<xmin<<std::endl;
        std::cout<<"ymin: "<<ymin<<std::endl;
        std::cout<<"width: "<<width<<std::endl;
        std::cout<<"height: "<<height<<std::endl;

        // add the rectangle to the image (for plotting)
        cv::Rect rect(xmin, ymin, width, height);
        cv::rectangle(clone, rect, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0);
        cv::putText(clone,lbl,cv::Point(xmin,ymin-5),cv::FONT_HERSHEY_DUPLEX,0.4,cv::Scalar(0,255,0),0.5,false);

    }

    // (for plotting)
    cv::imshow("result", clone); 
    cv::waitKey(0);
}