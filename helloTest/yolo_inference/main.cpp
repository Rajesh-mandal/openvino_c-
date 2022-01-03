// #ifndef DETECTOR_H
// #define DETECTOR_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
using namespace std;
using namespace cv;
using namespace InferenceEngine;



// Note that the threshold here is the threshold of the product of the box and the object prob
bool parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
    vector<Rect>& o_rect,vector<float>& o_rect_cof)
    {
    vector<Rect>& o_rect,vector<float>& o_rect_cof,
    vector<int> &classId){
    vector<int> anchors = get_anchors(net_grid);
   LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
   const float *output_blob = blobMapped.as<float *>();
   // 80 classes are 85, one class is 6, and n classes are n+5
   //int item_size = 6;
   int item_size = 85;
    size_t anchor_n = 3;
    for(int n=0;n<anchor_n;++n)
        for(int i=0;i<net_grid;++i)
            for(int j=0;j<net_grid;++j)
            {
                double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 4];
                box_prob = sigmoid(box_prob);
                //If the box confidence is not satisfied, the overall confidence is not satisfied
                if(box_prob < cof_threshold)
                    continue;
                
                // Note that the output here is the center point coordinates, which needs to be converted into corner point coordinates
                double x = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0];
                double y = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1];
                double w = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2];
                double h = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 3];

                double max_prob = 0;
                int idx=0;
                for(int t=5;t<85;++t){
                    double tp= output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ t];
                    tp = sigmoid(tp);
                    if(tp > max_prob){
                        max_prob = tp;
                        idx = t;
                    }
                }
                // for(int t=5;t<85;++t){
                //     double tp= output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ t];
                //     tp = sigmoid(tp);
                //     if(tp > max_prob){
                //         max_prob = tp;
                //         idx = t;
                //     }
                // }

                for (int t = 0; t < item_size; ++t) {
					double tp = sigmoid(output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + t]);
					if (tp > max_prob) {
						max_prob = tp;
						idx = t-5;
					}
				}
                float cof = box_prob * max_prob;                
                // For borders whose border confidence is less than the threshold, do not care about other values, do not perform calculations to reduce the amount of calculation
                if(cof < cof_threshold)
                    continue;
                x = (sigmoid(x)*2 - 0.5 + j)*640.0f/net_grid;
                y = (sigmoid(y)*2 - 0.5 + i)*640.0f/net_grid;
                w = pow(sigmoid(w)*2,2) * anchors[n*2];
                h = pow(sigmoid(h)*2,2) * anchors[n*2 + 1];
                double r_x = x - w/2;
                double r_y = y - h/2;
                Rect rect = Rect(round(r_x),round(r_y),round(w),round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                classId.push_back(idx);
            }
    if(o_rect.size() == 0) return false;
    else  return  true ;
}
// Initialization
bool init(string xml_path,double cof_threshold,double nms_area_threshold){
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 
    // Enter settings
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    // Output settings
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    // Get executable network
    //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}
// Release resources
bool Detector::uninit(){
    return true;
}
// Process the image to get the result
bool process_frame(Mat& inframe,vector<Object>& detected_objects){
    if(inframe.empty()){
        cout << " Invalid picture input " << endl;
        return false;
    }
    resize(inframe,inframe,Size(640,640));
    cvtColor(inframe,inframe,COLOR_BGR2RGB);
    size_t img_size = 640*640;
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    // nchw
    for(size_t row =0;row<640;row++){
        for(size_t col=0;col<640;col++){
            for(size_t ch =0;ch<3;ch++){
                blob_data[img_size*ch + row*640 + col] = float(inframe.at<Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }
    // Perform prediction
    infer_request->Infer();
    // Get the results of each layer
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    vector<int> classId;
    int s[3] = {80,40,20};
    int i=0;
    for (auto &output : _outputinfo) {
        // Temporary vector used to save the analysis results
		vector<cv::Rect> origin_rect_temp;
		vector<float> origin_rect_cof_temp;
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
       parse_yolov5(blob,s[i],_cof_threshold,origin_rect,origin_rect_cof);
        parse_yolov5(blob,s[i],_cof_threshold,origin_rect,origin_rect_cof, classId);
        origin_rect.insert(origin_rect.end(), origin_rect_temp.begin(), origin_rect_temp.end());
		origin_rect_cof.insert(origin_rect_cof.end(), origin_rect_cof_temp.begin(), origin_rect_cof_temp.end());

        ++i;
    }
    // Post-processing to obtain the final test result
    vector<int> final_id;
    dnn::NMSBoxes(origin_rect,origin_rect_cof,_cof_threshold,_nms_area_threshold,final_id);
    // Get the final result according to final_id
    for(int i=0;i<final_id.size();++i){
        Rect resize_rect= origin_rect[final_id[i]];
        detected_objects.push_back(Object{
            origin_rect_cof[final_id[i]],
            "",resize_rect
            className[classId[final_id[i]]],resize_rect
        });
        cout << className[classId[final_id[i]]] << endl;
    }
    return true;
}

// //The following are tool functions
// double Detector::sigmoid(double x){
//     return (1 / (1 + exp(-x)));
// }
// vector<int> Detector::get_anchors(int net_grid){
//     vector<int> anchors(6);
//     int a80[6] = {10,13, 16,30, 33,23};
//     int a40[6] = {30,61, 62,45, 59,119};
//     int a20[6] = {116,90, 156,198, 373,326}; 
//     if(net_grid == 80){
//         anchors.insert(anchors.begin(),a80,a80 + 6);
//     }
//     else if(net_grid == 40){
//         anchors.insert(anchors.begin(),a40,a40 + 6);
//     }
//     else if(net_grid == 20){
//         anchors.insert(anchors.begin(),a20,a20 + 6);
//     }
//     return anchors;
// }


public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;


    bool init(string xml_path,double cof_threshold,double nms_area_threshold);
   
    bool uninit();
 
    bool process_frame(Mat& inframe,vector<Object> &detected_objects);
private:
    double sigmoid(double x);
    vector<int> get_anchors(int net_grid);
    bool parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
        vector<Rect>& o_rect,vector<float>& o_rect_cof);
        vector<Rect>& o_rect,vector<float>& o_rect_cof, vector<int> &classId);
    Rect detet2origin(const Rect& dete_rect,float rate_to,int top,int left);
   
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    string _input_name;

    string _xml_path;                         
    double _cof_threshold;               
    double _nms_area_threshold;  
    string className[6] = {"MP15N","C9783","MP15","MG4730","UP15CS0090","DL2CAW9699" };


int main(int argc, char const *argv[])
{
    Detector* detector = new Detector;
    string xml_path = "../res/yolov5s.xml";
    // string xml_path = "../weights/yolov5s.xml";
    detector->init(xml_path,0.1,0.5);
    /*
    VideoCapture capture;
    capture.open(0);
    Mat src;
    while(1){
        capture >> src;
        vector<Detector::Object> detected_objects;
    detector->process_frame(src,detected_objects);
    for(int i=0;i<detected_objects.size();++i){
         int xmin = detected_objects[i].rect.x;
        int ymin = detected_objects[i].rect.y;
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
        cv::rectangle(src, rect, Scalar(255, 0, 0),1, LINE_8,0);
    }
        imshow("cap",src);
        waitKey(1);
    }
    */
    Mat src = imread("../res/bus.jpg");
    Mat osrc = src.clone();
    resize(osrc,osrc,Size(640,640));
    vector<Detector::Object> detected_objects;
    auto start = chrono::high_resolution_clock::now();
    detector->process_frame(src,detected_objects);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<<"use "<<diff.count()<<" s" << endl;
    for(int i=0;i<detected_objects.size();++i){
         int xmin = detected_objects[i].rect.x;
        int ymin = detected_objects[i].rect.y;
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(xmin, ymin, width, height);
        cv::rectangle(osrc, rect, Scalar(0, 0, 255),1, LINE_8,0);
        cout << "detected_objects[i].name：" << detected_objects[i].name << endl;
        putText(osrc, detected_objects[i].name,
				cv::Point(xmin, ymin - 10),
				cv::FONT_HERSHEY_SIMPLEX,
				0.7,
				cv::Scalar(0, 255, 0));
    }
    putText(osrc, "" + to_string(diff.count()),
			cv::Point(5, 20),
			cv::FONT_HERSHEY_SIMPLEX,
			0.5,
			cv::Scalar(0, 0, 0));
    imshow("result",osrc);
    waitKey(0);
}