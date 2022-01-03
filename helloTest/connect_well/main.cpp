
int main() 
{
    const std::string model_path = "/home/sensovision/Downloads/openvino_connect_well/openvino_connect_well/classification.xml";
    const std::string device = "CPU";

    // load the model
    InferenceEngine::ExecutableNetwork net = load_model(model_path);   

    // read the input image
    cv::Mat img = cv::imread("/home/sensovision/Desktop/Rajesh/Rajesh/datasets/connect_well/combined_unseen_data/Good10.jpg");

    // call the prediction function
    std::clock_t start = std::clock();
    int lbl = pred(img, net, std::make_pair(300, 300));
    std::clock_t end = std::clock();

    if(lbl == 1)
        std::cout << "[INFO] given image does not contain any defect" << std::endl;
    else if(lbl == 0)
        std::cout << "[INFO] given image has defects" << std::endl;
    else
        std::cout << "[ERROR] unable to process" << std::endl;

    std::cout << "[INFO] total inference time = " << (end - start) / (double) CLOCKS_PER_SEC << " seconds";
}