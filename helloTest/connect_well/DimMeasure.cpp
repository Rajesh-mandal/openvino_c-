#include "DimMeasure.h"

int pred(cv::Mat img, InferenceEngine::ExecutableNetwork exec_net, std::pair<int, int> img_size)
 {
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

