//Create IE plug-in, query supporting hardware devices
Core ie;
vector<string> availableDevices = ie.GetAvailableDevices();
for (int i = 0; i <availableDevices.size(); i++) {
    printf("supported device name: %s/n", availableDevices[i].c_str());
}

//Load the detection model
auto network = ie.ReadNetwork("D:/python/yolov5/yolov5s.xml", "D:/python/yolov5/yolov5s.bin");
//auto network = ie.ReadNetwork("D:/python/yolov5/yolov5s.onnx");


//Set the input format
for (auto &item: input_info) {
    auto input_data = item.second;
    input_data->setPrecision(Precision::FP32);
    input_data->setLayout(Layout::NCHW);
    input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
}

//Set the output format
for (auto &item: output_info) {
    auto output_data = item.second;
    output_data->setPrecision(Precision::FP32);
}
auto executable_network = ie.LoadNetwork(network, "CPU");

int64 start = getTickCount();
/** Iterating over all input blobs **/
for (auto & item: input_info) {
    auto input_name = item.first;

   /** Getting input blob **/
    auto input = infer_request.GetBlob(input_name);
    size_t num_channels = input->getTensorDesc().getDims()[1];
    size_t h = input->getTensorDesc().getDims()[2];
    size_t w = input->getTensorDesc().getDims()[3];
    size_t image_size = h*w;
    Mat blob_image;
    resize(src, blob_image, Size(w, h));
    cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

   //NCHW
    float* data = static_cast<float*>(input->buffer());
    for (size_t row = 0; row <h; row++) {
        for (size_t col = 0; col <w; col++) {
            for (size_t ch = 0; ch <num_channels; ch++) {
                data[image_size*ch + row*w + col] = float(blob_image.at<Vec3b>(row, col)[ch])/255.0;
            }
        }
    }
}

//Perform prediction
infer_request.Infer();

for (int i = 0; i <side_square; ++i) {
    for (int c = 0; c <out_c; c++) {
        int row = i/side_h;
        int col = i% side_h;
        int object_index = c*side_data_square + row*side_data_w + col*side_data;

       //Threshold filtering
        float conf = sigmoid_function(output_blob[object_index + 4]);
        if (conf <0.25) {
            continue;
        }

       //parse cx, cy, width, height
        float x = (sigmoid_function(output_blob[object_index]) * 2-0.5 + col)*stride;
        float y = (sigmoid_function(output_blob[object_index + 1]) * 2-0.5 + row)*stride;
        float w = pow(sigmoid_function(output_blob[object_index + 2]) * 2, 2)*anchors[anchor_index + c * 2];
        float h = pow(sigmoid_function(output_blob[object_index + 3]) * 2, 2)*anchors[anchor_index + c * 2 + 1];
        float max_prob = -1;
        int class_index = -1;

       //parse category
        for (int d = 5; d <85; d++) {
            float prob = sigmoid_function(output_blob[object_index + d]);
            if (prob> max_prob) {
                max_prob = prob;
                class_index = d-5;
            }
        }

       //Convert to top-left, bottom-right coordinates
        int x1 = saturate_cast<int>((x-w/2) * scale_x);//top left x
        int y1 = saturate_cast<int>((y-h/2) * scale_y);//top left y
        int x2 = saturate_cast<int>((x + w/2) * scale_x);//bottom right x
        int y2 = saturate_cast<int>((y + h/2) * scale_y);//bottom right y

       //parse the output
        classIds.push_back(class_index);
        confidences.push_back((float)conf);
        boxes.push_back(Rect(x1, y1, x2-x1, y2-y1));
       //rectangle(src, Rect(x1, y1, x2-x1, y2-y1), Scalar(255, 0, 255), 2, 8, 0);
    }
}


vector<int> indices;
NMSBoxes(boxes, confidences, 0.25, 0.5, indices);
for (size_t i = 0; i <indices.size(); ++i)
{
    int idx = indices[i];
    Rect box = boxes[idx];
    rectangle(src, box, Scalar(140, 199, 0), 4, 8, 0);
}
float fps = getTickFrequency()/(getTickCount()-start);
float time = (getTickCount()-start)/getTickFrequency();
