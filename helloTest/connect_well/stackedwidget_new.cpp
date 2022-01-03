
#include "stackedwidget_new.h"
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