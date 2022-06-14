/*//#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

void tst_torch()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}*/


#include <torch/script.h> // One-stop header.
//#include <torch/torch.h>

#include <iostream>
#include <memory>

void tst_torch() {
    
  // Deserialize the ScriptModule from a file using torch::jit::load().
  //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("/home/thermalview/Desktop/ThermalView/face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth");
  //torch::jit::script::Module module = torch::jit::load("/home/thermalview/Desktop/ThermalView/face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth");
  torch::pickle_load("/face_detection/yolact/weights/yolact_maskfacesnewwork0312added1ffw1a1vk1exp1nbgray640_64_146000.pth");
  //assert(module != nullptr);
  std::cout << "ok\n";

torch::Device device = torch::Device("cuda");
torch::Device device2 = torch::Device(c10::DeviceType::CUDA,0);
//auto res = torch::cuda::cudnn_is_available();

// https://habr.com/ru/company/ods/blog/480328/
//torch::NoGradGuard no_grad;
//    torch::Tensor tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
//    auto output = module.forward( { tensor } );
//    float* data = static_cast<float*>(output.toTensor().data_ptr());

//    // Create a vector of inputs.
//    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(torch::ones({1, 3, 224, 224}));
//
//    // Execute the model and turn its output into a tensor.
//    at::Tensor output = module->forward(inputs).toTensor();
//
//    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}