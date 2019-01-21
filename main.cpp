#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

int createTensor(at::Tensor& tensor)
{
    cv::Mat img(30,200,CV_8UC3);
    //std::cout<<img.size()<<std::endl;
    std::vector<int64_t > sizes = {1, 3, img.rows, img.cols};
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    tensor = torch::from_blob(img.data, torch::IntList(sizes), options);
    tensor = tensor.to(at::kCUDA);
    return 0;
}

void process(at::Tensor& tensor)
{
    std::string model_file = "../test/crnn.pt";
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_file);
    assert(module != nullptr);
    module->to(at::kCUDA);
    at::Tensor output = module->forward({tensor}).toTensor();
    std::cout<<output.slice(0,0,5)<<std::endl;
    std::cout<<output.sizes()<<std::endl;
}

int main(int argc, const char* argv[])
{
    at::Tensor input_tensor;
    createTensor(input_tensor);
    process(input_tensor);
}