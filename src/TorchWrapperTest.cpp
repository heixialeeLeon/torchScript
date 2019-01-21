#include "TorchWrapper.hpp"

int main()
{
    TorchWrapper wrapper("../test/crnn.pt","0.5,0.5,0.5,","0.5,0.5,0.5");
    cv::Mat img(30, 200, CV_8UC3);
    at::Tensor output = wrapper.Forward(img);
    std::cout<<output.slice(0,0,5)<<std::endl;
    std::cout<<output.sizes()<<std::endl;
}
