#pragma once

#include <torch/script.h>
#include <opencv2/opencv.hpp>

class TorchWrapper {
public:
    TorchWrapper(const std::string &model_path, const std::string &mean, const std::string &std, bool use_gpu = true);
    ~TorchWrapper() {}
    at::Tensor Forward(const cv::Mat& image);
private:
    std::vector<float> StringToVector(const std::string &str);
private:
    bool m_use_gpu;
    std::vector<float> m_mean;
    std::vector<float> m_std;
    std::shared_ptr<torch::jit::script::Module> m_model;
};
