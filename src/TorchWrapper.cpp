#include "TorchWrapper.hpp"

TorchWrapper::TorchWrapper(const std::string &model_path, const std::string &mean, const std::string &std,
                           bool use_gpu) {
    // Must load model in cpu mode to avoid corrupted RNN parameters
    m_model = torch::jit::load(model_path, torch::kCPU);
    m_mean = StringToVector(mean);
    m_std = StringToVector(std);
    m_use_gpu = use_gpu;
    if (m_use_gpu)
        m_model->to(at::kCUDA);
}

std::vector<float> TorchWrapper::StringToVector(const std::string &str) {
    std::vector<float> values;
    if (!str.empty()) {
        std::stringstream ss(str);
        std::string item;
        while (getline(ss, item, ',')) {
            float value = static_cast<float>(std::atof(item.c_str()));
            values.push_back(value);
        }
    }
    return values;
}

at::Tensor TorchWrapper::Forward(const cv::Mat &image) {
    // Load tensor from cv::Mat, shape is (H, W, C)
    std::vector<int64_t> sizes = {image.rows, image.cols, image.channels()};
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kByte);
    at::Tensor tensor_image = torch::from_blob(image.data, torch::IntList(sizes), options);
    // Permute tensor, shape is (C, H, W)
    tensor_image = tensor_image.permute({2, 0, 1});
    // Convert tensor dtype to float32, and range from [0, 255] to [0, 1],  act as transforms.ToTensor()
    tensor_image = tensor_image.toType(torch::ScalarType::Float).div(255.0f);
    // Subtract mean and Divide std act as transforms.Normalize()
    // Subtract mean value
    for (int i = 0; i < MIN(m_mean.size(), tensor_image.size(0)); i++) {
        tensor_image[i] = tensor_image[i].sub(m_mean[i]);
    }
    // Divide by std value
    for (int i = 0; i < MIN(m_std.size(), tensor_image.size(0)); i++) {
        tensor_image[i] = tensor_image[i].div(m_std[i]);
    }
    // Reshape to (1, C, H, W)
    tensor_image = tensor_image.unsqueeze(0);
    // Convert to cuda
    if (m_use_gpu)
        tensor_image = tensor_image.to(at::kCUDA);
    at::Tensor result = m_model->forward({tensor_image}).toTensor();
    return result;
}
