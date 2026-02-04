#include <vector>
#include "logger.hpp"
#include "slykitlearn.hpp"

extern Logger logger;

void Model::backward(const std::vector<float> &grad, float learning_rate) {
    const std::vector<float>* curr_grad = &grad;
    for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
        auto curr_layer = it->get();
        curr_layer->backward(*curr_grad, learning_rate);
        curr_grad = &curr_layer->get_delta();
    }
}

std::vector<float> Model::forward(const std::vector<float> &input) {
    const std::vector<float>* curr_input = &input;
    for (auto &layer : this->layers) {
        layer->forward(*curr_input);
        curr_input = &layer->get_output();
    }
    return *curr_input;
}

void Model::train(std::vector<float> &target, float lr) {
    const auto& output = this->get_output();
    std::vector<float> grad(output.size());

    for (int i = 0; i < output.size(); i++) {
        grad[i] = output[i] - target[i];
    }

    this->backward(grad, lr);
}

const std::vector<float>& Model::get_output() const {
    auto* output_layer = &this->layers.back();
    return output_layer->get()->get_output();
}
