#include "slykitlearn.hpp"
#include <stdexcept>
#include <vector>

#define LEAKY_RELU_SLOPE 0.01

template <typename T>
struct LeakyRelu {
    static void f(const std::vector<T> &inputs, std::vector<T> &outputs) {
        for (int i = 0; i < inputs.size(); i++) {
            outputs[i] = inputs[i] >= 0 ? inputs[i] : LEAKY_RELU_SLOPE * inputs[i];
        }
    }

    static void df(const std::vector<T> &inputs, std::vector<T> &outputs) {
        for (int i = 0; i < inputs.size(); i++) {
            outputs[i] = inputs[i] >= 0 ? 1 : LEAKY_RELU_SLOPE;
        }
    }
    static constexpr ActivationName name = RELU;
};

template <typename T>
struct Softmax {
    static void f(const std::vector<T> &inputs, std::vector<T> &outputs) {
        T max_val = inputs[0];
        for (int i = 1; i < inputs.size(); i++) {
            if (inputs[i] > max_val) max_val = inputs[i];
        }

        T denom = 0;
        for (int i = 0; i < inputs.size(); i++) {
            outputs[i] = exp(inputs[i] - max_val);
            denom += outputs[i];
        }

        if (denom < 1e-20) denom = 1e-20;

        for (int i = 0; i < inputs.size(); i++) {
            outputs[i] /= denom;
        }
    }

    static void df(const std::vector<T> &inputs, std::vector<T> &outputs) {
        for (int i = 0; i < inputs.size(); i++) {
            outputs[i] = 1;
        }
    }

    static constexpr ActivationName name = SOFTMAX;
};

template<typename T>
T weighted_sum(std::vector<T> &inputs, std::vector<T> &weights, T bias) {
    if (inputs.size() != weights.size()) throw std::runtime_error("The number of inputs should be same as the number of outputs");
    T output = 0;

    for (int i = 0; i < inputs.size(); i++) {
        output += inputs[i] * weights[i];
    }
    output += bias;
    return output;
}

template<typename T, typename Activation>
DenseLayer<T, Activation>::DenseLayer(int n_inputs, int n_neurons) {
    this->n_inputs = n_inputs;
    this->n_neurons = n_neurons;
}

template<typename T, typename Activation>
void DenseLayer<T, Activation>::forward(std::vector<T> &inputs) {
    // For each neuron in the layer
    for (int i = 0; i < this->n_neurons; i++) {
        std::vector<T> &neuron_weights = this->weights + i * this->n_inputs;
        this->raw_outputs[i] = weighted_sum(inputs, neuron_weights, this->biases[i]);
    }
    Activation::f(this->raw_outputs, this->outputs);
}
