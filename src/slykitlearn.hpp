#pragma once
#include <cmath>
#include <vector>
#include <random>
#include "logger.hpp"

extern Logger logger;

struct LeakyRelu {
    static void f(const std::vector<float> &input, std::vector<float> &output) {
        for (int i = 0; i < input.size(); i++) {
            output[i] = input[i] >= 0 ? input[i] : 0.01 * input[i];
        }
        logger.log(Logger::LogLevel::DEBUG, "Calculated LeakyRelu");
    }

    static void df(const std::vector<float> &input, std::vector<float> &output) {
        for (int i = 0; i < input.size(); i++) {
            output[i] = input[i] >= 0 ? 1 : 0.01;
        }
    }
};


struct Softmax {
    static void f(const std::vector<float> &input, std::vector<float> &output) {
        float max_val = input[0];
        for (int i = 1; i < input.size(); i++) {
            if (input[i] > max_val) max_val = input[i];
        }

        float denom = 0;
        for (int i = 0; i < input.size(); i++) {
            output[i] = exp(input[i] - max_val);
            denom += output[i];
        }

        if (denom < 1e-20) denom = 1e-20;

        for (int i = 0; i < input.size(); i++) {
            output[i] /= denom;
        }
        logger.log(Logger::LogLevel::DEBUG, "Calculated Softmax");
    }

    static void df(const std::vector<float> &input, std::vector<float> &output) {
        for (int i = 0; i < input.size(); i++) {
            output[i] = 1;
        }
    }
};

class Layer {
public:
    virtual void forward(const std::vector<float>& in) = 0;
    virtual void backward(const std::vector<float> &grad_next_layer, float learning_rate) = 0;
    virtual const std::vector<float>& get_output() const = 0;
    virtual const std::vector<float>& get_delta() const = 0;
    virtual ~Layer() = default;
};


template <typename Activation>
class DenseLayer : public Layer {
private:
    int n_inputs;
    int n_neurons;

    std::vector<float> weights;
    std::vector<float> biases;

    std::vector<float> last_input;
    std::vector<float> raw_output;
    std::vector<float> output;
    std::vector<float> deltas;

    std::vector<float> derivatives_buf;
    std::vector<float> delta_neuron_buf;
    std::vector<float> grad_input_buf;
public:
    ~DenseLayer() = default;

    DenseLayer<Activation>(int n_neurons, int n_inputs) {
        this->n_inputs = n_inputs;
        this->n_neurons = n_neurons;
        this->weights.resize(n_inputs * n_neurons);
        this->deltas.resize(n_inputs);
        this->biases.resize(n_neurons);
        this->raw_output.resize(n_neurons);
        this->output.resize(n_neurons);
        this->derivatives_buf.resize(n_neurons);
        this->delta_neuron_buf.resize(n_neurons);
        this->grad_input_buf.resize(n_inputs);

        std::mt19937 gen(42);
        std::normal_distribution<float> d(0.0f, 0.1f);
        for(auto& w : weights) w = d(gen);
        for(auto& b : this->biases) b = 0.f;

        logger.log(Logger::LogLevel::DEBUG, "Created layer with %d neurons and %d inputs", this->n_neurons, this->n_inputs);
    }

    void forward(const std::vector<float> &input) override {
        this->last_input = input;
        logger.log(Logger::LogLevel::DEBUG, "Forwarding to layer...");

        for (int i = 0; i < this->n_neurons; i++) {
            const float* neuron_weights_ptr = &this->weights[i * this->n_inputs];
            float tmp = 0;
            for (int i = 0; i < input.size(); i++) {
                tmp += input[i] * neuron_weights_ptr[i];
            }
            tmp += this->biases[i];
            this->raw_output[i] = tmp;
        }
        Activation::f(this->raw_output, this->output);
    }

    void backward(const std::vector<float> &grad_next_layer, float learning_rate) override {
        Activation::df(this->raw_output, this->derivatives_buf);

        for(int i=0; i< n_neurons; i++) {
            delta_neuron_buf[i] = grad_next_layer[i] * derivatives_buf[i];
        }

        std::fill(grad_input_buf.begin(), grad_input_buf.end(), 0.0f); // Reset

        for (int i = 0; i < n_neurons; i++) {
            float d_neuron = delta_neuron_buf[i];
            float* w_ptr = &weights[i * n_inputs];

            for (int j = 0; j < n_inputs; j++) {
                grad_input_buf[j] += d_neuron * w_ptr[j];
                w_ptr[j] -= learning_rate * d_neuron * last_input[j];
            }
        }

        this->deltas = grad_input_buf;
    }

    const std::vector<float>& get_output() const override {
        return this->output;
    }

    const std::vector<float>& get_delta() const override {
        return this->deltas;
    }
};

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    void backward(const std::vector<float> &grad, float learning_rate);
public:
    Model() = default;
    ~Model() = default;
    std::vector<float> forward(const std::vector<float> &input);
    void train(std::vector<float> &target, float lr);
    const std::vector<float>& get_output() const;

    template<typename Activation>
    void add_layer(int n_neurons, int n_inputs) {
        this->layers.push_back(std::make_unique<DenseLayer<Activation>>(n_neurons, n_inputs));
        logger.log(Logger::LogLevel::DEBUG, "Added layer to model");
    }
};
