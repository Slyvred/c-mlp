#pragma once
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
    virtual void backward(const std::vector<float>& grad) = 0;
    virtual const std::vector<float>& get_output() const = 0;
    virtual ~Layer() = default;
};


template <typename Activation>
class DenseLayer : public Layer {
private:
    int n_inputs;
    int n_neurons;
    std::vector<float> derivatives;
    std::vector<float> raw_output;
    std::vector<float> deltas;
    std::vector<float> last_input;
    std::vector<float> output;
    std::vector<float> weights;
    std::vector<float> biases;
public:
    DenseLayer<Activation>(int n_neurons, int n_inputs) {
        this->n_inputs = n_inputs;
        this->n_neurons = n_neurons;
        this->weights.resize(n_inputs * n_neurons);
        this->biases.resize(n_neurons);
        this->raw_output.resize(n_neurons);
        this->output.resize(n_neurons);

        std::mt19937 gen(42);
        std::normal_distribution<float> d(0.0f, 0.1f); // Moyenne 0, écart-type 0.1
        for(auto& w : this->weights) w = d(gen);
        for(auto& b : this->biases) b = 0.f;

        logger.log(Logger::LogLevel::DEBUG, "Created layer with %d neurons and %d inputs", this->n_neurons, this->n_inputs);
    }

    void forward(const std::vector<float> &input) {
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

    void backward(const std::vector<float> &grad) {}

    const std::vector<float>& get_output() const {
        return this->output;
    }

    ~DenseLayer() = default;
};


class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    Model() = default;
    std::vector<float> forward(const std::vector<float> &input) {
        const std::vector<float>* curr_input = &input;
        for (auto &layer : this->layers) {
            layer->forward(*curr_input);
            curr_input = &layer->get_output();
        }
        return *curr_input;
    }

    void backward(std::vector<float> &input) {
        // // Calculate output layer error
        // Layer_t* l = &m->layers[m->n_layers - 1];
        // l->activation_function->df(l->raw_outputs, l->derivatives, l->n_outputs);

        // for (int i = 0; i < l->n_outputs; i++) {
        //     l->deltas[i] = (l->outputs[i] - target[i]) * l->derivatives[i];
        // }

        // // Calculate layer error from last hidden layer to input
        // for (int i = m->n_layers - 2; i >= 0; i--) {
        //     Layer_t* curr_layer = &m->layers[i];
        //     Layer_t* next_layer = &m->layers[i + 1];

        //     curr_layer->activation_function->df(curr_layer->raw_outputs, curr_layer->derivatives, curr_layer->n_outputs);

        //     #pragma omp parallel for
        //     for (int j = 0; j < curr_layer->n_outputs; j++) {
        //         float error = 0;
        //         // Sum deltas of next layer weighted by next neuron weights
        //         for (int k = 0; k < next_layer->n_outputs; k++) {
        //             // next_layer->weights[j] is the weight connecting the current neuron with the next neuron k
        //             int weight_index = k * next_layer->n_inputs + j;
        //             error += next_layer->deltas[k] * next_layer->weights[weight_index];
        //         }
        //         curr_layer->deltas[j] = error * curr_layer->derivatives[j];
        //     }
        // }
    }

    void train(std::vector<float> &input, std::vector<float> &target, float lr);

    template<typename Activation>
    void add_layer(int n_neurons, int n_inputs) {
        this->layers.push_back(std::make_unique<DenseLayer<Activation>>(n_neurons, n_inputs));
        logger.log(Logger::LogLevel::DEBUG, "Added layer to model");
    }

    const std::vector<float>& get_output() const {
        auto &output_layer = *this->layers.at(this->layers.size() - 1);
        return output_layer.get_output();
    }

    ~Model() = default;
};
