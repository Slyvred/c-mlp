#pragma once
#include <vector>

typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX,
    LINEAR
}ActivationName;

struct LeakyRelu;
struct Softmax;

class Layer {
public:
    virtual void forward(const std::vector<float>& in) = 0;
    virtual void backward(const std::vector<float>& grad) = 0;
    virtual const std::vector<float>& get_outputs() const = 0;
    virtual ~Layer() = default;
};


template <typename Activation>
class DenseLayer : public Layer {
private:
    std::vector<float> deltas;
    std::vector<float> derivatives;
    std::vector<float> raw_outputs;
public:
    int n_inputs;
    int n_neurons;
    std::vector<float> outputs;
    std::vector<float> weights;
    std::vector<float> biases;
public:
    DenseLayer<Activation>(int n_inputs, int n_neurons);
    void forward(const std::vector<float> &inputs);
    void backward(const std::vector<float> &grad);
    const std::vector<float>& get_outputs() const;
};


class Model {
public:
    std::vector<Layer> layers;
    Model(std::vector<Layer> &layers);
    void forward(std::vector<float> &inputs);
    void backward(std::vector<float> &inputs);
    void train(std::vector<float> &inputs, std::vector<float> &target, float lr);
    ~Model();
};
