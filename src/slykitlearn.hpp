#pragma once
#include <vector>

typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX,
    LINEAR
}ActivationName;

template <typename T>
struct LeakyRelu;

template <typename T>
struct Softmax;

template <typename T>
class Layer {
public:
    virtual void forward(const std::vector<T>& in) = 0;
    virtual void backward(const std::vector<T>& grad) = 0;
    virtual const std::vector<T>& output() const = 0;
    virtual ~Layer() = default;
};


template <typename T, typename Activation>
class DenseLayer {
private:
    std::vector<T> deltas;
    std::vector<T> derivatives;
    std::vector<T> raw_outputs;
public:
    int n_inputs;
    int n_neurons;
    std::vector<T> outputs;
    std::vector<T> weights;
    std::vector<T> biases;
public:
    DenseLayer<T, Activation>(int n_inputs, int n_neurons);
    void forward(const std::vector<T> &inputs);
    void backward(const std::vector<T> &grad);
    const std::vector<T>& output() const;
    std::vector<T>& get_outputs();
};


template <typename T, typename... Layers> class Model {
public:
    std::tuple<Layers...> layers;
    Model(std::tuple<Layers...> &layers);
    void forward(std::vector<T> &inputs);
    void backward(std::vector<T> &inputs);
    void train(std::vector<T> &inputs, std::vector<T> &target, T lr);
    ~Model();
};
