#pragma once

typedef enum {
  RELU,
  SIGMOID,
  SOFTMAX,
  LINEAR
}fn_name;

typedef enum {
    DENSE,
    CONV2D,
    POOLING,
    FLATTEN
}layer_type;

typedef struct {
    void (*f)(double* inputs, double* outputs, int len);
    void (*df)(double* inputs, double* outputs, int len);
    fn_name function_name;
}function;

extern function sig;
extern function lin;
extern function rel;
extern function softm;

// layer = list of neurons with activation function
typedef struct {
    layer_type type;
    int n_inputs;           // = Input shape = Number of weights
    int n_outputs;          // Number of neurons
    double* weights;        // List of all the weights of all the neurons in the layer
    double* biases;         // Bias of each neuron in the layer
    double* raw_outputs;    // Outputs of all the neurons (buffer for training)
    double* outputs;        // Outputs of all the neurons after the activation function
    double* derivatives;    // Outputs of all the neurons after the activation function's derivative
    double* deltas;         // Error of each neuron
    function* activation_function;
}layer;

typedef struct {
    layer* layers; // list of layers
    int n_layers;
} MLP;


double ranged_rand(double min, double max);
layer dense(int n_neurons, int n_inputs, function *activation_function);
void forward(MLP *m, double* inputs, int n_inputs);
void train(MLP *m, double* raw_inputs, double* target, double lr);
int get_num_parameters(MLP* mlp);
void print_model(MLP* m);
void print_output(MLP *m, double* input, int input_len, double *expected, int expected_len);
void print_list(double* list, int len);
void free_model(MLP* m);
void one_hot(double* output, int input, int n_classes);
void save_model(MLP* m, const char* path);
void load_model(MLP* m, const char* path);
