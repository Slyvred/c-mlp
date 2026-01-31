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
    void (*f)(float* inputs, float* outputs, int len);
    void (*df)(float* inputs, float* outputs, int len);
    fn_name function_name;
}function;

extern function sig;
extern function lin;
extern function rel;
extern function softm;

// layer = list of neurons with activation function
typedef struct {
    layer_type type;
    int n_inputs;          // = Input shape = Number of weights
    int n_outputs;         // Number of neurons
    float* weights;        // List of all the weights of all the neurons in the layer
    float* biases;         // Bias of each neuron in the layer
    float* raw_outputs;    // Outputs of all the neurons
    float* outputs;        // Outputs of all the neurons after the activation function
    float* derivatives;    // Outputs of all the neurons after the activation function's derivative (for training)
    float* deltas;         // Error of each neuron (for training)
    function* activation_function;
}layer;

typedef struct {
    layer* layers;      // List of connected layers
    int n_layers;       // Total number of layers in the network
    // A created model is in stack (cf main.c) while a loaded one is entierly in heap (I mean the layers here).
    // We need to keep track of that in order to free the model's layers properly
    int is_in_heap;
} MLP;


float ranged_rand(float min, float max);
layer dense(int n_neurons, int n_inputs, function *activation_function);
void forward(MLP *m, float* inputs, int n_inputs);
void train(MLP *m, float* raw_inputs, float* target, float lr);
int get_num_parameters(MLP* mlp);
void print_model(MLP* m);
void print_output(MLP *m, float* input, int input_len, float *expected, int expected_len);
void print_list(float* list, int len);
void free_model(MLP* m);
void one_hot(float* output, int input, int n_classes);
void save_model(MLP* m, const char* path);
void load_model(MLP* m, const char* path);
