#pragma once

typedef enum {
  RELU,
  SIGMOID,
  SOFTMAX,
  LINEAR
}fn_name;

typedef struct {
    void (*f)(float* inputs, float* outputs, int len);
    void (*df)(float* inputs, float* outputs, int len);
    fn_name function_name;
}function;

extern function sig;
extern function lin;
extern function rel;
extern function softm;

typedef struct {
    // y = a_i * x_i + b
    float* weights;
    int n_weights;
    float output;
    float bias;
    float delta; // Neuron error
}neuron;

// layer = list of neurons with activation function
typedef struct {
    int n_neurons;
    neuron* neurons;
    function* activation_function;
    float* raw_outputs;
    float* outputs; // Output of each neuron of the layer
    float* derivatives;
}layer;

typedef struct {
    layer* layers; // list of layers
    int n_layers;
} MLP;


float ranged_rand(float min, float max);
void init_neuron(neuron* neuron, int n_parameters);
layer dense(int n_neurons, int n_parameters, function *activation_function);
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
