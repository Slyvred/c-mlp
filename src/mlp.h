#pragma once

typedef enum {
  RELU,
  SIGMOID,
  SOFTMAX,
  LINEAR
}fn_name;

typedef struct {
    void (*f)(double* inputs, double* outputs, int len);
    void (*df)(double* inputs, double* outputs, int len);
    fn_name function_name;
}function;

extern function sig;
extern function lin;
extern function rel;
extern function softm;

typedef struct {
    // y = a_i * x_i + b
    double* weights;
    int n_weights;
    double output;
    double bias;
    double delta; // Neuron error
}neuron;

// layer = list of neurons with activation function
typedef struct {
    int n_neurons;
    neuron* neurons;
    function* activation_function;
    double* raw_outputs;
    double* outputs; // Output of each neuron of the layer
    double* derivatives;
}layer;

typedef struct {
    layer* layers; // list of layers
    int n_layers;
} MLP;


double ranged_rand(double min, double max);
void init_neuron(neuron* neuron, int n_parameters);
layer dense(int n_neurons, int n_parameters, function *activation_function);
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
