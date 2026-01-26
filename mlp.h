#pragma once

typedef struct {
    double (*f)(double);
    double (*df)(double);
}function;

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
    double* outputs; // Output of each neuron of the layer
}layer;

double sigmoid(double x);
double sigmoid_deriv(double x);
double linear(double x);
double linear_deriv(double x);
double ranged_rand(double min, double max);


typedef struct {
    layer* layers; // list of layers
    int n_layers;
} MLP;

double sum(double inputs[], double weights[], double bias, int len);
void init_neuron(neuron* neuron, int n_parameters);
layer dense(int n_neurons, int n_parameters, function *activation_function);
void forward(MLP *m, double* inputs, int n_inputs);
void train(MLP *m, double* raw_inputs, double* target, double lr);
int get_num_parameters(MLP* mlp);
void print_model(MLP* m);
void print_output(MLP *m, double* input, int input_len, double *expected, int expected_len);
void print_list(double* list, int len);
