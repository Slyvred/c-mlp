#pragma once

typedef struct {
    void (*f)(double* inputs, double* outputs, int len);
    void (*df)(double* inputs, double* outputs, int len);
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
    double* raw_outputs;
    double* outputs; // Output of each neuron of the layer
    double* derivatives;
}layer;

typedef struct {
    layer* layers; // list of layers
    int n_layers;
} MLP;

double sigmoid(double x);
double sigmoid_deriv(double x);
void linear(double* inputs, double* outputs, int len);
void linear_deriv(double* inputs, double* outputs, int len);
double ranged_rand(double min, double max);

void leaky_relu(double* inputs, double* outputs, int len);
void leaky_relu_deriv(double* inputs, double* outputs, int len);
void softmax(double* inputs, double* outputs, int len);
double sum(double inputs[], double weights[], double bias, int len);
void normalize(double* values, int length, double max);
void denormalize(double* values, int length, double max);
void init_neuron(neuron* neuron, int n_parameters);
layer dense(int n_neurons, int n_parameters, function *activation_function);
double mse(double* predicted, double* actual, int length);
void forward(MLP *m, double* inputs, int n_inputs);
void train(MLP *m, double* raw_inputs, double* target, double lr);
int get_num_parameters(MLP* mlp);
void print_model(MLP* m);
void print_output(MLP *m, double* input, int input_len, double *expected, int expected_len);
void print_list(double* list, int len);
double* one_hot(int input, int n_classes);
int index_of_max(double* array, int len);
