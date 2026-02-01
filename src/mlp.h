#pragma once
#include "math_functions.h"

typedef enum {
  RELU,
  SIGMOID,
  SOFTMAX,
  LINEAR
}FnName_t;

typedef enum {
    DENSE,
    CONV2D,
    POOLING,
    FLATTEN
}LayerType_t;

typedef struct {
    void (*f)(float* inputs, float* outputs, int len);
    void (*df)(float* inputs, float* outputs, int len);
    FnName_t function_name;
}Function_t;

extern Function_t sig;
extern Function_t lin;
extern Function_t rel;
extern Function_t softm;

// layer = list of neurons with activation function
typedef struct {
    LayerType_t type;
    int n_inputs;          // = Input shape = Number of weights
    int n_outputs;         // Number of neurons
    float* weights;        // List of all the weights of all the neurons in the layer
    float* biases;         // Bias of each neuron in the layer
    float* raw_outputs;    // Outputs of all the neurons
    float* outputs;        // Outputs of all the neurons after the activation function
    float* derivatives;    // Outputs of all the neurons after the activation function's derivative (for training)
    float* deltas;         // Error of each neuron (for training)
    Function_t* activation_function;
}Layer_t;

typedef struct {
    LayerType_t type;
    float* filters;         // Contains our convolution filters
    float* outpouts;        // Contains our feature maps
    int stride;             // Stride of filter (= step of the filter)
    int n_filters;          // Number of filters of the layer
    Vec2_t kernel_size;     // Size of the filter (width, height)
    Vec2_t input_size;      // Size of the input image (width, height, depth)
    Vec2_t output_size;     // Dimensions of the resulting feature maps
    Function_t* activation_function;
}Conv2DLayer_t;

typedef struct {
    LayerType_t type;
    int n_inputs;           // Number of feature maps we'll receive
    float* outpouts;        // Contains our pooled feature maps
    Vec2_t kernel_size;     // Size of the pooling filter (width, height)
    Vec2_t output_size;     // Dimensions of the resulting feature maps
    Vec2_t input_size;      // Size of the input feature map (width, height, depth)
}PoolingLayer_t;

typedef struct {
    Layer_t* layers;        // List of connected layers
    int n_layers;           // Total number of layers in the network
    // A created model is in stack (cf main.c) while a loaded one is entierly in heap (I mean the layers here).
    // We need to keep track of that in order to free the model's layers properly
    int is_in_heap;
} Model_t;

float ranged_rand(float min, float max);
Layer_t dense(int n_neurons, int n_inputs, Function_t *activation_function);
Conv2DLayer_t conv_2d(int n_filters, Vec2_t kernel_size, Vec2_t input_size, Function_t *activation_function);
PoolingLayer_t max_pool_2d(int n_inputs, Vec2_t input_size, Vec2_t kernel_size);
void convolve(Conv2DLayer_t* l, float* inputs);
void forward(Model_t *m, float* inputs, int n_inputs);
void train(Model_t *m, float* raw_inputs, float* target, float lr);
int get_num_parameters(Model_t* mlp);
void print_model(Model_t* m);
void print_output(Model_t *m, float* input, int input_len, float *expected, int expected_len);
void print_list(float* list, int len);
void free_model(Model_t* m);
void one_hot(float* output, int input, int n_classes);
void save_model(Model_t* m, const char* path);
void load_model(Model_t* m, const char* path);
