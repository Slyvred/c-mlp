#pragma once

typedef struct {
    double* weights;
    int n_weights;
    double bias;
    double output;
    double delta;
}neuron;

typedef struct {
    neuron* neurons;
    int n_neurons;
}layer;

typedef struct {
    layer** hidden_layers;
    int n_hidden_layers;
    layer output_layer;
}mlp;

void init_neuron(neuron* neuron, int n_weights);

void init_layer(layer* layer, int n_neurones, int n_weights);

int heaviside(double x);

double relu(double x);

double df_relu(double x);

double sigmoid(double x);

double df_sigmoid(double x);

double sum(double inputs[], double weights[], double bias, int len);

void get_layer_outputs(layer* layer, double* arr);

void print_layer(layer* layer);

void print_neuron(neuron* neuron);

void print_mlp(mlp* mlp);

void print_outputs(layer* layer);
