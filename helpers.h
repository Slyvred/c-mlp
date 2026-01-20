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

void init_neuron(neuron* neuron, int n_weights);

void init_layer(layer* layer, int n_neurones, int n_weights);

double abs_double(double x);

int heaviside(double x);

double relu(double x);

double df_relu(double x);

double sigmoid(double x);

double df_sigmoid(double x);

double sum(double inputs[], double weights[], double bias, int len);

void print_layer(layer* layer);

void print_neuron(neuron* neuron);
