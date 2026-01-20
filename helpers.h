#pragma once

typedef struct {
    double* weights;
    int n_weights;
    double bias;
}neuron;

typedef struct {
    neuron* neurons;
    int n_neurons;
}layer;

void init_neuron(neuron* neuron, int n_weights);

void init_layer(layer* layer, int n_neurones, int n_weights);

int heaviside(double x);

double sigmoid(double x);

double sum(double inputs[], double weights[], double bias, int len);

void print_layer(layer* layer);

void print_neuron(neuron* neuron);
