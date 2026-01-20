#include "helpers.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void init_neuron(neuron* neuron, int n_weights) {
    neuron->bias = (double)rand() / RAND_MAX;
    neuron->n_weights = n_weights;
    neuron->weights = malloc(n_weights * sizeof(double)); // allocate weights

    // Init weights
    for (int i = 0; i < n_weights; i++) {
        neuron->weights[i] = (double)rand() / RAND_MAX;
    }
}

void init_layer(layer* layer, int n_neurons, int n_weights) {
    layer->n_neurons = n_neurons;
    layer->neurons = malloc(n_neurons * sizeof(neuron)); // allocate neurons

    // Init neurons
    for (int i = 0; i < layer->n_neurons; i++) {
        init_neuron(&layer->neurons[i], n_weights);
    }
}

int heaviside(double x) {
    return x > 0 ? 1 : 0;
}

double sigmoid(double x) {
    return (double)1 / ((double)1 + exp(-x));
}

double d_sigmoid(double x) {
    return sigmoid(x) * ((double)1 - sigmoid(x));
}

double sum(double inputs[], double weights[], double bias, int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    return sum;
}

void print_layer(layer *layer) {
    printf("Layer:\n  - Number of neurons: %d\n", layer->n_neurons);
    for (int i = 0; i < layer->n_neurons; i++) {
        neuron *neuron = &layer->neurons[i];
        print_neuron(neuron);
    }
    printf("\n");
}

void print_neuron(neuron *neuron) {
    printf("    - Neuron:\n");
    printf("      - Weights: %d\n      - Bias: %9f\n", neuron->n_weights, neuron->bias);
    for (int j = 0; j < neuron->n_weights; j++) {
        printf("      - W%d: %11f\n", j, neuron->weights[j]);
    }
}
