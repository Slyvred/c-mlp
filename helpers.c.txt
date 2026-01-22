#include "helpers.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void init_neuron(neuron* neuron, int n_weights) {
    neuron->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    neuron->output = 0;
    neuron->delta = 0;
    neuron->n_weights = n_weights;
    neuron->weights = malloc(n_weights * sizeof(double)); // allocate weights

    // Init weights
    for (int i = 0; i < n_weights; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
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

void get_layer_outputs(layer* layer, double* arr) {
    for (int i = 0; i < layer->n_neurons; i++) {
        neuron* n = &layer->neurons[i];
        arr[i] = n->output;
    }
}

int heaviside(double x) {
    return x > 0 ? 1 : 0;
}

double sigmoid(double x) {
    return (double)1 / ((double)1 + exp(-x));
}

double df_sigmoid(double x) {
    return x * ((double)1 - x);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double df_relu(double x) {
    return x > 0 ? 1 : 0;
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

void print_mlp(mlp *mlp) {
    for (int i = 0; i < mlp->n_hidden_layers; i++) {
        print_layer(mlp->hidden_layers[i]);
    }
    print_layer(&mlp->output_layer);
}

void print_outputs(layer* layer) {
    printf("Outputs: ");
    for (int i = 0; i < layer->n_neurons; i++) {
        printf("%f ", layer->neurons[i].output);
    }
    printf("\n");
}
