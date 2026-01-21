#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "helpers.h"

typedef struct{
    double X;
    double c[4];
}binary;


int main(int argc, char** argv) {
    srand(time(NULL));

    // Binary
    binary input[16] = {
        //   8  4  2  1
        {0, {0, 0, 0, 0}},
        {1, {0, 0, 0, 1}},
        {2, {0, 0, 1, 0}},
        {3, {0, 0, 1, 1}},
        {4, {0, 1, 0, 0}},
        {5, {0, 1, 0, 1}},
        {6, {0, 1, 1, 0}},
        {7, {0, 1, 1, 1}},
        {8, {1, 0, 0, 0}},
        {9, {1, 0, 0, 1}},
        {10, {1, 0, 1, 0}},
        {11, {1, 0, 1, 1}},
        {12, {1, 1, 0, 0}},
        {13, {1, 1, 0, 1}},
        {14, {1, 1, 1, 0}},
        {15, {1, 1, 1, 1}},
    };

    layer perceptron;
    init_layer(&perceptron, 4, 1);
    int epochs = 32;
    double learning_rate = 0.1;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        learning_rate = atof(argv[2]);
    }
    print_layer(&perceptron);
    printf("--- Training ---\n");
    int i = 0;
    while (i < epochs) { // 1 epoch = running the perceptron trough all examples
        for (int j = 0; j < 16; j++) {
            binary sample = input[j];
            double output[4];
            for (int k = 0; k < perceptron.n_neurons; k++) {
                neuron* n = &perceptron.neurons[k]; // Get neuron

                // Calculate neuron output and error
                n->output = heaviside(sum(&sample.X, n->weights, n->bias, n->n_weights));
                n->delta = (sample.c[k] - n->output); // sample.c[k] because each neurone computes 1 bit

                output[k] = n->output;

                // Update weights
                for (int l = 0; l < n->n_weights; l++) {
                    n->weights[l] = n->weights[l] + learning_rate * n->delta * sample.X;
                }
                // Update bias
                n->bias = n->bias + learning_rate * n->delta;

            }

            if (i % 100 == 0) {
                printf("Input: %.0f -> Target: [%.0f, %.0f, %.0f, %.0f] -> Predicted: [%.0f, %.0f, %.0f, %.0f]\n",
                    sample.X, sample.c[0], sample.c[1], sample.c[2], sample.c[3], output[0], output[1], output[2], output[3]);
            }
        }
        i++;
    }
    printf("Epochs: %d\n", i);

    print_layer(&perceptron);
    printf("--- Results ---\n");
    for (int i = 0; i < 16; i++) {
        binary sample = input[i];
        double output[4];
        // Calculate neuron output and error
        for (int j = 0; j < perceptron.n_neurons; j++) {
            neuron* n = &perceptron.neurons[j]; // Get neuron
            n->output = heaviside(sum(&sample.X, n->weights, n->bias, n->n_weights));
            output[j] = n->output;
        }
        printf("Input: %.0f -> Target: [%.0f, %.0f, %.0f, %.0f] -> Predicted: [%.0f, %.0f, %.0f, %.0f]\n",
            sample.X, sample.c[0], sample.c[1], sample.c[2], sample.c[3], output[0], output[1], output[2], output[3]);
    }
}
