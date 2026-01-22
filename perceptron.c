#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "helpers.h"

typedef struct{
    double X[4];
    double c;
}binary;

binary input[16] = {
    {{0, 0, 0, 0}, 0},
    {{0, 0, 0, 1}, 1},
    {{0, 0, 1, 0}, 2},
    {{0, 0, 1, 1}, 3},
    {{0, 1, 0, 0}, 4},
    {{0, 1, 0, 1}, 5},
    {{0, 1, 1, 0}, 6},
    {{0, 1, 1, 1}, 7},
    {{1, 0, 0, 0}, 8},
    {{1, 0, 0, 1}, 9},
    {{1, 0, 1, 0}, 10},
    {{1, 0, 1, 1}, 11},
    {{1, 1, 0, 0}, 12},
    {{1, 1, 0, 1}, 13},
    {{1, 1, 1, 0}, 14},
    {{1, 1, 1, 1}, 15},
};

int main(int argc, char** argv) {
    srand(time(NULL));

    layer perceptron;
    init_layer(&perceptron, 1, 4);
    int epochs = 32;
    double learning_rate = 0.1;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        learning_rate = atof(argv[2]);
    }
    print_layer(&perceptron);
    printf("--- Training ---\n");
    int correct_guesses = 0;
    int i = 0;
    int sample_size = 16 * 0.7; // 11 samples out of 16 (split tain/test)
    while (correct_guesses < sample_size && i < epochs) { // 1 epoch = running the perceptron trough all training examples
        correct_guesses = 0; // Reset streak at the beginning of each epoch
        for (int j = 0; j < sample_size; j++) {
            binary sample = input[j];
            neuron* n = &perceptron.neurons[0]; // Get neuron

            // Calculate neuron output and error
            n->output = relu(sum(sample.X, n->weights, n->bias, n->n_weights));
            n->delta = (sample.c - n->output);

            // If we guessed correctly, move on to the next sample
            if (fabs(sample.c - n->output) < 1e-8) {
                correct_guesses += 1;
                continue;
            }

            // Update weights
            for (int k = 0; k < n->n_weights; k++) {
                n->weights[k] = n->weights[k] + learning_rate * n->delta * sample.X[k];
            }
            // Update bias
            n->bias = n->bias + learning_rate * n->delta;

            if (i % 100 == 0 && j == 0) {
                printf("Epoch: %d | Loss: %f\n", i, n->delta);
            }
        }
        i++;
    }

    printf("--- End ---\nEpochs: %d\nTrained Neuron:\n", i);
    print_layer(&perceptron);

    // Testing the perceptron
    printf("--- Results ---\n");
    for (int i = 0; i < 16; i++) { // 16 to test with all cases
        binary sample = input[i];
        neuron* n = &perceptron.neurons[0]; // Get neuron

        // Calculate neuron output and error
        n->output = relu(sum(sample.X, n->weights, n->bias, n->n_weights));
        printf("Input: [%.0f, %.0f, %0.f, %0.f] | Predicted: %0.f | Actual: %0.f\n", sample.X[0], sample.X[1],sample.X[2], sample.X[3], n->output, sample.c);
    }
}
