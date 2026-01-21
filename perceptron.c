#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "helpers.h"

typedef struct{
    double X[2];
    double c;
}couple;

int main(int argc, char** argv) {
    srand(time(NULL));

    // OR
    couple OR[4] = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1}
    };

    // AND
    couple AND[4] = {
        {{0, 0}, 0},
        {{0, 1}, 0},
        {{1, 0}, 0},
        {{1, 1}, 1}
    };

    layer perceptron;
    init_layer(&perceptron, 1, 2);
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
    while (correct_guesses < 4 && i < epochs) { // 1 epoch = running the perceptron trough all examples
        correct_guesses = 0; // Reset streak at the beginning of each epoch
        for (int j = 0; j < 4; j++) {
            couple sample = AND[j];
            neuron* n = &perceptron.neurons[0]; // Get neuron

            // Calculate neuron output and error
            n->output = (double)heaviside(sum(sample.X, n->weights, n->bias, n->n_weights));
            n->delta = (sample.c - n->output);

            // If we guessed correctly, move on to the next sample
            if (n->output == sample.c) {
                correct_guesses += 1;
                continue;
            }

            // Update weights
            for (int k = 0; k < n->n_weights; k++) {
                n->weights[k] = n->weights[k] + learning_rate * n->delta * sample.X[k];
            }
            // Update bias
            n->bias = n->bias + learning_rate * n->delta;

            if (i % 10 == 0 && j == 0) {
                printf("Epoch: %d, Predicted: %f | Actual: %f | Loss: %f\n", i, n->output, sample.c, n->delta);
            }
        }
        i++;
    }
    printf("Epochs: %d\n", i);

    print_layer(&perceptron);
    printf("--- Results ---\n");
    for (int i = 0; i < 4; i++) {
        couple sample = AND[i];
        neuron* n = &perceptron.neurons[0]; // Get neuron

        // Calculate neuron output and error
        n->output = heaviside(sum(sample.X, n->weights, n->bias, n->n_weights));
        printf("Input: [%.0f, %.0f] -> Target: %.0f -> Predicted: %f\n", sample.X[0], sample.X[1], sample.c, n->output);
    }
}
