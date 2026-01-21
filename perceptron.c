#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "helpers.h"

typedef struct{
    double X[2];
    double c;
}couple;

int main() {
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
    int epochs = 256;
    double learning_rate = 0.1;
    for (int i = 0; i < epochs; i++) { // 1 epoch = running the perceptron trough all examples
        for (int j = 0; j < 4; j++) {
            couple sample = OR[j];
            neuron* n = &perceptron.neurons[0]; // Get neuron
            n->output = sigmoid(sum(sample.X, n->weights, n->bias, n->n_weights));

        }
    }
}
