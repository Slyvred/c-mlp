#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helpers.h"

int main(int argc, char** argv) {
    srand(42); // Pour la reproductibilité


    // ======== DATASETS ========

    // XOR example
    // double X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    // double y[4][1] = {{0},   {1},   {1},   {0}};

    // Decimal to binary converter
    double X[16][1];
    for(int i = 0; i < 16; i++) {
        // Normalize inputs because our activation function is a sigmoid [0, 1]
        // If we don't do that it will saturate the outputs
        X[i][0] = i / 15.0;
    }

    double y[16][4] = {
        {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
        {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
        {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
        {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
    };

    // ======== MODEL ARCHITECTURE ========
    function sig = {sigmoid, sigmoid_deriv};
    function id = {linear, linear_deriv};

    layer layers[3] = {
        dense(4, 1, &sig), // 1 is our input shape
        dense(16, 4, &sig),
        dense(4, 16, &sig), // 4 is our output shape
    };
    MLP model = { layers, sizeof(layers) / sizeof(layer) };

    print_model(&model);

    // Default values
    int epochs = 70000;
    double lr = 0.5;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        lr = atof(argv[2]);
    }


    printf("\n --- Training model ---\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        int i = rand() % 16; // Random input
        forward(&model, X[i], 1);
        train(&model, X[i], y[i], lr);
        if (epoch % (int)(0.1 * epochs) == 0) printf("Epoch %d...\n", epoch);
    }
    printf("--- End ---\n");

    printf("\n--- Results ---\n");
    for (int i = 0; i < 16; i++) {
        forward(&model, X[i], 1);
        double input[1] = {i};
        print_output(&model, input, 1, y[i], 4);
    }
    return 0;
}
