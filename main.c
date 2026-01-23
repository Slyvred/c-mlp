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
    // 2 hidden layers and one output layer
    layer hidden, hidden2, output;
    function sig = {sigmoid, sigmoid_deriv};
    init_layer(&hidden, 4, 1, &sig);
    init_layer(&hidden2, 16, hidden.n_neurons, &sig);
    init_layer(&output, 4, hidden2.n_neurons, &sig);

    layer* layers[3] = {&hidden, &hidden2, &output};
    MLP model = {layers, 3};
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
        double* outputs = model.layers[model.n_layers - 1]->outputs;
        printf("In: %2d | Out: [%0.f %.0f %.0f %.0f] | Expected: [%0.f %0.f %0.f %0.f]\n",
               i, outputs[0], outputs[1], outputs[2], outputs[3], y[i][0], y[i][1], y[i][2], y[i][3]);
    }
    return 0;
}
