#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mlp.h"

int main(int argc, char** argv) {
    srand(42); // Pour la reproductibilité

    // Decimal to binary converter
    double y[16][1];
    for(int i = 0; i < 16; i++) {
        y[i][0] = i / 15.0;
    }

    double X[16][4] = {
        {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
        {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
        {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
        {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
    };

    // ======== MODEL ARCHITECTURE ========
    // function sig = {sigmoid, sigmoid_deriv};
    function lin = {linear, linear_deriv};
    function rel = {leaky_relu, leaky_relu_deriv};

    layer layers[3] = {
        dense(4, 4, &rel), // 4 is our input shape,
        dense(16, 4, &rel),
        dense(1, 16, &lin), // 1 is our output shape
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
        for (int i = 0; i < 16; i++) {
            // int i = rand() % 16; // Random input
            forward(&model, X[i], 4);
            train(&model, X[i], y[i], lr);
            if (epoch % (int)(0.1 * epochs) == 0) printf("Epoch %d...\n", epoch);
        }
    }
    printf("--- End ---\n");

    printf("\n--- Results ---\n");
    for (int i = 0; i < 16; i++) {
        forward(&model, X[i], 4);
        double* outputs = model.layers[model.n_layers - 1].outputs;
        int n_outputs = model.layers[model.n_layers - 1].n_neurons;
        denormalize(outputs, n_outputs, 15);
        double y_denorm[1] = {i};
        print_output(&model, X[i], 4, y_denorm, 1);
    }
    return 0;
}
