#include <stdio.h>
#include <stdlib.h>
// #include <time.h>
#include "mlp.h"
#include "mnist.h"
#include "math_functions.h"
// #include "mnist.h"

int main(int argc, char** argv) {
    srand(42); // Pour la reproductibilité

    // double X[16][4] = {
    //     {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
    //     {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
    //     {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
    //     {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
    // };

    // double y[16][16];
    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         y[i][j] = 0.0;
    //     }
    //     y[i][i] = 1.0;
    // }


    idx3 x_train = read_images_mnist("/Users/remi/Documents/datasets/MNIST/train-images-idx3-ubyte");
    idx1 y_train = read_labels_mnist("/Users/remi/Documents/datasets/MNIST/train-labels-idx1-ubyte");

    // ======== MODEL ARCHITECTURE ========
    // function sig = {sigmoid, sigmoid_deriv};
    // function lin = {linear, linear_deriv};
    function rel = {leaky_relu, leaky_relu_deriv};
    function softm = {softmax, NULL};

    layer layers[4] = {
        dense(256, 784, &rel),
        dense(128, 256, &rel),
        dense(64, 128, &rel),
        dense(10, 64, &softm),
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

    double image_buffer[784];
    // int i_copy = 0;
    printf("\n --- Training model ---\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < x_train.n_images / 5; i++) {
            // int i = rand() % 16; // Random input
            get_mnist_image_norm(image_buffer, &x_train, i);
            forward(&model, image_buffer, 784);
            double* actual = one_hot(y_train.labels[i], 10);
            train(&model, image_buffer, actual, lr);

            // i_copy = i;
            free(actual);
        }
        if (epoch % (int)(0.1 * epochs) == 0) {
            printf("Epoch: %d...\n", epoch);
            // double* outputs = model.layers[model.n_layers - 1].outputs;
            // double* actual = one_hot(y_train.labels[i_copy], 10);
            // double loss = categ_cross_entropy(outputs, actual, 10);
            // printf("Epoch %d - Loss: %f\n", epoch, loss);
            // free(actual);
        }
    }
    printf("--- End ---\n");

    idx3 x_test = read_images_mnist("/Users/remi/Documents/datasets/MNIST/t10k-images-idx3-ubyte");
    idx1 y_test = read_labels_mnist("/Users/remi/Documents/datasets/MNIST/t10k-labels-idx1-ubyte");

    printf("\n--- Results ---\n");
    for (int i = 0; i < x_test.n_images / 5; i++) {
        get_mnist_image_norm(image_buffer, &x_test, i);
        forward(&model, image_buffer, 784);
        double* outputs = model.layers[model.n_layers - 1].outputs;
        if (i % 100 == 0)
            printf("Output: %d | Actual: %d\n", index_of_max(outputs, 10), y_test.labels[i]);
    }
    return 0;
}
