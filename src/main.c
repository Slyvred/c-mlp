#include <stdio.h>
#include <stdlib.h>

#include "mlp.h"
#include "mnist.h"
#include "math_functions.h"

int main(int argc, char** argv) {
    srand(42); // Pour la reproductibilité

    idx3 x_train = read_images_mnist("/Users/remi/Documents/datasets/MNIST/train-images-idx3-ubyte");
    idx1 y_train = read_labels_mnist("/Users/remi/Documents/datasets/MNIST/train-labels-idx1-ubyte");

    // -------- MODEL ARCHITECTURE --------
    // function sig = {sigmoid, sigmoid_deriv};
    // function lin = {linear, linear_deriv};
    function rel = {leaky_relu, leaky_relu_deriv};
    function softm = {softmax, softmax_deriv};

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
    printf("\n --- Training model ---\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < x_train.n_images / 5; i++) {
            get_mnist_image_norm(image_buffer, &x_train, i);
            double* actual = one_hot(y_train.labels[i], 10);
            forward(&model, image_buffer, 784);
            train(&model, image_buffer, actual, lr);
            free(actual);
        }
        if (epoch % (int)(0.1 * epochs) == 0) printf("Epoch: %d...\n", epoch);
    }
    printf("--- End ---\n");

    // Test using unseen data
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
