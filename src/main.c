#include <stdio.h>
#include <stdlib.h>

#include "mlp.h"
#include "mnist.h"
#include "math_functions.h"

int main(int argc, char** argv) {
    srand(42); // Pour la reproductibilité

    idx3 x_train = read_images_mnist(getenv("IMAGES_TRAIN_PATH"));
    idx1 y_train = read_labels_mnist(getenv("LABELS_TRAIN_PATH"));

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

    save_model(&model, "/Users/remi/Documents/dev/c-mlp/model.weights");

    free_model(&model);
    free_mnist_images(&x_train);
    free_mnist_labels(&y_train);


    MLP model2;
    load_model(&model2, "/Users/remi/Documents/dev/c-mlp/model.weights");

    print_model(&model2);

    // Test using unseen data
    idx3 x_test = read_images_mnist(getenv("IMAGES_TEST_PATH"));
    idx1 y_test = read_labels_mnist(getenv("LABELS_TEST_PATH"));

    printf("\n--- Results ---\n");
    for (int i = 0; i < x_test.n_images; i++) {
        get_mnist_image_norm(image_buffer, &x_test, i);
        forward(&model2, image_buffer, 784);
        double* outputs = model2.layers[model2.n_layers - 1].outputs;
        if (i % 100 == 0)
            printf("Output: %d | Actual: %d\n", index_of_max(outputs, 10), y_test.labels[i]);
    }

    free_model(&model2);
    free_mnist_images(&x_test);
    free_mnist_labels(&y_test);

    return 0;
}
