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
        dense(256, 784, &rel), // 784 is our input shape
        dense(128, 256, &rel),
        dense(64, 128, &rel),
        dense(10, 64, &softm), // 10 is our output shape (because we have 10 classes)
    };
    MLP model = { layers, sizeof(layers) / sizeof(layer) };

    print_model(&model);

    // Default values
    int epochs = 6;
    double lr = 0.01;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        lr = atof(argv[2]);
    }

    printf("\n --- Training model ---\n");
    double image_buffer[784];
    double one_hot_buffer[10];
    double last_loss = 999;
    double* losses = malloc(sizeof(double) * x_train.n_images);
    // 1 epoch = 1 run through all the train dataset
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < x_train.n_images; i++) {
            // "Formatting inputs"
            get_mnist_image_norm(image_buffer, &x_train, i);
            one_hot(one_hot_buffer, y_train.labels[i], 10);

            // Actual training
            forward(&model, image_buffer, 784);
            train(&model, image_buffer, one_hot_buffer, lr);

            double* outputs = model.layers[model.n_layers - 1].outputs;
            losses[i] = categ_cross_entropy(outputs, one_hot_buffer, 10);
        }
        // Display average loss for each epoch
        double avg_loss = average(losses, x_train.n_images);
        // We use the categorical cross entropy function because it's adapted
        // for multiclass classification with one hot encoded vectors
        printf("Epoch: %d - Loss: %.10f\n", epoch+1, avg_loss);

        // Checkpointing: if the loss is lower than the previous loss we save the model
        if (avg_loss < last_loss) {
            printf("  Average loss is lower than last epoch, saving new best model...\n");
            printf("  ");
            save_model(&model, getenv("MODEL_PATH"));
            printf("\n");
        }

        last_loss = avg_loss;
    }
    printf("--- End ---\n");

    free(losses);
    free_model(&model);
    free_mnist_images(&x_train);
    free_mnist_labels(&y_train);

    // --- Inference example ---
    MLP model2;
    load_model(&model2, getenv("MODEL_PATH"));

    // Test using unseen data
    idx3 x_test = read_images_mnist(getenv("IMAGES_TEST_PATH"));
    idx1 y_test = read_labels_mnist(getenv("LABELS_TEST_PATH"));

    double* test_losses = malloc(sizeof(double) * x_test.n_images);

    printf("\n--- Results ---\n");
    for (int i = 0; i < x_test.n_images; i++) {
        // Actual inference
        get_mnist_image_norm(image_buffer, &x_test, i);
        forward(&model2, image_buffer, 784);

        // Loss computing
        double* outputs = model2.layers[model2.n_layers - 1].outputs;
        one_hot(one_hot_buffer, y_test.labels[i], 10);
        test_losses[i] = categ_cross_entropy(outputs, one_hot_buffer, 10);

        if (i % 100 == 0) {
            int predicted = index_of_max(outputs, 10);
            printf("Output: %d | Actual: %d\n", predicted, y_test.labels[i]);
        }
    }
    printf("Average loss: %.10f\n", average(test_losses, x_test.n_images));

    free(test_losses);
    free_model(&model2);
    free_mnist_images(&x_test);
    free_mnist_labels(&y_test);
    return 0;
}
