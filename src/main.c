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
    int i_copy = 0;
    double last_loss = 999;
    // 1 epoch = 1 run through all the train dataset
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < x_train.n_images; i++) {
            // "Formatting inputs"
            get_mnist_image_norm(image_buffer, &x_train, i);
            double* one_hot_y = one_hot(y_train.labels[i], 10);

            // Actual training
            forward(&model, image_buffer, 784);
            train(&model, image_buffer, one_hot_y, lr);

            free(one_hot_y);
            i_copy = i;
        }
        // Display loss for each epoch
        double* outputs = model.layers[model.n_layers - 1].outputs;
        double* one_hot_y = one_hot(y_train.labels[i_copy], 10);

        // We use the categorical cross entropy function because it's adapted
        // for multiclass classification with one hot encoded vectors
        double loss = categ_cross_entropy(outputs, one_hot_y, 10);
        printf("Epoch: %d - Loss: %.10f\n", epoch+1, loss);
        free(one_hot_y);

        // Checkpointing: if the loss is lower than the previous loss we save the model
        if (loss < last_loss) {
            printf("  Loss is lower than last loss, saving new best model...\n");
            printf("  ");
            save_model(&model, getenv("MODEL_PATH"));
            printf("\n");
        }

        last_loss = loss;
    }
    printf("--- End ---\n");

    free_model(&model);
    free_mnist_images(&x_train);
    free_mnist_labels(&y_train);


    MLP model2;
    load_model(&model2, getenv("MODEL_PATH"));

    print_model(&model2);

    // Test using unseen data
    idx3 x_test = read_images_mnist(getenv("IMAGES_TEST_PATH"));
    idx1 y_test = read_labels_mnist(getenv("LABELS_TEST_PATH"));

    printf("\n--- Results ---\n");
    for (int i = 0; i < x_test.n_images; i++) {
        get_mnist_image_norm(image_buffer, &x_test, i);
        forward(&model2, image_buffer, 784);
        double* outputs = model2.layers[model2.n_layers - 1].outputs;
        if (i % 100 == 0) {
            double* one_hot_y = one_hot(y_test.labels[i], 10);
            int predicted = index_of_max(outputs, 10);

            // If we correctly predict the loss is 0, no need to compute it
            double loss = (predicted == y_test.labels[i]) ? 0 : categ_cross_entropy(outputs, one_hot_y, 10);

            printf("Output: %d | Actual: %d | Loss: %.10f\n", predicted, y_test.labels[i], loss);
            free(one_hot_y);
        }
    }

    free_model(&model2);
    free_mnist_images(&x_test);
    free_mnist_labels(&y_test);
    return 0;
}
