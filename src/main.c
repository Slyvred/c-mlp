#include <stdio.h>
#include <stdlib.h>
#include "mlp.h"
#include "iris.h"
#include "math_functions.h"

int main(int argc, char** argv) {
    srand(42); // Pour la reproductibilité

    Dataset_t dataset = read_iris(getenv("IRIS_DATASET"));

    layer layers[2] = {
        dense(32, 4, &rel), // 4 is our input shape
        dense(3, 32, &softm), // 3 is our output shape (because we have 3 species)
    };
    MLP model = { layers, sizeof(layers) / sizeof(layer) };

    print_model(&model);

    // Default values
    int epochs = 6;
    float lr = 0.01;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        lr = atof(argv[2]);
    }

    printf("\n --- Training model ---\n");
    float input_buffer[4];
    float one_hot_buffer[3];
    float min_val_loss = 999;
    float* train_losses = malloc(sizeof(float) * 105);
    float* val_losses = malloc(sizeof(float) * 45);

    // 1 epoch = 1 run through all the train dataset
    for (int epoch = 0; epoch < epochs; epoch++) {

        // Train model
        for (int i = 0; i < dataset.n_rows * 0.7; i++) {
            // "Formatting inputs"
            one_hot(one_hot_buffer, dataset.y[i], 3);

            input_buffer[0] = dataset.X[i].sepal_length;
            input_buffer[1] = dataset.X[i].sepal_width;
            input_buffer[2] = dataset.X[i].petal_length;
            input_buffer[3] = dataset.X[i].petal_width;

            // Actual training
            forward(&model, input_buffer, 4);
            train(&model, input_buffer, one_hot_buffer, lr);

            float* outputs = model.layers[model.n_layers - 1].outputs;
            train_losses[i] = categ_cross_entropy(outputs, one_hot_buffer, 3);
        }

        // Validate model
        for (int i = dataset.n_rows * 0.7; i < dataset.n_rows; i++) {
            // Actual inference
            input_buffer[0] = dataset.X[i].sepal_length;
            input_buffer[1] = dataset.X[i].sepal_width;
            input_buffer[2] = dataset.X[i].petal_length;
            input_buffer[3] = dataset.X[i].petal_width;

            forward(&model, input_buffer, 4);

            // Loss computing
            float* outputs = model.layers[model.n_layers - 1].outputs;
            one_hot(one_hot_buffer, dataset.y[i], 3);
            val_losses[i] = categ_cross_entropy(outputs, one_hot_buffer, 3);
        }

        // Display average loss (average categorical cross entropy) for each epoch
        // We use the categorical cross entropy function because it's adapted
        // for multiclass classification with one hot encoded vectors
        float avg_train_loss = average(train_losses, dataset.n_rows * 0.7);
        float avg_val_loss = average(val_losses, dataset.n_rows * 0.3);
        printf("Epoch: %d - Loss: %.10f - Validation loss: %.10f\n", epoch+1, avg_train_loss, avg_val_loss);

        // Checkpointing: if the loss is lower than the previous loss we save the model
        if (avg_val_loss < min_val_loss) {
            printf("  Average loss is lower than last best, saving new best model...\n");
            printf("  ");
            save_model(&model, getenv("MODEL_PATH"));
            printf("\n");
            min_val_loss = avg_val_loss;
        }

    }
    printf("--- End ---\n");

    // free(train_losses);
    // free(val_losses);
    free_model(&model);
    return 0;
}
