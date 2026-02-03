#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mlp.h"
#include "mnist.h"
#include "math_functions.h"

int main(int argc, char** argv) {

    srand(time(NULL)); // Pour la reproductibilité

    IDX3_t x_train = read_images_mnist(getenv("IMAGES_TRAIN_PATH"));
    IDX1_t y_train = read_labels_mnist(getenv("LABELS_TRAIN_PATH"));

    IDX3_t x_test = read_images_mnist(getenv("IMAGES_TEST_PATH"));
    IDX1_t y_test = read_labels_mnist(getenv("LABELS_TEST_PATH"));

    Vec2_t kernel_size = {4, 4};
    Vec2_t maxpool_kernel_size = {2, 2};
    Vec2_t input_size = {28, 28};
    Conv2DLayer_t conv_layer = conv_2d(4, kernel_size, input_size, &rel);
    PoolingLayer_t pooling_layer = max_pool_2d(4, conv_layer.output_size, maxpool_kernel_size);


    int input_shape = pooling_layer.n_inputs * pooling_layer.output_size.x * pooling_layer.output_size.y;
    Layer_t layers[3] = {
        dense(64, input_shape , &rel), // 784 is our input shape
        dense(128, 64, &rel),
        dense(10, 128, &softm), // 10 is our output shape (because we have 10 classes)
    };
    Model_t model = { layers, sizeof(layers) / sizeof(Layer_t), 0};

    CNN_t cnn = {
        1,
        1,
        0,
        &conv_layer,
        &pooling_layer,
        &model,
    };

    float img_buffer[784];
    get_mnist_image_norm(img_buffer, &x_train, 64);
    forward_cnn(&cnn, img_buffer, input_size);

    float* outputs = cnn.fully_connected->layers[cnn.fully_connected->n_layers - 1].outputs;
    int predicted = index_of_max(outputs, 10);
    printf("Output: %d | Actual: %d\n", predicted, y_train.labels[64]);


    // // Default values
    // int epochs = 6;
    // float lr = 0.01;

    // if (argc == 3) {
    //     epochs = atoi(argv[1]);
    //     lr = atof(argv[2]);
    // }

    // printf("\n --- Training model ---\n");
    // float image_buffer[784];
    // float one_hot_buffer[10];
    // float min_val_loss = 999;
    // float* train_losses = malloc(sizeof(float) * x_train.n_images);
    // float* val_losses = malloc(sizeof(float) * x_test.n_images);

    // // 1 epoch = 1 run through all the train dataset
    // for (int epoch = 0; epoch < epochs; epoch++) {

    //     // Train model
    //     for (int i = 0; i < x_train.n_images; i++) {
    //         // "Formatting inputs"
    //         get_mnist_image_norm(image_buffer, &x_train, i);
    //         one_hot(one_hot_buffer, y_train.labels[i], 10);

    //         // Actual training
    //         forward(&model, image_buffer, 784);
    //         train(&model, image_buffer, one_hot_buffer, lr);

    //         float* outputs = model.layers[model.n_layers - 1].outputs;
    //         train_losses[i] = categ_cross_entropy(outputs, one_hot_buffer, 10);
    //     }

    //     // Validate model
    //     for (int i = 0; i < x_test.n_images; i++) {
    //         // Actual inference
    //         get_mnist_image_norm(image_buffer, &x_test, i);
    //         forward(&model, image_buffer, 784);

    //         // Loss computing
    //         float* outputs = model.layers[model.n_layers - 1].outputs;
    //         one_hot(one_hot_buffer, y_test.labels[i], 10);
    //         val_losses[i] = categ_cross_entropy(outputs, one_hot_buffer, 10);
    //     }

    //     // Display average loss (average categorical cross entropy) for each epoch
    //     // We use the categorical cross entropy function because it's adapted
    //     // for multiclass classification with one hot encoded vectors
    //     float avg_train_loss = average(train_losses, x_train.n_images);
    //     float avg_val_loss = average(val_losses, x_test.n_images);
    //     printf("Epoch: %d - Loss: %.10f - Validation loss: %.10f\n", epoch+1, avg_train_loss, avg_val_loss);

    //     // Checkpointing: if the loss is lower than the previous loss we save the model
    //     if (avg_val_loss < min_val_loss) {
    //         printf("  Average loss is lower than last best, saving new best model...\n");
    //         printf("  ");
    //         save_model(&model, getenv("MODEL_PATH"));
    //         printf("\n");
    //         min_val_loss = avg_val_loss;
    //     }

    // }
    // printf("--- End ---\n");

    // free(train_losses);
    // free(val_losses);
    // free_model(&model);
    // free_mnist_images(&x_train);
    // free_mnist_labels(&y_train);

    // --- Inference example on already trained model ---

    // Model_t model2;
    // load_model(&model2, getenv("MODEL_PATH"));



    // float* test_losses = malloc(sizeof(float) * x_test.n_images);

    // printf("\n--- Results ---\n");
    // for (int i = 0; i < x_test.n_images; i++) {
    //     // Actual inference
    //     get_mnist_image_norm(image_buffer, &x_test, i);
    //     forward(&model2, image_buffer, 784);

    //     // Loss computing
    //     float* outputs = model2.layers[model2.n_layers - 1].outputs;
    //     one_hot(one_hot_buffer, y_test.labels[i], 10);
    //     test_losses[i] = categ_cross_entropy(outputs, one_hot_buffer, 10);

    //     if (i % 100 == 0) {
    //         int predicted = index_of_max(outputs, 10);
    //         printf("Output: %d | Actual: %d\n", predicted, y_test.labels[i]);
    //     }
    // }
    // float average_loss = average(test_losses, x_test.n_images);
    // printf("Average loss: %.10f\n", average_loss);

    // free(test_losses);
    // free_model(&model2);
    // free_mnist_images(&x_test);
    // free_mnist_labels(&y_test);
    return 0;
}
