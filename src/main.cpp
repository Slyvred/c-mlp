#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "slykitlearn.hpp"
#include "mnist.hpp"
#include "logger.hpp"

Logger logger = Logger(Logger::LogLevel::INFO, "%m/%d/%y %H:%M:%S");

IDX3 x_train("/Users/remi/Documents/datasets/MNIST/train-images-idx3-ubyte");
IDX1 y_train("/Users/remi/Documents/datasets/MNIST/train-labels-idx1-ubyte");

IDX3 x_test("/Users/remi/Documents/datasets/MNIST/t10k-images-idx3-ubyte");
IDX1 y_test("/Users/remi/Documents/datasets/MNIST/t10k-labels-idx1-ubyte");


int index_of_max(const std::vector<float> &array) {
    int max = 0;
    for (size_t i = 0; i < array.size(); i++) {
        if (array[i] > array[max]) max = i;
    }
    return max;
}

float categ_cross_entropy(const std::vector<float> &predicted, const std::vector<float> &actual, int n_classes) {
    int correct_class_idx = index_of_max(actual);
    return -log(predicted[correct_class_idx]);
}

int main(int argc, char** argv) {

    int epochs = 6;
    float lr = 0.01;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        lr = atof(argv[2]);
    }

    Model model;
    model.add_layer<LeakyRelu>(800, 784);
    model.add_layer<Softmax>(10, 800);

    std::vector<float> x_buf(784);
    std::vector<float> y_buf(10);
    std::vector<float> train_losses(x_train.get_n_images());
    std::vector<float> test_losses(x_test.get_n_images());

    std::cout << "Training...\n";
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < static_cast<int>(x_train.get_n_images()); j++) {
            x_train.get_image_norm(x_buf, j);
            model.forward(x_buf);

            y_train.get_label_one_hot<float>(y_buf, j, 10);
            model.train(y_buf, lr);
            train_losses[j] = categ_cross_entropy(model.get_output(), y_buf, 10);
        }

        for (int j = 0; j < static_cast<int>(x_test.get_n_images()); j++) {
            x_test.get_image_norm(x_buf, j);
            model.forward(x_buf);

            y_test.get_label_one_hot<float>(y_buf, j, 10);
            test_losses[j] = categ_cross_entropy(model.get_output(), y_buf, 10);
        }

        float avg_loss_train = std::accumulate(train_losses.begin(), train_losses.end(), 0.f) / train_losses.size();
        float avg_loss_test = std::accumulate(test_losses.begin(), test_losses.end(), 0.f) / test_losses.size();
        logger.log(Logger::LogLevel::INFO, "Training loss: %g - Testing loss: %g", avg_loss_train, avg_loss_test);
    }

    std::cout << "Inference test:\n";
    for (int j = 0; j < static_cast<int>(x_test.get_n_images()); j++) {
        x_test.get_image_norm(x_buf, j);
        model.forward(x_buf);

        if (j % 500 == 0)
            logger.log(Logger::LogLevel::INFO, "Actual: %d - Predicted: %d", static_cast<unsigned int>(y_test.get_label(j)), index_of_max(model.get_output()));
    }

    return 0;
}
