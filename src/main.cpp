#include <iostream>
#include <tuple>
#include <vector>
#include "slykitlearn.hpp"
#include "logger.hpp"

Logger logger = Logger(Logger::LogLevel::DEBUG, "%m/%d/%y %H:%M:%S");

int main(int argc, char** argv) {
    logger.log(Logger::LogLevel::INFO, "Info");

    std::vector<std::unique_ptr<Layer>> layers;

    layers.push_back(
        std::make_unique<DenseLayer<LeakyRelu>>(800, 784)
    );


    // Model model(layers);

    return 0;
}
