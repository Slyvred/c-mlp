#include "slykitlearn.hpp"
#include "logger.hpp"
#include <iostream>
#include <vector>

Logger logger = Logger(Logger::LogLevel::DEBUG, "%m/%d/%y %H:%M:%S");

int main(int argc, char** argv) {
    logger.log(Logger::LogLevel::INFO, "Info");

    Model model;
    model.add_layer<LeakyRelu>(256, 784);
    model.add_layer<Softmax>(10, 256);

    std::vector<float> dummy_input;
    for (float i = 0; i < 784; i++)
        dummy_input.push_back(i / 255);

    model.forward(dummy_input);
    auto output = model.get_output();
    for (auto &out : output) {
        std::cout << out << " ";
    }
    std::cout << "\n";

    return 0;
}
