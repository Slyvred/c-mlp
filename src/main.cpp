#include <iostream>
#include <vector>
#include "slykitlearn.hpp"
#include "logger.hpp"

Logger logger = Logger(Logger::LogLevel::DEBUG, "%m/%d/%y %H:%M:%S");

int main(int argc, char** argv) {
    logger.log(Logger::LogLevel::INFO, "Info");

    DenseLayer<float, LeakyRelu<float>> input(800, 784);
    std::vector<float> vec;

    return 0;
}
