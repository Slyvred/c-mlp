#include <iostream>
#include "slykitlearn.hpp"
#include "logger.hpp"

Logger logger = Logger(Logger::LogLevel::DEBUG, "%m/%d/%y %H:%M:%S");

int main(int argc, char** argv) {
    logger.setLevel(Logger::LogLevel::INFO);
    logger.log(Logger::LogLevel::DEBUG, "Caca");
    logger.log(Logger::LogLevel::INFO, "Info");
    return 0;
}
