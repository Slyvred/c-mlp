#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <fstream>
#include "mnist.hpp"

IDX3::IDX3(const char* path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open file, check path or permissions.");
    }

    // We read all the header until the data part
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&n_images), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(uint32_t));

    // Swap the endians
    magic_number = __builtin_bswap32(magic_number);
    n_images = __builtin_bswap32(n_images);
    n_rows = __builtin_bswap32(n_rows);
    n_cols = __builtin_bswap32(n_cols);

    if (magic_number != 2051) {
        throw std::runtime_error("Wrong magic number: expected 2051\n");
    }

    // Read the images
    const size_t length = static_cast<size_t>(n_images) * n_rows * n_cols;
    images.resize(length);
    file.read(reinterpret_cast<char*>(images.data()), length);
    file.close();
}

void IDX3::get_image_norm(std::vector<float> &buf, int index) {
    auto start = images.begin() + index * (n_rows * n_cols);
    auto end = start + (n_rows * n_cols);
    std::transform(start, end, buf.begin(), [](uint8_t x) {return x / 255.f;});
}

uint32_t IDX3::get_n_cols() {
    return this->n_cols;
}

uint32_t IDX3::get_n_rows() {
    return this->n_rows;
}

uint32_t IDX3::get_n_images() {
    return this->n_images;
}

IDX1::IDX1(const char* path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open file, check path or permissions.\n");
    }

    // We read all the header until the data part
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&n_labels), sizeof(uint32_t));

    // Swap the endians
    magic_number = __builtin_bswap32(magic_number);
    n_labels = __builtin_bswap32(n_labels);

    if (magic_number != 2049) {
        throw std::runtime_error("Wrong magic number: expected 2049\n");
    }

    // Read the images
    const size_t length = static_cast<size_t>(n_labels);
    labels.resize(length);
    file.read(reinterpret_cast<char*>(labels.data()), length);
    file.close();
}

uint8_t IDX1::get_label(int index) {
    if (static_cast<uint32_t>(index) > this->n_labels - 1) {
        throw std::out_of_range("Index out of range");
    }

    return this->labels[index];
}

uint32_t IDX1::get_n_labels() {
    return this->n_labels;
}
