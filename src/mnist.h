#pragma once
#include <stdint.h>

typedef struct {
    uint32_t magic_number;
    uint32_t n_images;
    uint32_t n_rows;
    uint32_t n_cols;
    uint8_t* images;
}idx3;

typedef struct {
    uint32_t magic_number;
    uint32_t n_labels;
    uint8_t* labels;
}idx1;

idx3 read_images_mnist(const char* path);
idx1 read_labels_mnist(const char* path);
void get_mnist_image_norm(float* output, idx3* dataset, int index);
void free_mnist_images(idx3* dataset);
void free_mnist_labels(idx1* dataset);
