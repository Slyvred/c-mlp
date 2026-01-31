#pragma once
#include <stdint.h>

typedef struct {
    uint32_t magic_number;
    uint32_t n_images;
    uint32_t n_rows;
    uint32_t n_cols;
    uint8_t* images;
}IDX3_t;

typedef struct {
    uint32_t magic_number;
    uint32_t n_labels;
    uint8_t* labels;
}IDX1_t;

IDX3_t read_images_mnist(const char* path);
IDX1_t read_labels_mnist(const char* path);
void get_mnist_image_norm(float* output, IDX3_t* dataset, int index);
void free_mnist_images(IDX3_t* dataset);
void free_mnist_labels(IDX1_t* dataset);
