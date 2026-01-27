#include "mnist.h"
#include "mlp.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <arpa/inet.h> // Endian swapping
#include <string.h>

idx3 read_images_mnist(const char* path) {
    FILE* file = fopen(path, "rb");
    idx3 dataset = { 0, 0, 0, 0, NULL };

    if (file == NULL) {
        printf("Failed to open file, check path or permissions.\n");
        return dataset;
    }

    // We read all the header until the data part
    fread(&dataset, sizeof(dataset) - sizeof(dataset.images), 1, file);

    // Swap the endians
    dataset.magic_number = ntohl(dataset.magic_number);
    dataset.n_images = ntohl(dataset.n_images);
    dataset.n_rows = ntohl(dataset.n_rows);
    dataset.n_cols = ntohl(dataset.n_cols);

    if (dataset.magic_number != 2051) {
        printf("Wrong magic number: got %d expected 2051\n", dataset.magic_number);
        return dataset;
    }

    // Read the images
    int length = dataset.n_images * dataset.n_cols * dataset.n_rows;
    dataset.images = malloc(length * sizeof(uint8_t));
    fread(dataset.images, length * sizeof(uint8_t), 1, file);

    fclose(file);

    return dataset;
}

idx1 read_labels_mnist(const char* path) {
    FILE* file = fopen(path, "rb");
    idx1 dataset = {0, 0, NULL};

    if (file == NULL) {
        printf("Failed to open file, check path or permissions.\n");
        return dataset;
    }

    // We read all the header until the data part
    fread(&dataset, sizeof(dataset) - sizeof(dataset.labels), 1, file);

    // Swap the endians
    dataset.magic_number = ntohl(dataset.magic_number);
    dataset.n_labels = ntohl(dataset.n_labels);

    if (dataset.magic_number != 2049) {
        printf("Wrong magic number: got %d expected 2049\n", dataset.magic_number);
        return dataset;
    }

    // Read the images
    int length = dataset.n_labels * sizeof(uint8_t);
    dataset.labels = malloc(length * sizeof(uint8_t));
    fread(dataset.labels, length * sizeof(uint8_t), 1, file);

    fclose(file);

    return dataset;
}

void get_mnist_image_norm(double* output, idx3* dataset, int index) {
    for (int i = 0; i < 784; i++) {
        output[i] = (double)dataset->images[i+784*index];
    }
    normalize(output, 784, 255.0);
}
