#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE 28
#define MAX_BRIGHTNESS 255

int main() {
    FILE* fptr;
    fptr = fopen("/home/slyvred/Documents/MNIST_ORG/t10k-images.idx3-ubyte", "rb");

    if (fptr == NULL)
        printf("The file is not opened.\n");
    else
        printf("The file is created Successfully.\n");

    unsigned char buffer[IMAGE_SIZE * IMAGE_SIZE];
    fseek(fptr, 784, 1);
    fread(buffer, sizeof(buffer), 1, fptr);

    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            double normalized = (double)(buffer[i * j]) / (double)MAX_BRIGHTNESS;
            printf("%.1f", normalized);
        }
        printf("\n");
    }

    return 0;
}
