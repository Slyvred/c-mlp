#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef struct{
    int X[2];
    int c;
}couple;

void init_weights(float *w0, float *w1, float *b) {
    *w0 = (float)rand() / RAND_MAX;
    *w1 = (float)rand() / RAND_MAX;
    *b = (float)rand() / RAND_MAX;
}

int heaviside(float val) {
    return val > 0 ? 1 : 0;
}

int main() {
    srand(time(NULL));

    // OR
    couple S[4] = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1}
    };

    // AND
    // couple S[4] = {
    //     {{0, 0}, 0},
    //     {{0, 1}, 0},
    //     {{1, 0}, 0},
    //     {{1, 1}, 1}
    // };

    float b;
    float w[2];
    float rate = 0.1;
    init_weights(&w[0], &w[1], &b);
    printf("Initial weights: w0: %f, w1: %f, b: %f\n", w[0], w[1], b);

    printf("\n--- Training ---\n");
    for (int i = 0; i < 64; i++) {
        couple sample = S[i % 4];
        float omega = sample.X[0] * w[0] + sample.X[1] * w[1] + b; // Output
        omega = omega > 0 ? 1 : 0; // Activation function
        float loss = (sample.c - omega);
        // Update weights
        for (int j = 0; j <= 1; j++) {
            w[j] = w[j] + rate * loss * sample.X[j];
        }
        b = b + rate * loss;
        // printf("Predicted: %f Actual: %d - Weights: w0: %f, w1: %f, b: %f - Loss: %f\n", omega, sample.c, w[0], w[1], b, loss);
        printf("Predicted: %f | Actual: %d | Loss: %f\n", omega, sample.c, loss);
    }
    printf("\n--- Trained Weights ---\n");
    printf(" - w0: %f\n - w1: %f\n - bias: %f\n", w[0], w[1], b);

    printf("\n--- Final Test ---\n");
    for (int i = 0; i < 4; i++) {
        float sum = S[i].X[0] * w[0] + S[i].X[1] * w[1] + b;
        int prediction = sum > 0 ? 1 : 0;
        printf("Input: %d, %d | Target: %d | Prediction: %d\n",
                S[i].X[0], S[i].X[1], S[i].c, prediction);
    }
}
