#include <math.h>
#include "math_functions.h"

#define LEAKY_RELU_SLOPE 0.01

void sigmoid(double* inputs, double* outputs, int len) {
    for (int i = 0; i < len; i++) {
        outputs[i] = 1.0 / (1.0 + exp(-inputs[i]));
    }
}

void sigmoid_deriv(double* inputs, double* outputs, int len) {
    for (int i = 0; i < len; i++) {
        outputs[i] = inputs[i] * (1.0 - inputs[i]);
    }
}

void linear(double* inputs, double* outputs, int len) {
    for (int i = 0; i < len; i++) {
        outputs[i] = inputs[i];
    }
}

void linear_deriv(double* inputs, double* outputs, int len) {
    for (int i = 0; i < len; i++) {
        outputs[i] = 1.0;
    }
}

void leaky_relu(double* inputs, double* outputs, int len) {
    for (int i = 0; i < len; i++) {
        outputs[i] = inputs[i] >= 0 ? inputs[i] : LEAKY_RELU_SLOPE * inputs[i];
    }
}

void leaky_relu_deriv(double* inputs, double* outputs, int len) {
    for (int i = 0; i < len; i++) {
        outputs[i] = inputs[i] >= 0 ? 1 : LEAKY_RELU_SLOPE;
    }
}

void softmax(double* inputs, double* outputs, int len) {
    double max_val = inputs[0];
    for (int i = 1; i < len; i++) {
        if (inputs[i] > max_val) max_val = inputs[i];
    }

    double denom = 0;
    for (int i = 0; i < len; i++) {
        outputs[i] = exp(inputs[i] - max_val);
        denom += outputs[i];
    }

    if (denom < 1e-20) denom = 1e-20;

    for (int i = 0; i < len; i++) {
        outputs[i] /= denom;
    }
}

// This function does NOT compute the true derivative of softmax.
//
// When using Softmax + Cross-Entropy loss, the gradient w.r.t the logits is:
//     dL/dz = output - target
// The derivative of softmax is already analytically absorbed into the loss.
// In this framework, the training code computes:
//     delta = (output - target) * derivative
// So for softmax, we must set derivative = 1 to avoid applying any extra factor.
// This function therefore acts as a neutral "pass-through" derivative.
// It exists only to keep the same training code path for both regression and classification.
void softmax_deriv(double* inputs, double* outputs, int len) {
    for (int i = 0; i < len; i++) {
        outputs[i] = 1;
    }
}

double sum(double inputs[], double weights[], double bias, int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    return sum;
}

void normalize(double* values, int length, double max) {
    for (int i = 0; i < length; i++) {
        values[i] /= max;
    }
}

void denormalize(double* values, int length, double max) {
    for (int i = 0; i < length; i++) {
        values[i] *= max;
    }
}

int index_of_max(double* array, int len) {
    int max = 0;
    for (int i = 0; i < len; i++) {
        if (array[i] > array[max]) max = i;
    }
    return max;
}

double mse(double* predicted, double* actual, int length) {
    double mse = 0;
    for (int i = 0; i < length; i++) {
        mse += (actual[i] - predicted[i])*(actual[i] - predicted[i]);
    }
    mse /= (double)length;
    return mse;
}

double categ_cross_entropy(double* predicted, double* actual, int n_classes) {
    int correct_class_idx = index_of_max(actual, n_classes);
    return -log(predicted[correct_class_idx]);
}
