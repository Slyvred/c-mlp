#pragma once

// Activations functions
void sigmoid(float* inputs, float* outputs, int len);
void sigmoid_deriv(float* inputs, float* outputs, int len);
void linear(float* inputs, float* outputs, int len);
void linear_deriv(float* inputs, float* outputs, int len);
void leaky_relu(float* inputs, float* outputs, int len);
void leaky_relu_deriv(float* inputs, float* outputs, int len);
void softmax(float* inputs, float* outputs, int len);
void softmax_deriv(float* inputs, float* outputs, int len);

// Loss functions
float categ_cross_entropy(float* predicted, float* actual, int n_classes);
float mse(float* predicted, float* actual, int length);

// Other helpful functions
void normalize(float* values, int length, float max);
void denormalize(float* values, int length, float max);
float sum(float inputs[], float weights[], float bias, int len);
int index_of_max(float* array, int len);
float average(float* values, int length);
