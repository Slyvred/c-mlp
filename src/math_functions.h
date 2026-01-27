#pragma once

// Activations functions
double sigmoid(double x);
double sigmoid_deriv(double x);
void linear(double* inputs, double* outputs, int len);
void linear_deriv(double* inputs, double* outputs, int len);
void leaky_relu(double* inputs, double* outputs, int len);
void leaky_relu_deriv(double* inputs, double* outputs, int len);
void softmax(double* inputs, double* outputs, int len);

// Loss functions
double categ_cross_entropy(double* predicted, double* actual, int n_classes);
double mse(double* predicted, double* actual, int length);

// Other helpful functions
void normalize(double* values, int length, double max);
void denormalize(double* values, int length, double max);
double sum(double inputs[], double weights[], double bias, int len);
int index_of_max(double* array, int len);
