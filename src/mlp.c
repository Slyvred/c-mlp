#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mlp.h"
#include "math_functions.h"

// "Global" activation functions
function sig = {sigmoid, sigmoid_deriv, SIGMOID};
function lin = {linear, linear_deriv, LINEAR};
function rel = {leaky_relu, leaky_relu_deriv, RELU};
function softm = {softmax, softmax_deriv, SOFTMAX};

double ranged_rand(double min, double max) {
    return ((double)rand() / RAND_MAX) * (max - min) + min;
}

layer dense(int n_neurons, int n_inputs, function *activation_function) {
    layer l;
    l.type = DENSE;
    l.n_inputs = n_inputs;
    l.n_outputs = n_neurons;
    l.biases = calloc(n_neurons, sizeof(double));                   // Biases set to 0
    l.weights = malloc(n_neurons * n_inputs * sizeof(double));      // n_neurons with n_inputs per neuron
    l.outputs = malloc(n_neurons * sizeof(double));
    l.raw_outputs = malloc(n_neurons * sizeof(double));
    l.derivatives = malloc(n_neurons * sizeof(double));
    l.deltas = malloc(n_neurons * sizeof(double));

    l.activation_function = activation_function;
    // Initialize weights
    double limit = sqrt(2.0 / n_inputs);
    for (int i = 0; i < n_neurons * n_inputs; i++) {
        l.weights[i] = ranged_rand(-limit, limit);
    }
    return l;
}

void forward(MLP *m, double* inputs, int n_inputs) {
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];
        double* layer_inputs;

        // For input layer, input is the actual input
        if (i == 0) layer_inputs = inputs;
        else layer_inputs = m->layers[i - 1].outputs;

        // For each neuron in the layer
        for (int j = 0; j < l->n_outputs; j++) {
            // Output of neuron_i is the sum of the previous layer's outputs
            double* weights = &l->weights[j * l->n_inputs];
            l->raw_outputs[j] = sum(layer_inputs, weights, l->biases[j], l->n_inputs);
        }
        l->activation_function->f(l->raw_outputs, l->outputs, l->n_outputs);
    }
}

// Entraînement (Backpropagation)
void train(MLP *m, double* raw_inputs, double* target, double lr) {
    // Calculate output layer error
    layer* l = &m->layers[m->n_layers - 1];
    l->activation_function->df(l->raw_outputs, l->derivatives, l->n_outputs);

    for (int i = 0; i < l->n_outputs; i++) {
        l->deltas[i] = (l->outputs[i] - target[i]) * l->derivatives[i];
    }

    // Calculate layer error from last hidden layer to input
    for (int i = m->n_layers - 2; i >= 0; i--) {
        layer* curr_layer = &m->layers[i];
        layer* next_layer = &m->layers[i + 1];

        curr_layer->activation_function->df(curr_layer->raw_outputs, curr_layer->derivatives, curr_layer->n_outputs);

        for (int j = 0; j < curr_layer->n_outputs; j++) {
            double error = 0;
            // Sum deltas of next layer weighted by next neuron weights
            for (int k = 0; k < next_layer->n_outputs; k++) {
                // next_layer->weights[j] is the weight connecting the current neuron with the next neuron k
                int weight_index = k * next_layer->n_inputs + j;
                error += next_layer->deltas[k] * next_layer->weights[weight_index];
            }
            curr_layer->deltas[j] = error * curr_layer->derivatives[j];
        }
    }

    // Update weights for each layer
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];
        double* layer_inputs;

        // For input layer, input is the actual input
        if (i == 0) layer_inputs = raw_inputs;
        else layer_inputs = m->layers[i - 1].outputs;

        // For each neuron
        for (int j = 0; j < l->n_outputs; j++) {
            // l->deltas[i] = (l->outputs[i] - target[i]) * l->derivatives[i];
            // For each weight
            for (int k = 0; k < l->n_inputs; k++) {
                int index = k * l->n_inputs + j;
                l->weights[index] -= lr * l->deltas[k] * layer_inputs[k];
            }
            l->biases[j] -= lr * l->deltas[j];
        }
    }
}

int get_num_parameters(MLP* mlp) {
    int parameters = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        layer* l = &mlp->layers[i];
        parameters += l->n_outputs * (l->n_inputs + 1); // + 1 for bias
    }
    return parameters;
}

void print_model(MLP* m) {
    printf("\n");
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];
        printf("Layer %d: Neurons: %d | Parameters: %d\n", i, l->n_outputs, l->n_inputs);
    }
    printf("Total number of parameters: %d\n", get_num_parameters(m));
}

void print_list(double* list, int len) {
    printf("[");
    for (int i = 0; i < len; i++) {
        if (i+1 >= len) printf("%.2f", list[i]);
        else printf("%.2f ", list[i]);
    }
    printf("]");
}

void print_output(MLP *m, double* input, int input_len, double *expected, int expected_len) {
    layer* output = &m->layers[m->n_layers - 1];
    printf("Inputs: ");
    print_list(input, input_len);

    printf(" | Outputs: ");
    print_list(output->outputs, output->n_outputs);
    printf(" | Expected: ");
    print_list(expected, expected_len);
    printf("\n");
}

void one_hot(double* output, int input, int n_classes) {
    for (int i = 0; i < n_classes; i++) output[i] = 0; // Set/Reset buffer
    output[input] = 1.0;
}

void free_model(MLP* m) {
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];

        free(l->biases);
        free(l->weights);
        free(l->outputs);
        free(l->raw_outputs);

        // It's null if we loaded a pre trained for inference since the derivatives are just used during training
        // for backprop. If you wish to re-train/finetune a pre train model, uncomment the malloc in the load_model function below
        if (l->deltas != NULL) free(l->deltas);
        if (l->derivatives != NULL) free(l->derivatives);
    }
}

void save_model(MLP* m, const char* path) {
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        printf("Failed to open model file\n");
        return;
    }

    // Write number of layers
    fwrite(&m->n_layers, sizeof(int), 1, f);

    // For each layer
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];

        // Write number of neurons
        fwrite(&l->type, sizeof(int), 1, f);
        fwrite(&l->n_outputs, sizeof(int), 1, f);
        // Write number of weights
        fwrite(&l->n_inputs, sizeof(int), 1, f);
        fwrite(l->weights, sizeof(double), l->n_inputs, f);
        fwrite(l->biases, sizeof(double), l->n_outputs, f);
        fwrite(&l->activation_function->function_name, sizeof(int), 1, f);
    }
    fclose(f);
    printf("Model saved to: %s\n", path);
}

void load_model(MLP* m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        printf("Failed to open model file\n");
        return;
    }

    // Read number of layers
    fread(&m->n_layers, sizeof(int), 1, f);

    // For each layer
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];

        // Read number of neurons
        fread(&l->type, sizeof(int), 1, f);
        fread(&l->n_outputs, sizeof(int), 1, f);
        // Write number of weights
        fread(&l->n_inputs, sizeof(int), 1, f);

        l->weights = malloc(l->n_inputs * l->n_outputs * sizeof(double));
        l->biases = malloc(l->n_outputs * sizeof(double));
        l->outputs = malloc(l->n_outputs * sizeof(double));
        l->raw_outputs = malloc(l->n_outputs * sizeof(double));
        l->derivatives = NULL;
        l->deltas = NULL;

        fread(l->weights, sizeof(double), l->n_inputs * l->n_outputs, f);
        fread(l->biases, sizeof(double), l->n_outputs, f);

        int fn_name_buf;
        fread(&fn_name_buf, sizeof(int), 1, f);

        switch (fn_name_buf) {
            case RELU:
                l->activation_function = &rel;
                break;
            case LINEAR:
                l->activation_function = &lin;
                break;
            case SIGMOID:
                l->activation_function = &sig;
                break;
            case SOFTMAX:
                l->activation_function = &softm;
                break;
            default:
                printf("ERROR: Unknown activation function, got: %d which isn't in the fn_name enum.\n", fn_name_buf);
                exit(EXIT_FAILURE);
                break;
        }
    }
    fclose(f);
    printf("Successfully loaded model !\n");
    print_model(m);
}
