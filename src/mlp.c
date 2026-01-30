#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
    l.activation_function = activation_function;
    l.biases = calloc(n_neurons, sizeof(double));                   // Biases set to 0
    l.weights = malloc(n_neurons * n_inputs * sizeof(double));      // n_neurons with n_inputs per neuron
    l.outputs = malloc(n_neurons * sizeof(double));
    l.raw_outputs = malloc(n_neurons * sizeof(double));
    l.derivatives = malloc(n_neurons * sizeof(double));
    l.deltas = malloc(n_neurons * sizeof(double));

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
        int offset = 0;
        for (int i = 0; i < l->n_outputs; i++) {
            // Output of neuron_i is the sum of the previous layer's outputs
            l->raw_outputs[i] = sum(layer_inputs, &l->weights[offset], l->biases[offset], l->n_inputs);

            // We jump to the next neuron in the layer (aka next "region" with the weights)
            offset += l->n_inputs;
        }
        l->activation_function->f(l->raw_outputs, l->outputs, l->n_outputs);
    }
}

// Entraînement (Backpropagation)
void train(MLP *m, double* raw_inputs, double* target, double lr) {
    // Calculate output layer error
    layer* l = &m->layers[m->n_layers - 1];
    l->activation_function->df(l->raw_outputs, l->derivatives, l->n_outputs);

    int offset = 0;
    for (int i = 0; i < l->n_outputs; i++) {
        l->deltas[i] = (l->outputs[i] - target[i]) * l->derivatives[i];
        offset += l->n_inputs;
    }

    // Calculate layer error from last hidden layer to input
    for (int i = m->n_layers - 2; i >= 0; i--) {
        layer* curr_layer = &m->layers[i];
        layer* next_layer = &m->layers[i + 1];

        curr_layer->activation_function->df(curr_layer->raw_outputs, curr_layer->derivatives, curr_layer->n_outputs);

        int offset = 0;
        for (int j = 0; j < curr_layer->n_outputs; j++) {
            double error = 0;

            // Sum deltas of next layer weighted by next neuron weights
            for (int k = 0; k < next_layer->n_outputs; k++) {
                // next_layer->weights[j] is the weight connecting the current neuron with the next neuron k
                error += next_layer->deltas[k] * next_layer->weights[j];
            }
            offset += curr_layer->n_inputs;
            curr_layer->deltas[j] = error * curr_layer->derivatives[j];
        }

        // for (int j = 0; j < curr_layer->n_neurons; j++) {
        //     neuron* n = &curr_layer->neurons[j];
        //     double error = 0;

        //     // Sum deltas of next layer weighted by next neuron weights
        //     for (int k = 0; k < next_layer->n_neurons; k++) {
        //         neuron* next_n = &next_layer->neurons[k];
        //         // next_n->weights[j] is the weight connecting the current neuron with the next neuron k
        //         error += next_n->delta * next_n->weights[j];
        //     }
        //     // n->delta = error * curr_layer->activation_function->df(n->output);
        //     n->delta = error * curr_layer->derivatives[j];
        // }
    }

    // Update weights for each layer
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];
        double* inputs_for_layer;

        // For input layer, input is the actual input
        if (i == 0) inputs_for_layer = raw_inputs;
        else inputs_for_layer = m->layers[i - 1].outputs;

        // For each neuron
        for (int j = 0; j < l->n_neurons; j++) {
            neuron* n = &l->neurons[j];

            // Update weights
            for (int k = 0; k < n->n_weights; k++) {
                // weight -= learning_rate * delta * input
                n->weights[k] -= lr * n->delta * inputs_for_layer[k];
            }

            // Update bias
            n->bias -= lr * n->delta;
        }
    }
}

int get_num_parameters(MLP* mlp) {
    int parameters = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        layer* l = &mlp->layers[i];
        parameters += l->n_neurons * (l->neurons[0].n_weights + 1); // + 1 for bias
    }
    return parameters;
}

void print_model(MLP* m) {
    printf("\n");
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];
        printf("Layer %d: Neurons: %d | Parameters: %d\n", i, l->n_neurons, l->neurons[0].n_weights);
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
    print_list(output->outputs, output->n_neurons);
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

        for (int j = 0; j < l->n_neurons; j++) {
            neuron* n = &l->neurons[j];
            free(n->weights);
        }

        // It's null if we loaded a pre trained for inference since the derivatives are just used during training
        // for backprop. If you wish to re-train/finetune a pre train model, uncomment the malloc in the load_model function below
        if (l->derivatives != NULL) free(l->derivatives);
        free(l->outputs);
        free(l->raw_outputs);
        free(l->neurons);
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
        fwrite(&l->n_neurons, sizeof(int), 1, f);
        // Write number of weights
        fwrite(&l->neurons[0].n_weights, sizeof(int), 1, f);

        // For each neuron
        for (int j = 0; j < l->n_neurons; j++) {
            neuron* n = &l->neurons[j];
            // Write weights and bias
            fwrite(n->weights, sizeof(double), n->n_weights, f);
            fwrite(&n->bias, sizeof(double), 1, f);
        }
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

    m->layers = malloc(m->n_layers * sizeof(layer));

    // For each layer
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];

        // Read number of neurons
        fread(&l->n_neurons, sizeof(int), 1, f);

        l->neurons = malloc(l->n_neurons * sizeof(neuron));

        l->outputs = malloc(l->n_neurons * sizeof(double));

        l->raw_outputs = malloc(l->n_neurons * sizeof(double));

        // This is just used for training, so we don't need if we're only doing inference
        // If you wish to train a model you loaded just replace the NULL by the commented malloc
        l->derivatives = NULL; // malloc(l->n_neurons * sizeof(double));

        // Read number of weights
        fread(&l->neurons[0].n_weights, sizeof(int), 1, f);

        // For each neuron
        for (int j = 0; j < l->n_neurons; j++) {
            neuron* n = &l->neurons[j];

            n->n_weights = l->neurons[0].n_weights; // Same number of weights for each neuron of a given layer
            n->weights = malloc(sizeof(double) * n->n_weights);

            // Read weights and bias
            fread(n->weights, sizeof(double), n->n_weights, f);
            fread(&n->bias, sizeof(double), 1, f);
        }

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
    printf("Successfully loaded model !\n");
    print_model(m);
}
