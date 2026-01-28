#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "math_functions.h"

double ranged_rand(double min, double max) {
    return ((double)rand() / RAND_MAX) * (max - min) + min;
}

void init_neuron(neuron* neuron, int n_parameters) {
    // neuron->inputs = malloc(n_parameters * sizeof(double));
    neuron->weights = malloc(n_parameters * sizeof(double));
    neuron->n_weights = n_parameters;

    double limit = sqrt(2.0 / n_parameters);

    // Random init value for neurons and bias
    for (int i = 0; i < n_parameters; i++) {
        neuron->weights[i] = ranged_rand(-limit, limit);
    }
    neuron->bias = 0;
}

layer dense(int n_neurons, int n_parameters, function *activation_function) {
    layer layer;
    layer.activation_function = activation_function;
    layer.n_neurons = n_neurons;
    layer.neurons = malloc(n_neurons * sizeof(neuron));
    layer.outputs = malloc(n_neurons * sizeof(double));
    layer.raw_outputs = malloc(n_neurons * sizeof(double));
    layer.derivatives = malloc(n_neurons * sizeof(double));

    for (int i = 0; i < n_neurons; i++) { init_neuron(&layer.neurons[i], n_parameters); }
    return layer;
}

// Passage en avant (Forward Pass)
void forward(MLP *m, double* inputs, int n_inputs) {

    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];
        double* layer_inputs;

        // For input layer, input is the actual input
        if (i == 0) layer_inputs = inputs;
        else layer_inputs = m->layers[i - 1].outputs;

        for (int j = 0; j < l->n_neurons; j++) {
            neuron* n = &l->neurons[j];
            n->output = sum(layer_inputs, n->weights, n->bias, n->n_weights);
            l->raw_outputs[j] = n->output;
        }
        l->activation_function->f(l->raw_outputs, l->outputs, l->n_neurons);
    }
}

// Entraînement (Backpropagation)
void train(MLP *m, double* raw_inputs, double* target, double lr) {
    // Calculate output layer error
    layer* l = &m->layers[m->n_layers - 1];
    l->activation_function->df(l->raw_outputs, l->derivatives, l->n_neurons);
    for (int i = 0; i < l->n_neurons; i++) {
        // double output = l->neurons[i].output;
        neuron* n = &l->neurons[i];

        // Error = (target - output) * f'(output)
        n->delta = (l->outputs[i] - target[i]) * l->derivatives[i];
    }

    // Calculate layer error from last hidden layer to input
    for (int i = m->n_layers - 2; i >= 0; i--) {
        layer* curr_layer = &m->layers[i];
        layer* next_layer = &m->layers[i + 1];

        curr_layer->activation_function->df(curr_layer->raw_outputs, curr_layer->derivatives, curr_layer->n_neurons);

        for (int j = 0; j < curr_layer->n_neurons; j++) {
            neuron* n = &curr_layer->neurons[j];
            double error = 0;

            // Sum deltas of next layer weighted by next neuron weights
            for (int k = 0; k < next_layer->n_neurons; k++) {
                neuron* next_n = &next_layer->neurons[k];
                // next_n->weights[j] is the weight connecting the current neuron with the next neuron k
                error += next_n->delta * next_n->weights[j];
            }
            // n->delta = error * curr_layer->activation_function->df(n->output);
            n->delta = error * curr_layer->derivatives[j];
        }
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

double* one_hot(int input, int n_classes) {
    double* one_hot = calloc(n_classes, sizeof(double));
    one_hot[input] = 1.0;
    return one_hot;
}

void free_model(MLP* m) {
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = &m->layers[i];

        for (int j = 0; j < l->n_neurons; j++) {
            neuron* n = &l->neurons[j];
            free(n->weights);
        }

        free(l->derivatives);
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
            fwrite(&n->weights, sizeof(double), n->n_weights, f);
            fwrite(&n->bias, sizeof(double), 1, f);
        }
        // TODO Write activation function name
    }
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
        layer l = &m->layers[i];

        // Write number of neurons
        fwrite(&l->n_neurons, sizeof(int), 1, f);
        // Write number of weights
        fwrite(&l->neurons[0].n_weights, sizeof(int), 1, f);

        // For each neuron
        for (int j = 0; j < l->n_neurons; j++) {
            neuron* n = &l->neurons[j];
            // Write weights and bias
            fwrite(&n->weights, sizeof(double), n->n_weights, f);
            fwrite(&n->bias, sizeof(double), 1, f);
        }
        // TODO Write activation function name
    }
}
