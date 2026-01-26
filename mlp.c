#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEAKY_RELU_SLOPE 0.01

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_deriv(double x) { return x * (1.0 - x); }
double ranged_rand(double min, double max) { return ((double)rand() / RAND_MAX) * (max - min) + min; }
double linear(double x) { return x; };
double linear_deriv(double x) { return 1; };

double leaky_relu(double x) {
    return x >= 0 ? x : LEAKY_RELU_SLOPE * x;
}

double leaky_relu_deriv(double x) {
    return x >= 0 ? 1 : LEAKY_RELU_SLOPE;
}

double* softmax(double* inputs, int len) {
    double* outputs = malloc(len * sizeof(double));
    double denom = 0;

    for (int i = 0; i < len; i++) {
        outputs[i] = exp(inputs[i]);
        denom += outputs[i];
    }

    for (int i = 0; i < len; i++) {
        outputs[i] /= denom;
    }
    return outputs;
}

double sum(double inputs[], double weights[], double bias, int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    return sum;
}

void init_neuron(neuron* neuron, int n_parameters) {
    // neuron->inputs = malloc(n_parameters * sizeof(double));
    neuron->weights = malloc(n_parameters * sizeof(double));
    neuron->n_weights = n_parameters;

    // Random init value for neurons and bias
    for (int i = 0; i < n_parameters; i++) {
        neuron->weights[i] = ranged_rand(-0.5, 0.5);
    }
    neuron->bias = ranged_rand(-0.5, 0.5);
}

layer dense(int n_neurons, int n_parameters, function *activation_function) {
    layer layer;
    layer.activation_function = activation_function;
    layer.n_neurons = n_neurons;
    layer.neurons = malloc(n_neurons * sizeof(neuron));
    layer.outputs = malloc(n_neurons * sizeof(double));

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
            n->output = l->activation_function->f(sum(layer_inputs, n->weights, n->bias, n->n_weights));
            l->outputs[j] = n->output;
        }
    }
}

// Entraînement (Backpropagation)
void train(MLP *m, double* raw_inputs, double* target, double lr) {
    // Calculate output layer error
    layer* l = &m->layers[m->n_layers - 1];
    for (int i = 0; i < l->n_neurons; i++) {
        double output = l->neurons[i].output;
        neuron* n = &l->neurons[i];

        // Error = (target - output) * f'(output)
        n->delta = (target[i] - output) * l->activation_function->df(n->output);
    }

    // Calculate layer error from last hidden layer to input
    for (int i = m->n_layers - 2; i >= 0; i--) {
        layer* curr_layer = &m->layers[i];
        layer* next_layer = &m->layers[i + 1];

        for (int j = 0; j < curr_layer->n_neurons; j++) {
            neuron* n = &curr_layer->neurons[j];
            double error = 0;

            // Sum deltas of next layer weighted by next neuron weights
            for (int k = 0; k < next_layer->n_neurons; k++) {
                neuron* next_n = &next_layer->neurons[k];
                // next_n->weights[j] is the weight connecting the current neuron with the next neuron k
                error += next_n->delta * next_n->weights[j];
            }
            n->delta = error * curr_layer->activation_function->df(n->output);
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
                // weight += learning_rate * delta * input
                n->weights[k] += lr * n->delta * inputs_for_layer[k];
            }

            // Update bias
            n->bias += lr * n->delta;
        }
    }
}

int get_num_parameters(MLP* mlp) {
    int parameters = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        layer* l = &mlp->layers[i];
        parameters += l->n_neurons * l->neurons[0].n_weights;
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
        if (i+1 >= len) printf("%0.f", list[i]);
        else printf("%0.f ", list[i]);
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
