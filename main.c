#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "helpers.h"

typedef struct{
    double X[2];
    double c;
}couple;

int main() {
    srand(time(NULL));

    // OR
    couple OR[4] = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1}
    };

    // XOR
    couple XOR[4] = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 0}
    };

    layer hidden_layer, output_layer;
    layer* mlp[2] = {&hidden_layer, &output_layer};
    init_layer(&hidden_layer, 3, 2);
    init_layer(&output_layer, 1, 2);

    for (int i = 0; i < 2; i++) {
        print_layer(mlp[i]);
    }

    printf("\n--- Training ---\n");
    int epochs = 1024;
    double rate = 0.1;

    for (int i = 0; i < epochs; i++) {
        double total_error = 0;

        for (int s = 0; s < 4; s++) {
            couple sample = XOR[s];

            // Forward pass
            for (int j = 0; j < hidden_layer.n_neurons; j++) {
                neuron* n = &hidden_layer.neurons[j];
                n->output = relu(sum(sample.X, n->weights, n->bias, n->n_weights));
            }

            for (int j = 0; j < output_layer.n_neurons; j++) {
                neuron* n = &output_layer.neurons[j];

                double* h_outs = malloc(hidden_layer.n_neurons * sizeof(double));
                for(int k = 0; k < hidden_layer.n_neurons; k++) {
                    h_outs[k] = hidden_layer.neurons[k].output;
                }

                n->output = relu(sum(h_outs, n->weights, n->bias, n->n_weights));
                free(h_outs);

                double error = sample.c - n->output;
                total_error += error * error; // Somme des carrés pour la MSE

                // DELTA SORTIE
                n->delta = df_relu(n->output) * error;
            }

            // Back prop
            for (int j = 0; j < hidden_layer.n_neurons; j++) {
                neuron* n = &hidden_layer.neurons[j];
                double sum_errors = 0;
                for (int k = 0; k < output_layer.n_neurons; k++) {
                    sum_errors += output_layer.neurons[k].delta * output_layer.neurons[k].weights[j];
                }
                n->delta = df_relu(n->output) * sum_errors;
            }

            // Update output weights
            for (int j = 0; j < output_layer.n_neurons; j++) {
                neuron* n = &output_layer.neurons[j];
                for (int k = 0; k < n->n_weights; k++) {
                    n->weights[k] += rate * n->delta * hidden_layer.neurons[k].output;
                }
                n->bias += rate * n->delta;
            }
            // Update hidden weights
            for (int j = 0; j < hidden_layer.n_neurons; j++) {
                neuron* n = &hidden_layer.neurons[j];
                for (int k = 0; k < n->n_weights; k++) {
                    n->weights[k] += rate * n->delta * sample.X[k];
                }
                n->bias += rate * n->delta;
            }
        }

        // Affichage de la MSE toutes les 100 époques
        if (i % 100 == 0) {
            printf("Epoch %d - MSE: %f\n", i, total_error / 4.0);
        }
    }

    printf("\n--- Results ---\n");
    for (int i = 0; i < 2; i++) {
        print_layer(mlp[i]);
    }

    printf("\n--- Final Tests ---\n");
    for (int i = 0; i < 4; i++) {
        couple input = XOR[i];
        for (int i = 0; i < hidden_layer.n_neurons; i++) {
            neuron* n = &hidden_layer.neurons[i];
            n->output = relu(sum(input.X, n->weights, n->bias, n->n_weights));
        }

        for (int i = 0; i < output_layer.n_neurons; i++) {
            neuron* n = &output_layer.neurons[i];
            double h_outs[3] = {hidden_layer.neurons[0].output, hidden_layer.neurons[1].output, hidden_layer.neurons[2].output};
            n->output = relu(sum(h_outs, n->weights, n->bias, n->n_weights));
        }
        printf("Input: [%.0f, %.0f] -> Target: %.0f -> Predicted: %f\n",
                input.X[0], input.X[1], input.c, output_layer.neurons[0].output);
    }
}
