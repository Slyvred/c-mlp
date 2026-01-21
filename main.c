#include <assert.h>
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

    layer hidden_layer_1, hidden_layer_2, output_layer;
    layer* hidden_layers[2] = {&hidden_layer_1, &hidden_layer_2};
    init_layer(&hidden_layer_1, 2, 2);
    init_layer(&hidden_layer_2, 2, 2);
    init_layer(&output_layer, 1, 2);

    mlp mlp = {
        hidden_layers,
        2,
        output_layer
    };

    print_mlp(&mlp);

    printf("\n--- Training ---\n");
    // Augmente un peu le nombre d'époques pour être sûr
    int epochs = 10000;
    // Réduis un peu le rate pour éviter l'instabilité avec ReLU
    double rate = 0.05;

    for (int i = 0; i < epochs; i++) {
        double total_error = 0;

        for (int s = 0; s < 4; s++) {
            couple sample = XOR[s];

            // ==========================================
            // 1. FORWARD PASS (Passe Avant)
            // ==========================================

            // Buffer temporaire pour transporter les entrées d'une couche à l'autre
            // Taille 10 pour être large (suffisant pour ton réseau)
            double layer_inputs[10];

            // Initialisation : l'entrée de la 1ère couche est sample.X
            layer_inputs[0] = sample.X[0];
            layer_inputs[1] = sample.X[1];

            for (int j = 0; j < mlp.n_hidden_layers; j++) {
                layer* l = mlp.hidden_layers[j];

                // Calcul des sorties pour cette couche
                for (int k = 0; k < l->n_neurons; k++) {
                    neuron* n = &l->neurons[k];
                    // IMPORTANT: On utilise layer_inputs ici !
                    n->output = relu(sum(layer_inputs, n->weights, n->bias, n->n_weights));
                }

                // MISE A JOUR DU BUFFER :
                // Les sorties de cette couche deviennent les entrées de la suivante
                // On met à jour layer_inputs pour le prochain tour de boucle j
                for (int k = 0; k < l->n_neurons; k++) {
                    layer_inputs[k] = l->neurons[k].output;
                }
            }

            // Calcul sortie finale (Output Layer)
            // layer_inputs contient maintenant les sorties de la dernière couche cachée
            neuron* out_n = &mlp.output_layer.neurons[0];
            out_n->output = sigmoid(sum(layer_inputs, out_n->weights, out_n->bias, out_n->n_weights));

            // Calcul erreur
            double error = sample.c - out_n->output;
            total_error += error * error;
            out_n->delta = df_sigmoid(out_n->output) * error;

            // ==========================================
            // 2. BACK PROPAGATION
            // ==========================================

            // Calcul des deltas pour les couches cachées
            for (int j = mlp.n_hidden_layers - 1; j >= 0; j--) {
                layer* current_layer = mlp.hidden_layers[j];
                layer* next_layer = (j == mlp.n_hidden_layers - 1) ? &mlp.output_layer : mlp.hidden_layers[j + 1];

                for (int k = 0; k < current_layer->n_neurons; k++) {
                    neuron* n = &current_layer->neurons[k];
                    double sum_errors = 0;
                    for (int l = 0; l < next_layer->n_neurons; l++) {
                        neuron* next_n = &next_layer->neurons[l];
                        sum_errors += next_n->delta * next_n->weights[k];
                    }
                    n->delta = df_relu(n->output) * sum_errors;
                }
            }

            // ==========================================
            // 3. UPDATE WEIGHTS
            // ==========================================

            // Update Output Layer
            for (int j = 0; j < mlp.output_layer.n_neurons; j++) {
                neuron* n = &mlp.output_layer.neurons[j];
                for (int k = 0; k < n->n_weights; k++) {
                    // L'input était la sortie de la dernière couche cachée
                    double input_val = mlp.hidden_layers[mlp.n_hidden_layers - 1]->neurons[k].output;
                    n->weights[k] += rate * n->delta * input_val;
                }
                n->bias += rate * n->delta;
            }

            // Update Hidden Layers
            for (int j = 0; j < mlp.n_hidden_layers; j++) {
                layer* layer = mlp.hidden_layers[j];
                for (int l = 0; l < layer->n_neurons; l++) {
                    neuron* n = &layer->neurons[l];
                    for (int k = 0; k < n->n_weights; k++) {
                        // Si couche 0, input est sample.X. Sinon, sortie couche précédente
                        double input_val = (j == 0) ? sample.X[k] : mlp.hidden_layers[j-1]->neurons[k].output;
                        n->weights[k] += rate * n->delta * input_val;
                    }
                    n->bias += rate * n->delta;
                }
            }
        }

        if (i % 10000 == 0) {
            printf("Epoch %d - MSE: %f\n", i, total_error / 4.0);
        }
    }

    printf("\n--- Results ---\n");
    print_mlp(&mlp);

    printf("\n--- Final Tests ---\n");
    for (int i = 0; i < 4; i++) {
        couple input = XOR[i];

        // 1. On crée un buffer temporaire pour faire transiter les données
        // On l'alloue assez grand pour tenir le max de neurones d'une couche
        double current_inputs[10]; // Taille 10 pour être large (ou malloc)

        // 2. On copie l'entrée initiale dans ce buffer
        current_inputs[0] = input.X[0];
        current_inputs[1] = input.X[1];

        // 3. Boucle sur les couches cachées
        for (int j = 0; j < mlp.n_hidden_layers; j++) {
            layer* l = mlp.hidden_layers[j];

            // Forward pass pour cette couche
            for (int k = 0; k < l->n_neurons; k++) {
                neuron* n = &l->neurons[k];
                // On lit depuis current_inputs (copie), pas input.X
                n->output = relu(sum(current_inputs, n->weights, n->bias, n->n_weights));
            }

            // Mise à jour du buffer pour la prochaine couche
            // Les sorties de cette couche deviennent les entrées de la suivante
            get_layer_outputs(l, current_inputs);
        }

        // 4. Output layer (inchangé, sauf qu'on prend les inputs du buffer)
        for (int j = 0; j < output_layer.n_neurons; j++) {
            neuron* n = &output_layer.neurons[j];
            // current_inputs contient maintenant les sorties de la DERNIÈRE couche cachée
            n->output = sigmoid(sum(current_inputs, n->weights, n->bias, n->n_weights));
        }

        // 5. Affichage (input.X est resté intact !)
        printf("Input: [%.0f, %.0f] -> Target: %.0f -> Predicted: %f\n",
                input.X[0], input.X[1], input.c, output_layer.neurons[0].output);
    }
}
