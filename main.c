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
    int epochs = 10000;
    double rate = 0.1;

    for (int i = 0; i < epochs; i++) {
        double total_error = 0;

        for (int s = 0; s < 4; s++) {
            couple sample = XOR[s];

            // --- 1. Forward pass (Passe avant) ---

            // On garde une référence vers les entrées de la couche actuelle
            double* layer_inputs = sample.X;

            for (int j = 0; j < mlp.n_hidden_layers; j++) {
                layer* l = mlp.hidden_layers[j];
                for (int k = 0; k < l->n_neurons; k++) {
                    neuron* n = &l->neurons[k];
                    // On utilise layer_inputs (qui est soit sample.X, soit la sortie précédente)
                    n->output = relu(sum(layer_inputs, n->weights, n->bias, n->n_weights));
                }

                // Pour la prochaine couche, les entrées seront les sorties de celle-ci
                // Attention: on doit créer un tableau temporaire pour pointer vers les sorties
                // Mais ici, on peut simplement dire que la prochaine couche lira l -> neurons[...].output
            }

            // Calcul sortie finale
            neuron* out_n = &mlp.output_layer.neurons[0];

            // On récupère les sorties de la dernière couche cachée manuellement
            layer* last_hidden = mlp.hidden_layers[mlp.n_hidden_layers - 1];
            double last_hidden_outputs[2]; // Hardcodé pour 2 neurones cachés
            last_hidden_outputs[0] = last_hidden->neurons[0].output;
            last_hidden_outputs[1] = last_hidden->neurons[1].output;

            out_n->output = sigmoid(sum(last_hidden_outputs, out_n->weights, out_n->bias, out_n->n_weights));

            // Calcul erreur / Delta Sortie
            double error = sample.c - out_n->output;
            total_error += error * error;
            out_n->delta = df_sigmoid(out_n->output) * error;

            // --- 2. Back propagation (Rétropropagation) ---

            // A. D'abord calculer les deltas pour TOUTES les couches cachées
            // On part de la dernière couche cachée vers la première
            for (int j = mlp.n_hidden_layers - 1; j >= 0; j--) {
                layer* current_layer = mlp.hidden_layers[j];
                // La couche suivante est soit la sortie (si on est à la dernière cachée), soit la couche cachée j+1
                layer* next_layer = (j == mlp.n_hidden_layers - 1) ? &mlp.output_layer : mlp.hidden_layers[j + 1];

                for (int k = 0; k < current_layer->n_neurons; k++) {
                    neuron* n = &current_layer->neurons[k];
                    double sum_errors = 0;

                    // On regarde tous les neurones de la couche SUIVANTE
                    for (int l = 0; l < next_layer->n_neurons; l++) {
                        neuron* next_n = &next_layer->neurons[l];
                        // L'erreur vient du delta du neurone suivant * le poids qui relie k à l
                        // Le poids est stocké dans le neurone SUIVANT, à l'index k
                        sum_errors += next_n->delta * next_n->weights[k];
                    }
                    n->delta = df_relu(n->output) * sum_errors;
                }
            }

            // --- 3. Update weights (Mise à jour des poids) ---

            // A. Mise à jour Output Layer
            for (int j = 0; j < mlp.output_layer.n_neurons; j++) {
                neuron* n = &mlp.output_layer.neurons[j];
                for (int k = 0; k < n->n_weights; k++) {
                    // L'entrée de l'output layer est la sortie de la dernière hidden layer
                    double input_val = mlp.hidden_layers[mlp.n_hidden_layers - 1]->neurons[k].output;
                    n->weights[k] += rate * n->delta * input_val;
                }
                n->bias += rate * n->delta;
            }

            // B. Mise à jour Hidden Layers
            for (int j = 0; j < mlp.n_hidden_layers; j++) {
                layer* layer = mlp.hidden_layers[j];
                for (int l = 0; l < layer->n_neurons; l++) {
                    neuron* n = &layer->neurons[l];
                    for (int k = 0; k < n->n_weights; k++) {
                        // Si j=0, l'entrée est sample.X.
                        // Sinon, c'est la sortie de la couche j-1
                        double input_val = (j == 0) ? sample.X[k] : mlp.hidden_layers[j-1]->neurons[k].output;
                        n->weights[k] += rate * n->delta * input_val;
                    }
                    n->bias += rate * n->delta;
                }
            }
        }

        // Affichage de la MSE toutes les 100 époques
        if (i % 100 == 0) {
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
