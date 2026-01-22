#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Fonctions Mathématiques ---
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_deriv(double x) { return x * (1.0 - x); }

typedef struct {
    double input;
    double hidden[16]; // 16 neurones cachés
    double output[4];  // 4 neurones de sortie (les 4 bits)

    // Poids et Bias
    double w_ih[16];   // Poids Entrée -> Cachée
    double b_h[16];    // Biais couche cachée
    double w_ho[16][4];// Poids Cachée -> Sortie
    double b_o[4];     // Biais couche sortie
} MLP;

// Initialisation aléatoire
void init_mlp(MLP *m) {
    for (int i = 0; i < 16; i++) {
        m->w_ih[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        m->b_h[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        for (int j = 0; j < 4; j++) {
            m->w_ho[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int j = 0; j < 4; j++) m->b_o[j] = ((double)rand() / RAND_MAX) * 2 - 1;
}

// Passage en avant (Forward Pass)
void forward(MLP *m, double input) {
    m->input = input / 15.0; // Normalisation de l'entrée entre 0 et 1

    // Entrée -> Cachée
    for (int i = 0; i < 16; i++) {
        m->hidden[i] = sigmoid(m->input * m->w_ih[i] + m->b_h[i]);
    }

    // Cachée -> Sortie
    for (int j = 0; j < 4; j++) {
        double sum = m->b_o[j];
        for (int i = 0; i < 16; i++) {
            sum += m->hidden[i] * m->w_ho[i][j];
        }
        m->output[j] = sigmoid(sum);
    }
}

// Entraînement (Backpropagation)
void train(MLP *m, double target[4], double lr) {
    double delta_o[4];
    // 1. Calcul de l'erreur en sortie
    for (int j = 0; j < 4; j++) {
        double error = target[j] - m->output[j];
        delta_o[j] = error * sigmoid_deriv(m->output[j]);
    }

    // 2. Calcul de l'erreur sur la couche cachée
    double delta_h[16];
    for (int i = 0; i < 16; i++) {
        double error = 0;
        for (int j = 0; j < 4; j++) {
            error += delta_o[j] * m->w_ho[i][j];
        }
        delta_h[i] = error * sigmoid_deriv(m->hidden[i]);
    }

    // 3. Mise à jour des poids Cachée -> Sortie
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 16; i++) {
            m->w_ho[i][j] += lr * delta_o[j] * m->hidden[i];
        }
        m->b_o[j] += lr * delta_o[j];
    }

    // 4. Mise à jour des poids Entrée -> Cachée
    for (int i = 0; i < 16; i++) {
        m->w_ih[i] += lr * delta_h[i] * m->input;
        m->b_h[i] += lr * delta_h[i];
    }
}

int main() {
    srand(42); // Pour la reproductibilité
    MLP m;
    init_mlp(&m);

    double dataset[16][4] = {
        {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
        {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
        {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
        {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
    };

    printf("Entraînement en cours...\n");
    for (int epoch = 0; epoch < 50000; epoch++) {
        for (int i = 0; i < 16; i++) {
            forward(&m, (double)i);
            train(&m, dataset[i], 0.5);
        }
    }

    printf("\n--- Résultats ---\n");
    for (int i = 0; i < 16; i++) {
        forward(&m, (double)i);
        printf("In: %2d | Out: [%.0f %.0f %.0f %.0f]\n",
               i, round(m.output[0]), round(m.output[1]), round(m.output[2]), round(m.output[3]));
    }
    return 0;
}
