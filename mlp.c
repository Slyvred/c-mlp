#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Fonctions Mathématiques ---
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_deriv(double x) { return x * (1.0 - x); }
double ranged_rand(double min, double max) { return ((double)rand() / RAND_MAX) * (max - min) + min; }


#define N_HIDDEN 128 // Nombre de neurones cachés
#define N_OUTPUT 4 // Nombre de neurones de sortie

typedef struct {
    double input;                           // Entrée (entier base 10)
    double hidden[N_HIDDEN];                // 16 neurones cachés
    double output[N_OUTPUT];                // 4 neurones de sortie (les 4 bits)

    // Poids et Bias
    double w_ih[N_HIDDEN];                  // Poids Entrée -> Cachée
    double b_h[N_HIDDEN];                   // Biais couche cachée
    double w_ho[N_HIDDEN][N_OUTPUT];        // Poids Cachée -> Sortie
    double b_o[N_OUTPUT];                   // Biais couche sortie
    double (*activation_fn)(double);        // Fonction d'activation
    double (*activation_fn_deriv)(double);  // Dérivée de de la fn d'activation
} MLP;

// Initialisation aléatoire
void init_mlp(MLP *m, double (*activation_fn)(double), double (*activation_fn_deriv)(double)) {

    // Définition des fonctions d'activations
    m->activation_fn = activation_fn;
    m->activation_fn_deriv = activation_fn_deriv;

    for (int i = 0; i < N_HIDDEN; i++) {
        m->w_ih[i] = ranged_rand(-1, 1);
        m->b_h[i] = ranged_rand(-1, 1);
        for (int j = 0; j < N_OUTPUT; j++) {
            m->w_ho[i][j] = ranged_rand(-1, 1);
        }
    }
    for (int j = 0; j < N_OUTPUT; j++) m->b_o[j] = ranged_rand(-1, 1);
}

// Passage en avant (Forward Pass)
void forward(MLP *m, double input, double max_input_val) {
    m->input = input / max_input_val; // Normalisation de l'entrée entre 0 et 1

    // Entrée -> Cachée
    for (int i = 0; i < N_HIDDEN; i++) {
        m->hidden[i] = m->activation_fn(m->input * m->w_ih[i] + m->b_h[i]);
    }

    // Cachée -> Sortie
    for (int j = 0; j < N_OUTPUT; j++) {
        double sum = m->b_o[j];
        for (int i = 0; i < N_HIDDEN; i++) {
            sum += m->hidden[i] * m->w_ho[i][j];
        }
        m->output[j] = m->activation_fn(sum);
    }
}

// Entraînement (Backpropagation)
void train(MLP *m, double target[N_OUTPUT], double lr) {
    double delta_o[N_OUTPUT];
    // 1. Calcul de l'erreur en sortie
    for (int j = 0; j < N_OUTPUT; j++) {
        double error = target[j] - m->output[j];
        delta_o[j] = error * m->activation_fn_deriv(m->output[j]);
    }

    // 2. Calcul de l'erreur sur la couche cachée
    double delta_h[N_HIDDEN];
    for (int i = 0; i < N_HIDDEN; i++) {
        double error = 0;
        for (int j = 0; j < N_OUTPUT; j++) {
            error += delta_o[j] * m->w_ho[i][j];
        }
        delta_h[i] = error * m->activation_fn_deriv(m->hidden[i]);
    }

    // 3. Mise à jour des poids Cachée -> Sortie
    for (int j = 0; j < N_OUTPUT; j++) {
        for (int i = 0; i < N_HIDDEN; i++) {
            m->w_ho[i][j] += lr * delta_o[j] * m->hidden[i];
        }
        m->b_o[j] += lr * delta_o[j];
    }

    // 4. Mise à jour des poids Entrée -> Cachée
    for (int i = 0; i < N_HIDDEN; i++) {
        m->w_ih[i] += lr * delta_h[i] * m->input;
        m->b_h[i] += lr * delta_h[i];
    }
}

int main(int argc, char** argv) {
    srand(time(NULL)); // Pour la reproductibilité
    MLP m;
    init_mlp(&m, sigmoid, sigmoid_deriv);

    int epochs = 70000;
    double lr = 0.5;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        lr = atof(argv[2]);
    }

    double dataset[16][4] = {
        {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
        {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
        {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
        {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
    };

    printf("Entraînement en cours...\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < 16; i++) {
            forward(&m, (double)i, 15.0);
            train(&m, dataset[i], lr);
        }
    }

    printf("\n--- Résultats ---\n");
    for (int i = 0; i < 16; i++) {
        forward(&m, (double)i, 15.0);
        printf("In: %2d | Out: [%.0f %.0f %.0f %.0f]\n",
               i, round(m.output[0]), round(m.output[1]), round(m.output[2]), round(m.output[3]));
    }
    return 0;
}
