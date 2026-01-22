#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Fonctions Mathématiques ---
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_deriv(double x) { return x * (1.0 - x); }
double ranged_rand(double min, double max) { return ((double)rand() / RAND_MAX) * (max - min) + min; }

typedef struct {
    double (*f)(double);
    double (*df)(double);
}function;

// layer = parameters[], activation_fn, df_activation_fn
// n_outputs = n_parameters = n_neurons
// n_inputs

typedef struct {
    // y = a_i * x_i + b
    // len(inputs) == len(weights) == n_weights
    double* inputs;
    double* weights;
    double n_weights;
    double output;
    double bias;
}neuron;

// layer = list of neurons with activation function
typedef struct {
    int n_neurons;
    neuron* neurons;
    function* activation_function;
    double* outputs; // Output of each neuron of the layer
}layer;

typedef struct {
    layer* input;
    layer* hidden;
    layer* output;
} MLP;

double sum(double inputs[], double weights[], double bias, int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += inputs[i] * weights[i];
    }
    sum += bias;
    return sum;
}

void init_neuron(neuron* neuron, int n_parameters) {
    neuron->inputs = malloc(n_parameters * sizeof(double));
    neuron->weights = malloc(n_parameters * sizeof(double));
    neuron->n_weights = n_parameters;

    // Random init value for neurons and bias
    for (int i = 0; i < n_parameters; i++) {
        neuron->weights[i] = ranged_rand(-1, 1);
    }
    neuron->bias = ranged_rand(-1, 1);
}

void init_layer(layer *layer, int n_neurons, int n_parameters, function *activation_function) {
    layer->activation_function = activation_function;
    layer->n_neurons = n_neurons;
    layer->neurons = malloc(n_neurons * sizeof(neuron));
    layer->outputs = malloc(n_neurons * sizeof(neuron));
    for (int i = 0; i < n_neurons; i++) {
        init_neuron(&layer->neurons[i], n_parameters);
    }
}

// Passage en avant (Forward Pass)
void forward(MLP *m, double* inputs, int n_inputs) {

    // Input layer
    layer* l = m->input;
    for (int i = 0; i < l->n_neurons; i++) {
        neuron* n = &l->neurons[i];
        n->output = l->activation_function->f(sum(inputs, n->weights, n->bias, n->n_weights));
        l->outputs[i] = n->output;
    }

    // Hidden layer
    l = m->hidden;
    double* last_layer_outputs = m->input->outputs;
    for (int i = 0; i < l->n_neurons; i++) {
        neuron* n = &l->neurons[i];
        n->output = l->activation_function->f(sum(last_layer_outputs, n->weights, n->bias, n->n_weights));
        l->outputs[i] = n->output;
    }

    // Output layer
    l = m->output;
    last_layer_outputs = m->hidden->outputs;
    for (int i = 0; i < l->n_neurons; i++) {
        neuron* n = &l->neurons[i];
        n->output = l->activation_function->f(sum(last_layer_outputs, n->weights, n->bias, n->n_weights));
        l->outputs[i] = n->output;
    }
}

// Entraînement (Backpropagation)
void train(MLP *m, double* target, double* raw_inputs, double lr) {

    // Calculate output layer error
    layer* l = m->output;
    double* delta_output = malloc(l->n_neurons * sizeof(double));
    for (int i = 0; i < l->n_neurons; i++) {
        double output = l->neurons[i].output;
        double error = target[i] - output;
        delta_output[i] = error * l->activation_function->df(output);
    }

    // Calculate hidden layer error
    l = m->hidden;
    double* delta_hidden = malloc(l->n_neurons * sizeof(double));
    for (int i = 0; i < l->n_neurons; i++) {
        double output = l->neurons[i].output;
        double error = 0;

        for (int j = 0; j < m->output->n_neurons; j++) {
            error += delta_output[j] * m->output->neurons[j].weights[i];
        }
        delta_hidden[i] = error * l->activation_function->df(output);
    }

    // Calculate input layer error
    l = m->input;
    double* delta_input = malloc(l->n_neurons * sizeof(double));
    for (int i = 0; i < l->n_neurons; i++) {
        double output = l->neurons[i].output;
        double error = 0;

        for (int j = 0; j < m->hidden->n_neurons; j++) {
            error += delta_hidden[j] * m->output->neurons[j].weights[i];
        }
        delta_input[i] = error * l->activation_function->df(output);
    }
}

int main(int argc, char** argv) {
    srand(time(NULL)); // Pour la reproductibilité
    function sig = {sigmoid, sigmoid_deriv};

    layer input, hidden, output;
    init_layer(&input, 1, 1, &sig);                     // 1 neuron with 1 weight
    init_layer(&hidden, 8, input.n_neurons, &sig);      // 8 neurons with 1 weights each
    init_layer(&output, 4, hidden.n_neurons, &sig);     // 4 neurons with 8 weights each
    MLP model = {&input, &hidden, &output};


    double dataset[16][4] = {
        {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
        {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
        {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
        {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
    };

    int epochs = 70000;
    double lr = 0.5;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        lr = atof(argv[2]);
    }


    printf("Entraînement en cours...\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < 16; i++) {
            double input[1] = { i / 15.0 }; // Normalisation de l'entrée
            forward(&model, input, 1);
            printf("In: %f | Out: [%.8f %.8f %.8f %.8f]\n", input[0], model.output->neurons[0].output, model.output->neurons[1].output, model.output->neurons[2].output, model.output->neurons[3].output);
            train(&model, dataset[i], input, lr);
        }
    }

    // printf("\n--- Résultats ---\n");
    // for (int i = 0; i < 16; i++) {
    //     forward(&m, (double)i, 15.0);
    //     printf("In: %2d | Out: [%.0f %.0f %.0f %.0f]\n",
    //            i, round(m.output[0]), round(m.output[1]), round(m.output[2]), round(m.output[3]));
    // }
    return 0;
}
