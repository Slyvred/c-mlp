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
    double* weights;
    double n_weights;
    double output;
    double bias;
    double delta; // Neuron error
}neuron;

// layer = list of neurons with activation function
typedef struct {
    int n_neurons;
    neuron* neurons;
    function* activation_function;
    double* outputs; // Output of each neuron of the layer
}layer;

typedef struct {
    layer** layers; // list of layers pointers
    int n_layers;
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
    // neuron->inputs = malloc(n_parameters * sizeof(double));
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
    layer->outputs = malloc(n_neurons * sizeof(double));

    for (int i = 0; i < n_neurons; i++) { init_neuron(&layer->neurons[i], n_parameters); }
}

// Passage en avant (Forward Pass)
void forward(MLP *m, double* inputs, int n_inputs) {
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = m->layers[i];
        double* layer_inputs;

        // For input layer, input is the actual input
        if (i == 0) layer_inputs = inputs;
        else layer_inputs = m->layers[i - 1]->outputs;

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
    layer* l = m->layers[m->n_layers - 1];
    for (int i = 0; i < l->n_neurons; i++) {
        double output = l->neurons[i].output;
        neuron* n = &l->neurons[i];

        // Error = (target - output) * f'(output)
        n->delta = (target[i] - output) * l->activation_function->df(n->output);
    }

    // Calculate layer error from last hidden layer to input
    for (int i = m->n_layers - 2; i >= 0; i--) {
        layer* curr_layer = m->layers[i];
        layer* next_layer = m->layers[i + 1];

        for (int j = 0; j < curr_layer->n_neurons; j++) {
            neuron* n = &curr_layer->neurons[j];
            double error = 0;

            // Sum deltas of next layer weighted by next neuron weights
            for (int k = 0; k < next_layer->n_neurons; k++) {
                neuron* next_n = &next_layer->neurons[k];
                // next_n->weights[j] is the weight connecting the current neuron with the next neuron k
                error += next_n->delta * next_n->weights[j];
            }
            n->delta = error * m->layers[0]->activation_function->df(n->output);
        }
    }

    // Update weights for each layer
    for (int i = 0; i < m->n_layers; i++) {
        layer* l = m->layers[i];
        double* inputs_for_layer;

        // For input layer, input is the actual input
        if (i == 0) inputs_for_layer = raw_inputs;
        else inputs_for_layer = m->layers[i - 1]->outputs;

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

int main(int argc, char** argv) {
    srand(42); // Pour la reproductibilité

    layer hidden, output;
    function sig = {sigmoid, sigmoid_deriv};            // Activation function
    init_layer(&hidden, 4, 2, &sig);                   // 8 neurons with 1 weights eachh
    init_layer(&output, 1, hidden.n_neurons, &sig);     // 4 neurons with 8 weights each

    layer* layers[2] = {&hidden, &output};
    MLP model = {layers, 2};

    double X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double y[4][1] = {{0},   {1},   {1},   {0}};

    // double dataset[16][4] = {
    //     {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
    //     {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
    //     {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
    //     {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
    // };

    int epochs = 70000;
    double lr = 0.5;

    if (argc == 3) {
        epochs = atoi(argv[1]);
        lr = atof(argv[2]);
    }


    printf("Entraînement en cours...\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        int i = rand() % 4;
        forward(&model, X[i], 1);
        train(&model, X[i], y[i], lr);
    }

    printf("\n--- Résultats ---\n");
    for (int i = 0; i < 4; i++) {
        forward(&model, X[i], 1);
        double* outputs = model.layers[model.n_layers - 1]->outputs;
        printf("In: %2d | Out: [%.2f] | Expected: [%f]\n",
               i, outputs[0], y[i][0]);
    }
    return 0;
}
