#pragma once
#include "mlp.h"

typedef struct {
    int x;
    int y;
}vec2;

typedef struct {
    float* inputs;
    float* outputs;
    float* filter;
    float* deltas;
    int stride;
    vec2 filter_size;
    vec2 input_size;
    function activation_fn;
}conv_layer;

typedef struct {
    conv_layer* conv_layers;
    MLP* fully_connected_layer;
}CNN;
