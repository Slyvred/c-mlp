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
    couple S[4] = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1}
    };

    // AND
    // couple S[4] = {
    //     {{0, 0}, 0},
    //     {{0, 1}, 0},
    //     {{1, 0}, 0},
    //     {{1, 1}, 1}
    // };

    double rate = 0.1;
    layer input_layer;
    init_layer(&input_layer, 1, 2);
    print_layer(&input_layer);


}
