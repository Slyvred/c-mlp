#include <stdio.h>
#include <stdlib.h>
#include "mlp.h"
#include "math_functions.h"
#include "mongoose.h"

MLP model;

typedef struct {
    int label;
    double confidence;
}output;

output predict(double* image) {
    forward(&model, image, 784);
    double* outputs = model.layers[model.n_layers - 1].outputs;
    int class = index_of_max(outputs, 10);
    output predicted = {class, outputs[class]};
    return predicted;
}

static void fn(struct mg_connection *c, int ev, void *ev_data) {

    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;
        // Connection = 1. http upgrade request 2. upgrade connection to websocket
        if (mg_match(hm->uri, mg_str("/"), NULL)) {
            mg_ws_upgrade(c, hm, NULL);
        }
    }

    else if (ev == MG_EV_WS_OPEN) {
        printf("WebSocket client connected\n");
    }

    else if (ev == MG_EV_WS_MSG) {
        struct mg_ws_message *wm = (struct mg_ws_message *) ev_data;

        if (wm->data.len == 784) {
            unsigned char* raw_data = (unsigned char*) wm->data.buf;
            double input_vector[784];

            // Normalize vector here
            for (int i = 0; i < 784; i++) {
                input_vector[i] = (double)raw_data[i] / 255.0;
            }

            output prediction = predict(input_vector);
            char response[32];
            int len = sprintf(response, "%d - %.2f", prediction.label, prediction.confidence); // Format prediction and confidence
            mg_ws_send(c, response, len, WEBSOCKET_OP_TEXT);

        } else {
            printf("Error: received %lu bytes - expected 784)\n", (unsigned long)wm->data.len);
        }
    }
}

int main(int argc, char** argv) {
    char *model_path = getenv("MODEL_PATH");
    load_model(&model, model_path);

    struct mg_mgr mgr;
    mg_mgr_init(&mgr);

    mg_http_listen(&mgr, "http://0.0.0.0:8000", fn, NULL);
    printf("Listening on ws://127.0.0.1:8000\n");

    for (;;) {
        mg_mgr_poll(&mgr, 100); // 100ms
    }

    mg_mgr_free(&mgr);
    free_model(&model);
    return 0;
}
