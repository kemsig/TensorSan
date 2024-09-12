#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "activation.h"

float activation_none(float x, bool derivative){
    if (derivative)
        return 1;
    return x;
}

float activation_relu(float x, bool derivative){
    if (!derivative) return fmaxf(0.0f, x);

    if (x > 0)
            return 1.0f;
        return 0.0f;
}

void activation_softmax(FCLayer *layer){
    float max = -FLT_MAX;
    
    // Find the maximum value in the output array for numerical stability
    for (int i = 0; i < layer->output_size; ++i) {
        if (layer->output[i] > max) {
            max = layer->output[i];
        }
    }
    
    // Compute the sum of exponentials after subtracting the max value
    float total = 0.0f;
    for (int i = 0; i < layer->output_size; ++i){
        total += exp(layer->output[i]);
    }

    printf("layer %f=====\n", total);
    // change outputs
    for (int i = 0; i < layer->output_size; ++i){
        printf("old %f, new ", layer->output[i]);
        layer->output[i] = exp(layer->output[i]) / total;
        printf("%f\n", layer->output[i]);
    }
    
}

void derivative_ce_softmax();

float activation_sigmoid(float x, bool derivative){
    if (derivative){
        float sigmoid = activation_sigmoid(x, false);
        return sigmoid * (1.0f - sigmoid);
    }
    return 1.0f / (1.0f + exp(-x));
}


float (*apply_activation(Activation a))(float, bool){
    switch (a){
        case NONE:
            return activation_none;
        case ReLU:
            return activation_relu;
        case SOFTMAX:
            return activation_none;             // TEMPORARY
        case SIGMOID:
            return activation_sigmoid;
    }
    return activation_none;
}