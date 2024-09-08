#ifndef NN_BASE_H
#define NN_BASE_H

#include "activation.h"

float random_float();

typedef struct {
    unsigned int input_size;
    unsigned int output_size;
    float *weights;
    float *biases;
    float *output;
    Activation activation_function;
}FCLayer;

FCLayer* init_fc_layer(unsigned int input_size, unsigned int output_size, Activation afunc);

void fc_forward(FCLayer *layer, float *input);

#endif