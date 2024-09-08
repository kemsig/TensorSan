#ifndef NN_BASE_H
#define NN_BASE_H

float random_float();

typedef struct {
    unsigned int input_size;
    unsigned int output_size;
    float *weights;
    float *biases;
    float *output;
    Activation activation_function;
}FCLayer;

#include "activation.h"

FCLayer* init_fc_layer(unsigned int input_size, unsigned int output_size, Activation afunc);

void fc_forward(FCLayer *layer, float *input);

#endif