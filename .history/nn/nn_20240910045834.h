#ifndef NN_BASE_H
#define NN_BASE_H

#include "structs.h"

float random_float();

FCLayer* init_fc_layer(unsigned int input_size, unsigned int output_size, Activation afunc);

void fc_forward(FCLayer *layer, float *input);
void fc_backward(FCLayer *layer, float *d_output);
void fc_forward_softmax(FCLayer *layer, float *input);
float* derivative_softmax_categorical_cross_entropy(float *predicted, float *actual, size_t size)

#endif