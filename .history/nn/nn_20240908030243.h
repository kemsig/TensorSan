#ifndef NN_BASE_H
#define NN_BASE_H

float random_float();

FCLayer* init_fc_layer(unsigned int input_size, unsigned int output_size, Activation afunc);

void fc_forward(FCLayer *layer, float *input);

#endif