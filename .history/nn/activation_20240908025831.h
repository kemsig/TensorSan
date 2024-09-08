#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdbool.h>
#include "nn.h"
typedef enum {
    NONE = 0,
    ReLU = 1,
    SOFTMAX = 2,
    SIGMOID = 3
}Activation;

float activation_none(float x, bool derivative);
float activation_relu(float x, bool derivative);
void activation_softmax(FCLayer *layer);
float activation_sigmoid(float x, bool derivative);
float (*apply_activation(Activation a))(float, bool);


#endif