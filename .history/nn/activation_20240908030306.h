#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "structs.h"
#include <stdbool.h>


float activation_none(float x, bool derivative);
float activation_relu(float x, bool derivative);
void activation_softmax(FCLayer *layer);
float activation_sigmoid(float x, bool derivative);
float (*apply_activation(Activation a))(float, bool);


#endif