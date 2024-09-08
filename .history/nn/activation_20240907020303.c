#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "activation.h"

float activation_none(float x, bool derivative){
    if derivative
        return 1;
    return x;
}

float activation_relu(float x, bool derivative){
    if derivative{
        if (x > 0)
            return 1.0f;
        return 0.0f;
    }

    if (x > 0)
        return x;
    return 0.0f;
}

float activation_softmax(float x){
    return x;
}

float activation_sigmoid(float x, bool derivative){
    if derivative{
        float sigmoid = activation_sigmoid(x, false);
        return sigmoid * (1.0f - sigmoid);
    }
    return 1.0f / (1.0f + exp(-x));
}


float (*apply_activation(Activation a))(float){
    switch (a){
        case NONE:
            return activation_none;
        case ReLU:
            return activation_relu;
        case SOFTMAX:
            return activation_softmax;
        case SIGMOID:
            return activation_sigmoid;
    }
    return activation_none;
}