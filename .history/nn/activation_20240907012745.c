#include <stdio.h>
#include <math.h>
#include "activation.h"

float activation_none(float x){
    return x;
}

float activation_relu(float x){
    //float lol = fmaxf(0.0f, x);
   // printf("%f", lol);
    // for some reason fmaxf isn't working figure out later, bt 
    if (x > 0)
        return x;
    return 0.0f;
}

float activation_softmax(float x){
    return x;
}

float (*apply_activation(Activation a))(float){
    switch (a){
        case NONE:
            return activation_none;
        case ReLU:
            return activation_relu;
        case SOFTMAX:
            return activation_softmax;
    }
    return activation_none;
}