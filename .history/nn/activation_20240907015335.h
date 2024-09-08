#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef enum {
    NONE = 0,
    ReLU = 1,
    SOFTMAX = 2
}Activation;

float activation_none(float, bool)
float relu(float, bool);
float softmax(float x);
float softmax(float x);
float (*apply_activation(Activation a))(float);


#endif