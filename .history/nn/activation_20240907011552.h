#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef enum {
    NONE = 0,
    ReLU = 1,
    SOFTMAX = 2
}Activation;

float relu(float x);
float softmax(float x);
float (*apply_activation(Activation a))(float);


#endif