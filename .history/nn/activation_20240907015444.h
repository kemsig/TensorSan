#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef enum {
    NONE = 0,
    ReLU = 1,
    SOFTMAX = 2
}Activation;

float activation_none(float x, bool derivative);
float activation_relu(float x, bool derivative);
float activation_softmax(float x);
float softmax(float x);
float (*apply_activation(Activation a))(float);


#endif