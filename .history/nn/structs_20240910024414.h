#ifndef STRUCTS_H
#define STRUCTS_H

typedef enum {
    NONE = 0,
    ReLU = 1,
    SOFTMAX = 2,
    SIGMOID = 3
}Activation;

typedef struct {
    unsigned int input_size;
    unsigned int output_size;
    float *weights;
    float *biases;
    float *input;
    float *output;
    float *d_weights;
    float *d_biases;
    float *d_input;
    Activation activation_function;
}FCLayer;

#endif