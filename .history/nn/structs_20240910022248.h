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
    float *output;
    float *d_weight;
    float *d_bias;
    float *d_input;
    Activation activation_function;
}FCLayer;

#endif