#ifndef STRUCTS_H
#define STRUCTS_H

typedef enum {
    NONE = 0,
    ReLU = 1,
    SOFTMAX = 2,
    SIGMOID = 3
}Activation;

typedef struct {
    unsigned int input_size;        // size of input layer
    unsigned int output_size;       // size of output layer
    float *weights;                 // the weights of the layer
    float *biases;                  // the bias of the layer
    float *input;                   // the inputs given to the layer
    float *output;                  // the output given from the layer
    Activation activation_function; // the activation function the layer uses
    float *d_weights;               // the change in weights for back prop
    float *d_biases;                // the change in bias for back prop
    float *d_input;                 // the change in input for back prop
}FCLayer;

#endif