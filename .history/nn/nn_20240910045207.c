#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#include "nn.h"
#include "activation.h"

float random_float() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random float in the range [-1, 1]
}

FCLayer* init_fc_layer(unsigned int input_size, unsigned int output_size, Activation afunc){
    // Create Fully Connected Layer
    FCLayer *layer = (FCLayer *)malloc(sizeof(FCLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;

    // malloc memory for weights, bias, and output
    layer->weights = (float *)malloc(input_size * output_size * sizeof(float));
    layer->biases = (float *)malloc(output_size * sizeof(float));
    layer->input = (float *)malloc(layer->input_size * sizeof(float));
    layer->output = (float *)malloc(output_size * sizeof(float));
    layer->activation_function = afunc;

    // initialize components for back prop
    layer->d_weights = (float *)malloc(input_size * output_size * sizeof(float));
    layer->d_biases = (float *)malloc(output_size * sizeof(float));
    layer->d_input = (float *)malloc(layer->input_size * sizeof(float));

    // initialize random weights
    for (int i = 0; i < input_size*output_size; ++i){
        layer->weights[i] = random_float();
    }

    // initialize bias to random small value to avoid symmetry
    for (int i = 0; i < output_size; ++i) {
        layer->biases[i] = (float)rand() / RAND_MAX * 0.01f;
    }

    // return layer
    return layer;
}

void fc_forward(FCLayer *layer, float *input){
    // copy input into layer->input
    memcpy(layer->input, input, layer->input_size * sizeof(float));

    // copy biases into output layer
    memcpy(layer->output, layer->biases, layer->output_size * sizeof(float));
    
    // compute output = input * weight + biases
    for (int i = 0; i < layer->output_size; ++i){
        for (int j = 0; j < layer->input_size; j++) {
          //  printf("w: %f b: %f in: %f\n", layer->weights[j * layer->output_size + i], layer->biases[i], input[j]);
            layer->output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }
    }
    
    // get the activation function
    float (*acti_func)(float,bool) = apply_activation(layer->activation_function);

    printf("layer=====\n");
    // apply activation function on outputs
    for (int i = 0; i < layer->output_size; ++i){
        float a = acti_func(layer->output[i], false);
        printf("old %f, new ", layer->output[i]);
        printf("%f\n", a);
        layer->output[i] = a;
    }

}

void fc_backward(FCLayer *layer, float *d_output){
    // get activation function and it's derivative
    float (*acti_func)(float, bool) = apply_activation(layer->activation_function);

    // Compute the gradient of the activation function
    float *d_activation = (float *)malloc(layer->output_size * sizeof(float));
    for (int i = 0; i < layer->output_size; ++i) {
     //   printf("output: %f\n", layer->output[i]);
        d_activation[i] = acti_func(layer->output[i], true); // Derivative
    }

    // multiply d_output by d_activation
    for (int i = 0; i < layer->output_size; ++i) {
        d_output[i] *= d_activation[i];
       // printf("doutput: %f\n", d_output[i]);
    }

    // set d_bias to be d_output
    memcpy(layer->d_biases, d_output, layer->output_size * sizeof(float));

    // compute d_weights
    memset(layer->d_weights, 0, layer->input_size * layer->output_size * sizeof(float));
    for (int i = 0; i < layer->output_size; ++i) {
        for (int j = 0; j < layer->input_size; ++j) {
            //printf("cur %f\n", layer->d_weights[j * layer->output_size + i] );
            layer->d_weights[j * layer->output_size + i] = d_output[i] * layer->input[j];
         //   printf("%f, ", layer->d_weights[j * layer->output_size + i]);
        }
    }

    // Compute d_input
    memset(layer->d_input, 0, layer->input_size * sizeof(float));
    
    for (int i = 0; i < layer->input_size; ++i) {
        for (int j = 0; j < layer->output_size; ++j) {
            layer->d_input[i] += d_output[j] * layer->weights[i * layer->output_size + j];
        }
        
    }

    // Free temporary memory
    free(d_activation);
}



void fc_forward_softmax(FCLayer *layer, float *input){
    if (layer->activation_function != SOFTMAX){
        fprintf(stderr, "Tried to go forward on a non softmax layer");
        exit(1);
    }

    // copy input into layer->input
    memcpy(layer->input, input, layer->input_size * sizeof(float));

    // copy biases into output layer
    memcpy(layer->output, layer->biases, layer->output_size * sizeof(float));

    // compute output = input * weight + biases
    for (int i = 0; i < layer->output_size; ++i){
        for (int j = 0; j < layer->input_size; j++) {
            layer->output[i] += input[j] * layer->weights[j * layer->output_size + i];      // this way mimics the transposition
        }
    }
    
    // apply softmax
    activation_softmax(layer);
}

// the loss function returns the COST
float categorical_cross_entropy(float *predicted, float *actual, int size){
    return 0.0f;
}

// for softmax
float* derivative_softmax_categorical_cross_entropy(float *predicted, float *actual, size_t size){
    // make an array to store the errors
    float* errors = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i){
        errors[i] = actual[i] - predicted[i];
    }

    return errors;
}


