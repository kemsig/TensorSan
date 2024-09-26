#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#include "nn.h"
#include "activation.h"
#define SEED 1234

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
        layer->biases[i] = 0.0f;
        //layer->biases[i] = (float)rand() / RAND_MAX * 0.01f;
    }

    // return layer
    return layer;
}

// Function to copy an FC layer
FCLayer *copy_fc_layer(FCLayer *original_layer) {
    // Allocate a new layer
    FCLayer *new_layer = init_fc_layer(original_layer->input_size, original_layer->output_size, original_layer->activation_function);

    // Copy the weights
    memcpy(new_layer->weights, original_layer->weights, original_layer->input_size * original_layer->output_size * sizeof(float));

    // Copy the biases
    memcpy(new_layer->biases, original_layer->biases, original_layer->output_size * sizeof(float));

    return new_layer;
}

void fc_forward(FCLayer *layer, float *input){
    // copy input into layer->input
    memcpy(layer->input, input, layer->input_size * sizeof(float));

    // copy biases into output layer
    memcpy(layer->output, layer->biases, layer->output_size * sizeof(float));
    
    // compute output = input * weight + biases
    for (int i = 0; i < layer->output_size; ++i){
        for (int j = 0; j < layer->input_size; j++) {
            layer->output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }
    }
    
    // // get the activation function
    // float (*acti_func)(float,bool) = apply_activation(layer->activation_function);

    // // apply activation function on outputs
    // for (int i = 0; i < layer->output_size; ++i){
    //     float a = acti_func(layer->output[i], false);
    //     layer->output[i] = a;
    // }

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
            layer->output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }
    }

    // apply softmax
    activation_softmax(layer);
}

void fc_backward(FCLayer *layer, float *d_output){
    // get activation function and it's derivative
    float (*acti_func)(float, bool) = apply_activation(layer->activation_function);

    // Compute the gradient of the activation function
    float *dZ = (float *)malloc(layer->output_size * sizeof(float));
    for (int i = 0; i < layer->output_size; ++i) {
        dZ[i] = d_output[i] * acti_func(layer->output[i], true); // Derivative
    }

    for (int i = 0; i < layer->input_size; ++i){
        for (int j = 0; j < layer->output_size; ++j){
            layer->d_weights[i * layer->output_size + j] = layer->input[i] * dZ[j];         //segfdault
        }
    }

    for (int i = 0; i < layer->output_size; ++i){
        layer->d_biases[i] = dZ[i];
    }

    for (int i = 0; i < layer->input_size; ++i){
        layer->d_input[i] = 0.0f;

        for (int j = 0; j < layer->output_size; ++j){
            layer->d_input[i] += dZ[j] * layer->weights[i * layer->output_size + j];
        }
    }
    free(dZ);
}

void update_weights(FCLayer *layer, float learning_rate) {
    // Update weights
    for (int i = 0; i < layer->input_size * layer->output_size; ++i) {
        layer->weights[i] = layer->weights[i] - (learning_rate * layer->d_weights[i]);
    }

    // Update biases
    for (int i = 0; i < layer->output_size; ++i) {
        layer->biases[i] -= learning_rate * layer->d_biases[i];
    }
}

// the loss function returns the COST
float categorical_cross_entropy(float *predicted, float *actual, int size){
    float loss = 0.0f;

    for (int i = 0; i < size; ++i){
        if (actual[i] == 1)
            loss -= logf(fmaxf(predicted[i], 1e-15));
    }
    return loss;
}

float* derivative_softmax_categorical_cross_entropy(float *predicted, float *actual, size_t size){
    // make an array to store the errors
    float* errors = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i){
        errors[i] = predicted[i] - actual[i];
    }

    return errors;
}


