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
    layer->output = (float *)malloc(output_size * sizeof(float));
    layer->activation_function = afunc;

    // initialize random weights
    for (int i = 0; i < input_size*output_size; ++i){
        layer->weights[i] = random_float();
    }

    // initialize bias to 0
    for (int i = 0; i < output_size; ++i) {
        layer->biases[i] = 0.0f;
    }

    // return layer
    return layer;
}

void fc_forward(FCLayer *layer, float *input){
    // idk why look more into
    memcpy(layer->output, layer->biases, layer->output_size * sizeof(float));

    // compute output = input * weight + biases
    for (int i = 0; i < layer->output_size; ++i){
        for (int j = 0; j < layer->input_size; j++) {
            layer->output[i] += input[j] * layer->weights[j * layer->output_size + i];
        }
    }
    
    // get the activation function
    float (*acti_func)(float*,bool) = apply_activation(layer->activation_function);

    printf("layer=====\n");
    // apply activation function on outputs
    for (int i = 0; i < layer->output_size; ++i){
        float a = acti_func(layer->output[i], false);
        printf("old %f, new ", layer->output[i]);
        printf("%f\n", a);
        layer->output[i] = a;
    }

}

void fc_forward_softmax(FCLayer *layer, float *input){
    if (layer->activation_function != SOFTMAX){
        fprintf(stderr, "Tried to go forward on a non softmax layer");
        exit(1);
    }


    // idk why look more into
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




// the loss function returns the COST
float mean_squared_error(float *predicted, float *actual, int size){
    return 0.0f;
}

void fc_backward(FCLayer *layer, float *t);
