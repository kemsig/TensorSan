#include <stdio.h>
#include <stdlib.h>
#include "nn/structs.h"
#include "nn/nn.h"
#include "nn/activation.h"
#include "data_loader.h"
#include <string.h>

#define TRAIN_IMAGES "Datasets/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte"
#define TRAIN_LABELS "Datasets/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
#define IMAGE_SIZE 28*28


// Function to perform one-hot encoding for MNIST labels
void one_hot_encode(unsigned char *labels, int num_labels, float *one_hot_matrix, int num_classes) {
    // Initialize the matrix to all zeros
    memset(one_hot_matrix, 0, num_labels * num_classes * sizeof(int));

    for (int i = 0; i < num_labels; ++i) {
        int label = labels[i];
        if (label >= 0 && label < num_classes) {
            one_hot_matrix[i * num_classes + label] = 1;
        }
    }
}

void update_weights(FCLayer *layer, float learning_rate) {
    // Update weights
    for (int i = 0; i < layer->input_size * layer->output_size; ++i) {
        layer->weights[i] -= learning_rate * layer->d_weights[i];
    }

    // Update biases
    for (int i = 0; i < layer->output_size; ++i) {
        layer->biases[i] -= learning_rate * layer->d_biases[i];
    }
}

int main(){
	unsigned char *labels;
	float *images;
	int num_images, num_labels;
	
	load_mnist_images(TRAIN_IMAGES, &images, &num_images);
	printf("Num of images: %d\n", num_images);

	load_mnist_labels(TRAIN_LABELS, &labels, &num_labels);
	printf("Num of labels: %d\n", num_labels);

	for (int i = 0; i < 1; ++i){
		print_mnist_index(images, labels, i);
	}

	//one hot encoding
    float *one_hot_matrix = (float *)malloc(num_labels * 10 * sizeof(float));
	one_hot_encode(labels, num_labels, one_hot_matrix, 10);
	// // Print the one-hot encoded matrix
    // printf("One-hot encoded matrix:\n");
    // for (int i = 0; i < 1; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         printf("%f ", one_hot_matrix[i * 10 + j]);
    //     }
    //     printf("\n");
    // }

	// create layers
	FCLayer *layer1 = init_fc_layer(IMAGE_SIZE, 10, ReLU);
	FCLayer *layer2 = init_fc_layer(10, 10, SOFTMAX);

	for (int i = 0; i < 2; ++i){
		printf("LABEL++++++ %c", labels+(i));
		fc_forward(layer1, images+(728*i));
		fc_forward_softmax(layer2, layer1->output);

		// find d_output
		float *d_output = derivative_softmax_categorical_cross_entropy(layer2->output, one_hot_matrix+(10*i), 10);

		//float d_output[10] = {0.5f, 0.3f, 0.5f, 0.3f, 0.5f, 0.3f, 0.5f, 0.3f, 0.5f, 0.3f}; // Example gradient
		fc_backward(layer2, d_output);
		fc_backward(layer1, layer2->d_input);

		// update weights
		float learning_rate = 0.01f;
		update_weights(layer2, learning_rate);
		update_weights(layer1, learning_rate);


	}
	


	// Free allocated memory
    free(one_hot_matrix);	
	free(images);
	free(labels);
}
