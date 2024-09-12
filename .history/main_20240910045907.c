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

	// create layers
	FCLayer *layer1 = init_fc_layer(IMAGE_SIZE, 10, ReLU);
	FCLayer *layer2 = init_fc_layer(10, 10, SOFTMAX);

	fc_forward(layer1, images+(728*0));
	//fc_forward(layer2, layer1->output);
	fc_forward_softmax(layer2, layer1->output);

	//one hot encoding
    float *one_hot_matrix = (float *)malloc(num_labels * 10 * sizeof(float));
	one_hot_encode(labels, num_labels, one_hot_matrix, 10);

	// find d_output
	derivative_softmax_categorical_cross_entropy(float *predicted, one_hot_matrix, size_t 10)

	float d_output[10] = {0.5f, 0.3f, 0.5f, 0.3f, 0.5f, 0.3f, 0.5f, 0.3f, 0.5f, 0.3f}; // Example gradient
	fc_backward(layer2, d_output);
	fc_backward(layer1, layer2->d_input);



	free(images);
	free(labels);
}
