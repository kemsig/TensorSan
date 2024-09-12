#include <stdio.h>
#include <stdlib.h>
#include "nn/structs.h"
#include "nn/nn.h"
#include "nn/activation.h"
#include "data_loader.h"


#define TRAIN_IMAGES "Datasets/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte"
#define TRAIN_LABELS "Datasets/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
#define IMAGE_SIZE 28*28
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
	FCLayer *layer2 = init_fc_layer(10, 10, ReLU);
	FCLayer *layer3 = init_fc_layer(10, 2, ReLU);

	fc_forward(layer1, images+(728*0));
	fc_forward(layer2, layer1->output);
	fc_forward(layer3, layer2->output);
	//fc_forward_softmax(layer2, layer1->output);
	float d_output[2] = {0.5f, 0.3f}; // Example gradient
	fc_backward(layer3, d_output);



	free(images);
	free(labels);
}
