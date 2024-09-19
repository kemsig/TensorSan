#include <stdio.h>
#include <stdlib.h>
#include "nn/structs.h"
#include "nn/nn.h"
#include "nn/activation.h"
#include "data_loader.h"
#include <string.h>
#include "nn/cuda/nn_cuda.h"  // Include the CUDA header file
#include <cuda_runtime.h>


#define TRAIN_IMAGES "Datasets/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte"
#define TRAIN_LABELS "Datasets/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
#define IMAGE_SIZE 28*28 
#define TEST_IMAGES "Datasets/MNIST/t10k-images.idx3-ubyte"
#define TEST_LABELS "Datasets/MNIST/t10k-labels.idx1-ubyte"

int get_predicted_class(float *output, int size) {
    int predicted_class = 0;
    float max_value = output[0];

    // Find the index of the maximum value in the output array
    for (int i = 1; i < size; i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            predicted_class = i;
        }
    }
    return predicted_class;
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

	// create layers
	unsigned int epochs = 5;
	FCLayer *layer1 = init_fc_layer(IMAGE_SIZE, 128, ReLU);
	FCLayer *layer2 = init_fc_layer(128, 10, SOFTMAX);

	for (int j = 0; j < epochs; j++){
		int num_correct = 0;
		for (int i = 0; i < 60000; ++i){
			
			fc_forward_CUDA(layer1, images+(IMAGE_SIZE*i));
			cudaDeviceSynchronize(); // Ensure CUDA operations are complete
			// apply relu
			for (int i = 0; i < layer1->output_size; ++i){
				float a = activation_relu(layer1->output[i], false);
				layer1->output[i] = a;

			}
			for (int i = 0; i < layer1->output_size; ++i){
				printf("%f, ", layer1->output[i]);
			}
			printf("\n");

			fc_forward_CUDA(layer2, layer1->output);
			cudaDeviceSynchronize(); // Ensure CUDA operations are complete
			activation_softmax(layer2);
			
			// for (int i = 0; i < layer2->output_size; ++i){
			// 	printf("%f, ", layer2->output[i]);
			// }
			// printf("\n");


			// find d_output
			float *d_output = derivative_softmax_categorical_cross_entropy(layer2->output, one_hot_matrix+(10*i), 10);

			fc_backward(layer2, d_output);
			fc_backward(layer1, layer2->d_input);
			
			// calculate loss
			float loss = categorical_cross_entropy(layer2->output, one_hot_matrix+(10*i), layer2->output_size);

			if (*(labels+(i)) == get_predicted_class(layer2->output, 10))
				num_correct++;
			if (i % 1 == 0)
				printf("epoch: %d, loss, %f, actual: %d, predicted: %d, accuracy: %f\n", j+1, loss, *(labels+(i)), get_predicted_class(layer2->output, 10), num_correct/(float)i);

			// update weights
			float learning_rate = 0.001f;
			update_weights(layer2, learning_rate);
			update_weights(layer1, learning_rate);
			
			free(d_output);
		}
	}

	// load the training data
	unsigned char *test_labels;
	float *test_images;

	load_mnist_images(TEST_IMAGES, &test_images, &num_images);
	printf("Num of images: %d\n", num_images);

	load_mnist_labels(TEST_LABELS, &test_labels, &num_labels);
	printf("Num of labels: %d\n", num_labels);

	//one hot encoding
    float *test_one_hot_matrix = (float *)malloc(num_labels * 10 * sizeof(float));
	one_hot_encode(test_labels, num_labels, test_one_hot_matrix, 10);

	int num_correct = 0;
	for (int i = 0; i < 1000; ++i){
		fc_forward_CUDA(layer1, test_images+(IMAGE_SIZE*i));
		// apply relu
		for (int i = 0; i < layer1->output_size; ++i){
			float a = activation_relu(layer1->output[i], false);
			layer1->output[i] = a;
		}

		fc_forward_CUDA(layer2, layer1->output);
		// apply softmax
		activation_softmax(layer2);

		if (*(test_labels+(i)) == get_predicted_class(layer2->output, 10))
			num_correct++;

		if (i % 5 == 0)
			printf("accuracy: %f, actual: %d, predicted: %d\n", num_correct/10000.0f, *(test_labels+(i)), get_predicted_class(layer2->output, 10));
	}
	

	// Free allocated memory
    free(one_hot_matrix);	
	free(images);
	free(labels);
}
