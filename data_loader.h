#ifndef DATA_LOADER_H
#define DATA_LOADER_H
#define MNIST_IMAGE_MNUM 2051
#define MNIST_LABEL_MNUM 2049 
#define MNIST_ROW_SZ 28
#define MNIST_COL_SZ 28

/**
 * takes image file and laods it into a char (since it is only 8 bytes) array
 */
void load_mnist_images(const char *filename, float  **images, int *num_images);

void load_mnist_labels(const char*filename, unsigned char **labels, int *num_labels);

void print_mnist_index(float *images, unsigned char *labels, int index);

void one_hot_encode(unsigned char *labels, int num_labels, float *one_hot_matrix, int num_classes);

#endif

