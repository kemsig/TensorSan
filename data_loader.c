#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include "data_loader.h"

void load_mnist_images(const char *filename, float **images, int *num_images){
    // open file as read only binary mode.
    FILE *file = fopen(filename, "rb");

    // if file couldn't be open close the file
    if (!file){
        fprintf(stderr, "Could not load mnist image file.\n");
        exit(1);
    }

    // read the magic number/structure of the file
    int32_t magic_number = 0;
    fread(&magic_number, sizeof(int32_t), 1, file);
    magic_number = __builtin_bswap32(magic_number);     // convert from little to big endian
    printf("magic number: %d\n", magic_number);
    // compare magic number to verify if the file is correct
    if (MNIST_IMAGE_MNUM != magic_number){
        fprintf(stderr, "ERROR loading MNIST dataset: Not the correct file format!");
        exit(1);
    }

    // read  number of images and put into num_images
    fread(num_images, sizeof(int32_t), 1, file);
    *num_images = __builtin_bswap32(*num_images);

    // read num of rows and convert from big to little endian
    int32_t num_rows = 0;
    fread(&num_rows, sizeof(int32_t), 1, file);
    num_rows = __builtin_bswap32(num_rows);
    printf("num rows: %d\n", num_rows);
    

    // read num of cols and convert from big to little endian
    int32_t num_cols = 0;
    fread(&num_cols, sizeof(int32_t), 1, file);
    num_cols = __builtin_bswap32(num_cols);
    printf("num cols: %d\n", num_cols);

    // make space for images
    *images = (float *)malloc(*num_images * (num_cols * num_rows) * sizeof(float));
    unsigned char *raw_images = (unsigned char *)malloc(*num_images * (num_cols * num_rows) * sizeof(unsigned char));
    // put images into space
    fread(raw_images, sizeof(unsigned char), *num_images * num_cols * num_rows, file);

    // Normalize pixel values
    for (int i = 0; i < *num_images * (num_cols) * (num_rows); ++i) {
        (*images)[i] = raw_images[i] / 255.0f;
    }

    // free raw
    free(raw_images);

    // close file for memory safety
    fclose(file);
}




void load_mnist_labels(const char*filename, unsigned char **labels, int *num_labels){
    // open file
    FILE *file = fopen(filename, "rb");

    // verify file is open
    if (!file){
        fprintf(stderr, "Could not load mnist label file.\n");
        exit(1);
    }

    // read magic number
    int32_t magic_number = 0;
    fread(&magic_number, sizeof(int32_t), 1, file);
    magic_number = __builtin_bswap32(magic_number);

    // verify magic_number
    if (magic_number != MNIST_LABEL_MNUM){
        fprintf(stderr, "ERROR loading MNIST dataset: Not the correct file format!");
        exit(1);
    }

    // save number of labels
    fread(num_labels, sizeof(int32_t), 1, file);
    *num_labels = __builtin_bswap32(*num_labels);
    
    // make space for labels
    *labels = (unsigned char *)malloc(*num_labels * sizeof(unsigned char));
    fread(*labels, sizeof(unsigned char), *num_labels, file);

    // close file
    fclose(file);
}

// Function to map pixel values to characters for better visualization AI Generated
char pixel_to_char(float pixel_value) {
    if (pixel_value > 0.8f) return '#'; // Dark pixel
    else if (pixel_value > 0.6f) return 'O'; // Medium-dark pixel
    else if (pixel_value > 0.4f) return '+'; // Medium pixel
    else if (pixel_value > 0.2f) return '.';  // Light pixel
    else return ' ';  // Very light pixel (almost white)
}

void print_mnist_index(float *images, unsigned char *labels, int index){
    float *image = images + (index * MNIST_COL_SZ * MNIST_ROW_SZ);
    printf("Index: %d\t Label: %d", index, *(labels + index));
    for (int i = 0; i < MNIST_ROW_SZ; i++) {
        for (int j = 0; j < MNIST_COL_SZ; j++) {
            printf("%c ", pixel_to_char(image[i * MNIST_COL_SZ + j]));
        }
        printf("\n");
    }

}
