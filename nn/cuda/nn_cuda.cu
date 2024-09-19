#include <cuda_runtime.h>
#include <stdio.h>
#include "../structs.h"
#include "../activation.h"
extern "C" void fc_forward_CUDA(FCLayer *layer, float *input);

__global__ void matrix_mul(float *input, float *weights, float *biases, float *output, int input_size, int output_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < output_size){
        float result = biases[i];

        for (int j = 0; j < input_size; ++j){
            result += input[j] * weights[j  * output_size + i];
        }

        output[i] = result;
    }
}

extern "C" void fc_forward_CUDA(FCLayer *layer, float *input){
    // define pointers to the gpu
    float *gpu_input, *gpu_weights, *gpu_biases, *gpu_output;

    // allocate memory for the gpu
    cudaMalloc((float**)&gpu_input, layer->input_size * sizeof(float));
    cudaMalloc((float**)&gpu_weights, layer->input_size * layer->output_size * sizeof(float));
    cudaMalloc((float**)&gpu_biases, layer->output_size * sizeof(float));
    cudaMalloc((float**)&gpu_output, layer->output_size * sizeof(float));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(gpu_input, input, layer->input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weights, layer->weights, layer->input_size * layer->output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_biases, layer->biases, layer->output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    int blockSize = 256;
    int gridSize = (layer->output_size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    matrix_mul<<<gridSize, blockSize>>>(gpu_input, gpu_weights, gpu_biases, gpu_output, layer->input_size, layer->output_size);

    cudaDeviceSynchronize();

    // Copy the output data back from the GPU to the CPU
    cudaMemcpy(layer->output, gpu_output, layer->output_size * sizeof(float), cudaMemcpyDeviceToHost);
    

    // Free the GPU memory
    cudaFree(gpu_input);
    cudaFree(gpu_weights);
    cudaFree(gpu_biases);
    cudaFree(gpu_output);

    // // get the activation function
    // float (*acti_func)(float,bool) = apply_activation(layer->activation_function);

    // // apply activation function on outputs
    // for (int i = 0; i < layer->output_size; ++i){
    //     float a = acti_func(layer->output[i], false);
    //     layer->output[i] = a;
    // }
}