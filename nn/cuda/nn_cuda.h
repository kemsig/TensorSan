// nn_cuda.h
#ifndef NN_CUDA_H
#define NN_CUDA_H

#include "../structs.h"
#include "../activation.h"

void fc_forward_CUDA(FCLayer *layer, float *input);

#endif // NN_CUDA_H