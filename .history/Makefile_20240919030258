# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -g
LDFLAGS = -lm

# CUDA compiler and flags
NVCC = nvcc
NVCCFLAGS = -O2

# Source files
SRCS = cuda_main.c data_loader.c nn/nn.c nn/activation.c
CU_SRC = nn/cuda/nn_cuda.cu

# Object files
OBJS = $(SRCS:.c=.o)
CU_OBJ = $(CU_SRC:.cu=.o)

# Header files
HEADERS = data_loader.h nn/nn.h nn/activation.h nn/structs.h nn/cuda/nn_cuda.h

# Target to build the executable
all: tensor_san

# Link the object files to create the executable
tensor_san: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

tensor_san_cuda: $(OBJS) $(CU_OBJ)
	$(NVCC) -o tensor_san_cuda $(OBJS) $(CU_OBJ) $(LDFLAGS)


# Rule to compile .c files into .o files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@


# Rule to compile CUDA files
%.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(CU_OBJ) tensor_san tensor_san_cuda

run: tensor_san
	./tensor_san

cuda: tensor_san_cuda
	./tensor_san_cuda

.PHONY: all clean
