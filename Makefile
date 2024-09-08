# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -g
LDFLAGS = -lm

# Source files
SRCS = main.c data_loader.c nn/nn.c nn/activation.c

# Object files
OBJS = $(SRCS:.c=.o)

# Header files
HEADERS = data_loader.h nn/nn.h nn/activation.h nn/structs.h

# Target to build the executable
all: tensor_san

# Link the object files to create the executable
tensor_san: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

# Rule to compile .c files into .o files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) tensor_san

run: tensor_san
	./tensor_san

.PHONY: all clean
