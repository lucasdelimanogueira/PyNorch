# Compiler
CC = g++
NVCC = nvcc

# Compiler flags
CFLAGS = -Wall -Wextra -std=c++11
NVCCFLAGS = -std=c++11

# Directories
SRCDIR = ../csrc
BUILDDIR = ../build
TARGET = ../build/libtensor.so

# Files
SRCS = $(wildcard $(SRCDIR)/*.cpp)
CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.cu.o, $(CU_SRCS))

# CUDA flags and libraries
CUDAFLAGS = -arch=sm_75
CUDALIBS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda

# Rule to build the target
$(TARGET): $(OBJS) $(CU_OBJS)
	$(NVCC) --shared -o $(TARGET) $(OBJS) $(CU_OBJS) $(CUDALIBS)

# Rule to compile C++ source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -fPIC -c $< -o $@ $(CUDALIBS)

# Rule to compile CUDA source files
$(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) -Xcompiler -fPIC -c $< -o $@

# Clean rule
clean:
	rm -f $(BUILDDIR)/*.o $(TARGET)
