# Compiler
CC = g++
NVCC = nvcc
MPI_CC = mpiCC

# Compiler flags
CFLAGS = -Wall -Wextra -std=c++11
NVCCFLAGS = -std=c++11
MPICC_FLAGS = -Wall -Wextra -std=c++11

# CUDA flags and libraries
CUDAFLAGS = -arch=sm_75
CUDALIBS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda

# MPI flags and libraries
MPILIBS = -lmpi_cxx

# Directories
SRCDIR = ../norch/csrc
BUILDDIR = ../build
TARGET = ../norch/libtensor.so

# Files
SRCS := $(filter-out $(SRCDIR)/distributed.cpp, $(wildcard $(SRCDIR)/*.cpp))
CU_SRCS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.cu.o, $(CU_SRCS))
MPI_SRCS := $(SRCDIR)/distributed.cpp
MPI_OBJS := $(BUILDDIR)/distributed.o


# Rule to build the target
$(TARGET): $(OBJS) $(MPI_OBJS) $(CU_OBJS)
	$(NVCC) --shared -o $(TARGET) $(OBJS) $(MPI_OBJS) $(CU_OBJS) $(CUDALIBS) $(MPILIBS)

# Rule to compile C++ source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -fPIC -c $< -o $@ $(CUDALIBS)

# Rule to compile CUDA source files
$(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) -Xcompiler -fPIC -c $< -o $@

# Rule to compile distributed.cpp with mpiCC
$(BUILDDIR)/distributed.o: $(SRCDIR)/distributed.cpp
	$(MPI_CC) $(MPICC_FLAGS) -fPIC -c $< -o $@

# Clean rule
clean:
	rm -f $(BUILDDIR)/*.o $(TARGET)
