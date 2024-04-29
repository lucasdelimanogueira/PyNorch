# Compiler
CXX := g++
NVCC := $(shell command -v nvcc 2>/dev/null)

# Compiler flags
CXXFLAGS := -Wall -Werror -fpic
NVCCFLAGS := -Xcompiler -fPIC

# CUDA flags
CUDAFLAGS := -arch=sm_75

# Source files
SRCS := ../csrc/tensor.cpp
CU_SRCS := $(wildcard ../csrc/*.cu)

# Object files
OBJS := $(SRCS:.cpp=.o)

# CUDA object files (only if nvcc is available)
ifeq ($(NVCC),)
CU_OBJS :=
else
CU_OBJS := $(CU_SRCS:%.cu=%.cu.o)
endif

# Target shared object file
TARGET := libtensor.so

# Default target
all: $(TARGET)

# Rule to build the shared object file
$(TARGET): $(OBJS) $(CU_OBJS)
	$(CXX) -shared -o $@ $^

# Rule to compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile each CUDA source file into an object file (only if nvcc is available)
%.cu.o: %.cu
ifneq ($(NVCC),)
	$(NVCC) $(NVCCFLAGS) $(CUDAFLAGS) -c $< -o $@
else
	@echo "No CUDA compiler found, skipping CUDA compilation of $<"
endif

# Clean rule
clean:
	$(RM) $(OBJS) $(CU_OBJS) $(TARGET)
