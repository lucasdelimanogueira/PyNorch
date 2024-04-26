# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -Wall -Werror -fpic

# Source files
SRCS := ../src/tensor.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Target shared object file
TARGET := libtensor.so

# Default target
all: $(TARGET)

# Rule to build the shared object file
$(TARGET): $(OBJS)
	$(CXX) -shared -o $@ $^

# Rule to compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@   # Use -o $@ to specify the output file

# Clean rule
clean:
	$(RM) $(OBJS) $(TARGET)
