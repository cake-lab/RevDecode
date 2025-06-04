# Compiler
NVCC = nvcc

# Compute capability
CUDA_ARCH = -gencode=arch=compute_52,code=\"sm_52,compute_52\"

# Include directories
INCLUDES = -I./include \
           -I$(CUDA_HOME)/include \
           -I/home/tren/json/include

# Preprocessor macros
DEFINES = -D_DEBUG -D_CONSOLE

# Host compiler flags
XCOMPILER_FLAGS = -std=c++17 -Wall -O0 -g

# Define the host compiler
CCBIN = g++

# NVCC compile flags
NVCC_COMPILE_FLAGS = $(CUDA_ARCH) -G -g -std=c++17 $(INCLUDES) $(DEFINES) \
                     -Xcompiler "$(XCOMPILER_FLAGS)" \
                     -cudart static --use-local-env --machine 64 --compile -x cu \
                     -ccbin $(CCBIN)

# Linker flags
LINKER_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcuda

# NVCC link flags
NVCC_LINK_FLAGS = $(CUDA_ARCH) -G -g $(INCLUDES) $(DEFINES) \
                  -Xcompiler "$(XCOMPILER_FLAGS)" \
                  -cudart static --use-local-env --machine 64 \
                  -ccbin $(CCBIN)

# Directories
SRCDIR = src
OBJDIR = obj

# Targets
TARGETS = fine_grained_graph_traversal naive_graph_traversal parallel_estimation_graph_traversal

# Default target builds all kernels
all: $(TARGETS)

# Shared source files
SHARED_SRC = $(wildcard $(SRCDIR)/backtrack.cpp) \
             $(wildcard $(SRCDIR)/initialize_weight_matrix.cpp) \
             $(wildcard $(SRCDIR)/pre_forward_kernels.cu) \
             $(wildcard $(SRCDIR)/report.cpp) \
             $(wildcard $(SRCDIR)/update_ranking_scheme.cpp) \
             $(wildcard $(SRCDIR)/utils.cpp)

SHARED_OBJ = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SHARED_SRC:.cu=.cpp)) \
             $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SHARED_SRC:.cpp=.cu))

### Fine-Grained Kernel ###

FINE_SPECIFIC_SRC = $(SRCDIR)/fine_grained_approach_forward_kernel.cu \
                    $(SRCDIR)/fine_grained_approach_main.cpp

FINE_OBJ = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(FINE_SPECIFIC_SRC:.cu=.cpp)) \
           $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(FINE_SPECIFIC_SRC:.cpp=.cu))

FINE_TARGET = fine_grained_graph_traversal

$(FINE_TARGET): $(FINE_OBJ) $(SHARED_OBJ)
	@echo "Linking $(FINE_TARGET)..."
	$(NVCC) $(NVCC_LINK_FLAGS) -o $@ $^ $(LINKER_FLAGS)

### Naive Kernel ###

NAIVE_SPECIFIC_SRC = $(SRCDIR)/naive_approach_forward_kernel.cu \
                     $(SRCDIR)/naive_approach_main.cpp

NAIVE_OBJ = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(NAIVE_SPECIFIC_SRC:.cu=.cpp)) \
            $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(NAIVE_SPECIFIC_SRC:.cpp=.cu))

NAIVE_TARGET = naive_graph_traversal

$(NAIVE_TARGET): $(NAIVE_OBJ) $(SHARED_OBJ)
	@echo "Linking $(NAIVE_TARGET)..."
	$(NVCC) $(NVCC_LINK_FLAGS) -o $@ $^ $(LINKER_FLAGS)

### Parallel Estimation Kernel ###

PAR_EST_SPECIFIC_SRC = $(SRCDIR)/parallel_estimation_approach_forward_kernel.cu \
                       $(SRCDIR)/parallel_estimation_main.cpp

PAR_EST_OBJ = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(PAR_EST_SPECIFIC_SRC:.cu=.cpp)) \
              $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(PAR_EST_SPECIFIC_SRC:.cpp=.cu))

PAR_EST_TARGET = parallel_estimation_graph_traversal

$(PAR_EST_TARGET): $(PAR_EST_OBJ) $(SHARED_OBJ)
	@echo "Linking $(PAR_EST_TARGET)..."
	$(NVCC) $(NVCC_LINK_FLAGS) -o $@ $^ $(LINKER_FLAGS)

### Compile Rules ###

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

### Clean Up ###

clean:
	@rm -f $(OBJDIR)/*.o
	@rm -f $(FINE_TARGET) $(NAIVE_TARGET) $(PAR_EST_TARGET)

.PHONY: all clean
