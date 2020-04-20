CUDA_INSTALL_PATH := /usr/local/cuda

NVCC = nvcc
CXX = g++

INCLUDES = -I. -I./include -I/usr/local/cuda/include

COMMONFLAGS += $(INCLUDES)
COMMONFLAGS += -g
CXXFLAGS += $(COMMONFLAGS)
CXXFLAGS += -Wall -std=c++11
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += -std=c++11

SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin

OBJS  = $(OBJDIR)/linAlgFcns.cu.o 
OBJS += $(OBJDIR)/examples_linAlg.cu.o 
OBJS += $(OBJDIR)/cpp_utils.cpp.o 
OBJS += $(OBJDIR)/driver.cu.o

TARGET = run
LINK = $(NVCC) $(OBJS) -o $(BINDIR)/$(TARGET)

.SUFFIXES: .c .cpp .cu .o

$(BINDIR)/$(TARGET): $(OBJS)
	$(LINK)

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf bin/*
	rm -rf obj/*.o
