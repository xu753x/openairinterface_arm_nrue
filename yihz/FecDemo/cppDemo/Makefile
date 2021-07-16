
PREGCC=
TARGET=FFTest
GCC=$(PREGCC)g++


SYS_INC=/usr/local/include/dev2.0
INCLUDE_DIR := -I$(SYS_INC)
INCLUDE_DIR += -I$(SYS_INC)/9371base
INCLUDE_DIR += -I$(SYS_INC)/ecpriBase
INCLUDE_DIR += -I$(SYS_INC)/fr9371
INCLUDE_DIR += -I$(SYS_INC)/frecpri
INCLUDE_DIR += -I$(SYS_INC)/wi9371
INCLUDE_DIR += -I$(SYS_INC)/wiecpri

LIB_DIR    = -lpthread -lhugetlbfs
LIB_DIR1   = -lhugetlbfs
LIB_DIR2   = -ldev2.0

CURDIR:=$(shell pwd)
SRCDIR2 := $(CURDIR)
SRCS := $(wildcard *.cpp)
OBJS = $(SRCS:%.cpp=%.o)
CCFLAGS := -std=c++11 -g -O2 $(INCLUDE_DIR)


$(TARGET):$(OBJS)
	$(GCC) $(CCFLAGS) $(OBJS) -o $@ $(LIB_DIR) $(LIB_DIR2)

%.o: %.cpp
	$(GCC) $(INCLUDE_DIR) $(CCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJS) $(TARGET)
