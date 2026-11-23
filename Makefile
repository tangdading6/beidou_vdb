CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -ffp-contract=off
SRCS := src/main.cpp src/gltf.cpp src/scanfill.cpp src/export.cpp
HEADERS := src/scanfill.hpp src/gltf.hpp src/export.hpp

.PHONY: all clean

all: build/scanfill

build/scanfill: $(SRCS) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

clean:
	rm -rf build
