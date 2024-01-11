CXX = g++
CXXFLAGS = -std=c++11
LDFLAGS = -lgsl -lgslcblas

all: main

main: src/main.cpp src/src_ref/BetaDistGsl.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o bin/$@ $(LDFLAGS)

clean:
	rm -f bin/main