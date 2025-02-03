#!/bin/bash
c++ -O3 -c -march=broadwell -Wall -std=gnu++11 -fPIC $(python3-config --includes) diffusionND.cpp -I/c/modular-boost -lquadmath -o diffusionND.o
c++ -O3 -c -march=broadwell -Wall -std=gnu++11 -fPIC $(python3-config --includes) libDiffusion.cpp -I/c/modular-boost -lgsl -lgslcblas -lquadmath -o libDiffusion.o -I"/home/jacob/Desktop/Code/pybind11/include/"

g++ -shared -o "libDiffusion.so" libDiffusion.o diffusionND.o -I/c/modular-boost -lquadmath -lgsl -lgslcblas -I"/home/jacob/Desktop/Code/pybind11/include/" $(python3-config --includes)
