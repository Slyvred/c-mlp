#!/bin/bash

# Valgrind config
# gcc -g -O0 -Wall -o main src/*c -lm

# "Prod" config
# gcc -O3 -Wall -march=native -ffast-math -fopenmp -o main src/*.c -lm

# Fallback for prod if openmp isn't supported
g++ -O3 -Wall -march=native -ffast-math -o main src/*.cpp
