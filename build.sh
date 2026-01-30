#!/bin/bash

# gcc -O3 -Wall -o main src/*c -lm

# gcc -O3 -Wall -march=native -ffast-math -fopenmp -o main src/*.c -lm

gcc -O3 -Wall -march=native -ffast-math -o main src/*.c src/mongoose/*c -lm
