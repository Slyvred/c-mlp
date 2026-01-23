# C-MLP  
This repository provides a minimalist implementation of a Multi-Layer Perceptron (MLP) written in C. It is designed to demonstrate the core mechanics of neural networks, including forward propagation and backpropagation, without the overhead of external libraries.

---
## Overview
The goal of this project is to provide a transparent look at how neural networks function at a low level. By using only the C standard library, the code highlights the mathematical operations required for training and inference.

### Features
- **Customizable Architecture**: Support for multiple hidden layers with varying numbers of neurons.
- **Activation Functions**: Built-in Sigmoid activation and its derivative.
- **Stochastic Training**: Implementation of basic gradient descent.
- **Minimal Dependencies**: Requires only stdio.h, stdlib.h, time.h, and math.h.
---
## Mathematics
The network uses the Sigmoid function as the activation for all neurons, defined as: 
$$S(x) = \frac{1}{1 + e^{-x}}$$

To facilitate learning through backpropagation, the derivative of the Sigmoid is utilized during the weight update phase: 
$$S'(x) = S(x) \cdot (1 - S(x))$$

## References
- https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/
  - For data structures, formulas etc
- Materials from previous courses
  - Theory, perceptron+mlp pseudocode
- https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
  - Backprop

---
## Implementation Example: Decimal to Binary

The provided main.c contains a model trained to convert decimal numbers (0 to 15) into their 4-bit binary representation.

**Data Normalization**: To prevent neuron saturation and ensure efficient learning, the input decimal values are normalized to a range of $[0, 1]$ by dividing them by the maximum value (15.0).

**Model Configuration**:
- **Input**: 1 neuron (normalized decimal).
- **Hidden Layer 1**: 4 neurons.
- **Hidden Layer 2**: 16 neurons.
- **Output Layer**: 4 neurons (binary bits).

### Example Output
```
./main 250000 0.5

Layer 0: Neurons: 4 | Parameters: 1
Layer 1: Neurons: 16 | Parameters: 4
Layer 2: Neurons: 4 | Parameters: 16
Total number of parameters: 132

 --- Training model ---
Epoch 0...
Epoch 25000...
Epoch 50000...
Epoch 75000...
Epoch 100000...
Epoch 125000...
Epoch 150000...
Epoch 175000...
Epoch 200000...
Epoch 225000...
--- End ---

--- Results ---
In:  0 | Out: [0 0 0 0] | Expected: [0 0 0 0]
In:  1 | Out: [0 0 0 1] | Expected: [0 0 0 1]
In:  2 | Out: [0 0 1 0] | Expected: [0 0 1 0]
In:  3 | Out: [0 0 1 1] | Expected: [0 0 1 1]
In:  4 | Out: [0 1 0 0] | Expected: [0 1 0 0]
In:  5 | Out: [0 1 0 1] | Expected: [0 1 0 1]
In:  6 | Out: [0 1 1 0] | Expected: [0 1 1 0]
In:  7 | Out: [0 1 1 1] | Expected: [0 1 1 1]
In:  8 | Out: [1 0 0 0] | Expected: [1 0 0 0]
In:  9 | Out: [1 0 0 1] | Expected: [1 0 0 1]
In: 10 | Out: [1 0 1 0] | Expected: [1 0 1 0]
In: 11 | Out: [1 0 1 1] | Expected: [1 0 1 1]
In: 12 | Out: [1 1 0 0] | Expected: [1 1 0 0]
In: 13 | Out: [1 1 0 1] | Expected: [1 1 0 1]
In: 14 | Out: [1 1 1 0] | Expected: [1 1 1 0]
In: 15 | Out: [1 1 1 1] | Expected: [1 1 1 1]
```
