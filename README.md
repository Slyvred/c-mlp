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

---
## Implementation Example: Decimal to Binary

The provided main.c contains a model trained to convert decimal numbers (0 to 15) into their 4-bit binary representation.

**Data Normalization**: To prevent neuron saturation and ensure efficient learning, the input decimal values are normalized to a range of $[0, 1]$ by dividing them by the maximum value (15.0).

**Model Configuration**:
- **Input**: 1 neuron (normalized decimal).
- **Hidden Layer 1**: 4 neurons.
- **Hidden Layer 2**: 16 neurons.
- **Output Layer**: 4 neurons (binary bits).
