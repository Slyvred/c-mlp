# C-MLP  
This repository provides a minimalist implementation of a Multi-Layer Perceptron (MLP) written in C. It is designed to demonstrate the core mechanics of neural networks, including forward propagation and backpropagation, without the overhead of external libraries.

---
## Overview
The goal of this project is to provide a transparent look at how neural networks function at a low level. By using only the C standard library, the code highlights the mathematical operations required for training and inference.

### Features
- **Customizable Architecture**: Support for multiple hidden layers with varying numbers of neurons.
- **Activation Functions**:
  - Sigmoid
  - LeakyRelu
  - Linear
  - Softmax
- **Stochastic Training**: Implementation of basic gradient descent.
- **Minimal Dependencies**: Requires only stdio.h, stdlib.h, time.h, math.h, arpa/inet.h (for endian swapping only).
---
## Mathematics
The network uses the LeakyRelu function as the activation for all hidden layers, defined as:

$$f(x) = \max(\epsilon x, x)$$

With its derivative being:

$$f'(x) = \max(\epsilon, 1)$$
With $\epsilon$ being a small constant, e.g. $0.01$

To facilitate learning through backpropagation, the derivative of the function is utilized during the weight update phase.

As we are doing classification in the example provided in [main.c](src/main.c), the activation function of the output layer is the Softmax function, defined as:  

$$softmax(z)_{i} = \frac{e^{z_{i}}}{\sum_{j=1}^{N} e^{z_{j}}}$$

## References
- https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/
  - For data structures, formulas etc
- Materials from previous courses
  - Theory, perceptron+mlp pseudocode
- https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
  - Backprop

---
## Implementation Example: MNIST Dataset

The provided [main.c](src/main.c) contains a model trained to classify written digits from 0 to 9 to their correct label.

**Data Normalization**: To prevent neuron saturation and ensure efficient learning, the value of each pixel from the input image is normalized to a range in $[0, 1]$ by dividing it by the maximum value of a pixel (255).

**One Hot Encoding**: Since we want to predict the class we won't output the label directly. We instead output a probability distribution (with the softmax function) of the input belonging to a label. So each label is converted in a probability vector like so:

| Label | One Hot Encoded                |
|-------|--------------------------------|
| 0     | [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| 1     | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] |
| 2     | [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] |
| 3     | [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] |
| ...   | ...                            |

**Model Configuration**:
- **Input layer**: 256 neurons -> relu
- **First hidden layer**: 128 neurons -> relu
- **Second hidden layer**: 64 neurons -> relu
- **Output Layer**: 10 neurons (= 10 classes) -> softmax.

### Example Output

```
./main 256 0.01

Layer 0: Neurons: 256 | Parameters: 784
Layer 1: Neurons: 128 | Parameters: 256
Layer 2: Neurons: 64 | Parameters: 128
Layer 3: Neurons: 10 | Parameters: 64
Total number of parameters: 242762

 --- Training model ---
Epoch: 0...
Epoch: 25...
Epoch: 50...
Epoch: 75...
Epoch: 100...
Epoch: 125...
Epoch: 150...
Epoch: 175...
Epoch: 200...
Epoch: 225...
Epoch: 250...
--- End ---

--- Results ---
Output: 7 | Actual: 7
Output: 6 | Actual: 6
Output: 3 | Actual: 3
Output: 4 | Actual: 4
Output: 2 | Actual: 2
Output: 3 | Actual: 3
Output: 6 | Actual: 6
[...]
Output: 6 | Actual: 6
Output: 1 | Actual: 7
Output: 3 | Actual: 3
Output: 0 | Actual: 0
Output: 6 | Actual: 6
Output: 1 | Actual: 1
```
