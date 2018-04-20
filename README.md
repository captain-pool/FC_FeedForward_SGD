# Fully Connected Feed Forward Neural Network

## Overview
Lightweight Python 3 Library for Feed Forward Neural Network with Stochastic Gradient Descent (SGD) Optimizer and Euclidean Error Function.
## Dependencies:
 * Numpy 1.14.x```pip install numpy```
## Usage:
 * Create Network
``` from FC_SGD import Network
    
     import numpy as np
    network=Network([3,5,2]) # Fully Connected Neural Network created with 3 layers, having 3 neurons, 5 neurons, and 2 neurons
```
 * Train Network

```
  network.SGD(np.array([([<input layer data 1>],[output layer data 1]), ([<input layer data 2>],[output layer data 2])]))
```
*  Predict result
``` 
 output=network.feedforward(np.array([<input layer data>]))
 print(output[<last layer neuron number 0-base indexed>]) #0th Neuron is at the top of the layer
```
