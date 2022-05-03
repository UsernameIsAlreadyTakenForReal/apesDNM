"""
Neural Network warm up
Following this tutorial: https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=1
To be used as reference or tool in further development
"""

# # -------------------------------- section 1 --------------------------------
# # without numpy
# # for now, 3 output Neurons:
# inputs = [1, 2, 3, 2.5]

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# layer_outputs = []

# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for weight, n_input in zip(neuron_weights, inputs):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
    
# print(layer_outputs)

# # -------------------------------- section 2 --------------------------------
# # same thig, with numpy
# import numpy as np

# inputs = [[1, 2, 3, 2.5],
#           [2, 5, -1, 2],
#           [-1.5, 2.7, 3.3, -0.8]]

# weights = [[0.2, 0.8, -0.5, 1.0],
#             [0.5, -0.91, 0.26, -0.5],
#             [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# layer1_output = np.dot(inputs, np.array(weights).T) + biases
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

# print(layer2_output)

# # -------------------------------- section 3 --------------------------------
# # same thig, with objects

import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(0)

# generate spiral data. Get Milky Way'd
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmas:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #done like this so we don't transpose every fucking fucking cufking time
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Program itself here bih

# this is testing the Layer_Dense class
# X = [[1, 2, 3, 2.5],
#      [2, 5, -1, 2],
#      [-1.5, 2.7, 3.3, -0.8]]

# layer1 = Layer_Dense(4, 5) # has 5 neurons
# layer2 = Layer_Dense(5, 2) # so this one also needs to have 5 neurons to take from. Outputs 2 neurons
# layer1.forward(X)
# #print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)

X, y = spiral_data(100, 3)

# see that beautiful data
plt.scatter(X[:,0], X[:,1])
plt.show()
plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()

layer1 = Layer_Dense(2, 5) # 2 inputs because we are taking a 2d plot as input
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(layer1.output)
print(activation1.output)


# softmax
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, 1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs) # e^x pt fiecare
# fiecare vector din layer_outputs sub forma e^x / suma vectorului de e^x
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) 

print(norm_values)
