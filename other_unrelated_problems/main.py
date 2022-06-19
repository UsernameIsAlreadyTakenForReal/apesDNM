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

# activation function (Rectified Linear Unit)
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
# different activation function (Softmax)
class Activation_Softmas:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
        # Activation_Softmas - some explaining:
        # keep in mind we are considering having batches here, parallel runs of the model
        #   which means every 'input' is a matrix with multiple rows, each row signifying a run from the batch
        # exp_values = np.exp(inputs - np.max()) because exp() can go very big very quickly
        #   so we substract the biggest value of the vector from all elements of the vector
        # axis=1 means summing the matrix's rows
        # keepdims=True means results are put into a vertical vector

# class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # done like this so we don't transpose every fucking fucking cufking time
                                                                    # also * 0.10 to have the values scaled to [-1, 1]
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Small program here
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

classes = 3
samples = 100
X, y = spiral_data(samples, classes)

# FIGURE 1 - see that beautiful data
plt.figure(1)
plt.scatter(X[:,0], X[:,1])
plt.show()
plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()

layer1 = Layer_Dense(2, 5) # 2 inputs because we are taking a 2d plot as input, duuh
layer2 = Layer_Dense(5, classes) # 3 final outputs because we have 3 classes to sort

activation1 = Activation_ReLU()
activation2 = Activation_Softmas() 

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print("Layer 1 output:\n", layer1.output)
print("Layer 1 after activation:\n", activation1.output)

print("Layer 2 output:\n", layer2.output)
print("Layer 2 after activation (first 5):\n", activation2.output[:5])
print("At this point, what we see here is just the probability / distribution of each class (3 classes in a row, each with a 33% chance). This makes sense given the fact that the classes were randomly generated in this example")

# # -------------------------------- section 4 --------------------------------
# # loss is error, calculated in ln() (Part 7 & 8 in tutorial)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) # batch loss
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss): 
    def forward(self, y_prediction, y_true): # y_true = target training values
        samples = len(y_prediction)
        y_pred_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1: # passed scalar class values
            correct_confidences = y_pred_clipped[range(samples), y_true] # https://youtu.be/levekYbxauw?t=665
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) # https://youtu.be/levekYbxauw?t=764
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)

# # -------------------------------- section 5 --------------------------------
# # This section treats adapting the NN for a very simple, vertical data
# # This section uses quasi-random alteration of weights and biases, and only
# # works because the data_set is VERY simplistic

import numpy as np 
import matplotlib.pyplot as plt
# nnfs is Neural Networks From Scratch's python library. Useful for learning
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

X, y = vertical_data(samples=100, classes=3)

# FIGURE 2
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmas()

loss_function = Loss_CategoricalCrossEntropy()

# take the initial values as point of reference
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

number_of_iterations = 100000
losses = np.zeros(number_of_iterations) # save all losses just to plot them at the end

for iteration in range(number_of_iterations):
    
    # modify weights and biases in relation to the previous values, kind of like
    # you do with genetic algorithms
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)
    
    # run the NN
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output, y)
    losses[iteration] = loss
    
    # np.argmax returns the indices of the maximum values along the axis. Here, 
    # maximum values for each row. Accuracy goes up when loss goes down, duh
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)
    
    if loss < lowest_loss: # if we found better stuff, it becomes the new reference
        print('New set of weights found, iteration:', iteration, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
        
    else: # if not, we revert the changes made in this iteration
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

plt.figure(3)
plt.plot(range(number_of_iterations), losses)
plt.show()

# # -------------------------------- section 6 --------------------------------
# # So quasi-random alteration of the NN does not work for complex problems. 
# # Quasi-random alteration alters weights and biases by treating all of them the same,
# # but each neuron (with its weights and bias) has a different impact on the NN
# # A good way to find this impact of each neuron are derivatives
# # The following example is not optimal. Calculating derivatives like this for 
# # each w & b would be a rather brute-force approach
# # As such, let's welcome partial derivatives after we get the damn book

# # x^2 example
# def f(x):
#     return 2*x**2

# p2_delta = 0.0001

# x1 = 1
# x2 = x1 + p2_delta  # on a continuous function (i.e. with infinite points), 
#                     # x2 would be the 'next' point. We can't define such a point
#                     # but we can define a delta small enough to get us a good approximation

# y1 = f(x1)
# y2 = f(x2)

# approximate_derivative = (y2 - y1) / (x2 - x1)
# b = y2 - approximate_derivative * x2    # y = m*x + b

# print(approximate_derivative)