import PIL
import numpy as np
import pandas as pd
import random
import gc
import sys
from datetime import datetime
import os, psutil
import time
from matplotlib import pyplot as plt

from PIL import Image

def init_params(number_of_input_neurons, number_of_neurons_in_first_layer, number_of_outputs): # passing variables to np.random.rand() seems to create weird looking random numbers lol (i.e. they are treated as e-01)
    
    print("Starting init with inputs={inputs}, first_layer={layer}, outputs={out}".format(inputs=number_of_input_neurons,
                                                                                                      layer=number_of_neurons_in_first_layer,
                                                                                                      out=number_of_outputs))
    time_temp = datetime.now() 
    W1 = np.random.rand(number_of_neurons_in_first_layer, number_of_input_neurons).astype(np.float32) - 0.5
    time_alloc_W1 = datetime.now() - time_temp 
    print("time_alloc_W1 = {time_alloc_W1}s ({time_alloc_W1_ms}ms)".format(time_alloc_W1=time_alloc_W1.seconds,
                                                                           time_alloc_W1_ms=time_alloc_W1.microseconds))     
    
    time_temp = datetime.now() 
    b1 = np.random.rand(number_of_neurons_in_first_layer, 1).astype(np.float32) - 0.5        
    time_alloc_b1 = datetime.now() - time_temp 
    print("time_alloc_b1 = {time_alloc_b1}s ({time_alloc_b1_ms}ms)".format(time_alloc_b1=time_alloc_b1.seconds,
                                                                        time_alloc_b1_ms=time_alloc_b1.microseconds)) 
    
    time_temp = datetime.now()
    W2 = np.random.rand(number_of_outputs, number_of_neurons_in_first_layer).astype(np.float32) - 0.5      
    time_alloc_W2 = datetime.now() - time_temp 
    print("time_alloc_W2 = {time_alloc_W2}s ({time_alloc_W2_ms}ms)".format(time_alloc_W2=time_alloc_W2.seconds,
                                                                           time_alloc_W2_ms=time_alloc_W2.microseconds))     
    
    time_temp = datetime.now()
    b2 = np.random.rand(number_of_outputs, 1).astype(np.float32) - 0.5           
    time_alloc_b2 = datetime.now() - time_temp  
    print("time_alloc_b2 = {time_alloc_b2} ({time_alloc_b2_ms}ms)".format(time_alloc_b2=time_alloc_b2.seconds,
                                                                          time_alloc_b2_ms=time_alloc_b2.microseconds))  
    
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z)) 
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W2, X, Y, number_of_outputs):
    m_back = Y.size
    label = one_hot(Y, number_of_outputs)
    dZ2 = A2 - label
    dW2 = 1 / m_back * dZ2.dot(A1.T)
    db2 = 1 / m_back * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m_back * dZ1.dot(X.T)
    db1 = 1 / m_back * np.sum(dZ1)
    return dW1, db1, dW2, db2

def deriv_ReLU(Z):                  # don't be scared of this. This is the derivative of
    return Z > 0                    # ReLU, but think of how ReLU looks like and boom, you lookin' for this?

def one_hot(Y, number_of_outputs):
    one_hot_Y = np.zeros((Y.size, number_of_outputs))
    for i in range(Y.size):
        for j in range(0, number_of_outputs):
            if Y[i] == j:
                one_hot_Y[i][j] = 1
    return one_hot_Y.T

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha, number_of_input_neurons, number_of_neurons_in_first_layer, number_of_outputs): # optimization. This is literally the whole thing
    
    print("Starting gradient descent with inputs={inputs}, first_layer={layer}, outputs={out}".format(inputs=number_of_input_neurons,
                                                                                                      layer=number_of_neurons_in_first_layer,
                                                                                                      out=number_of_outputs))
    print("Hope you didn't forget to transpose the input and label matrices!")
    W1, b1, W2, b2 = init_params(number_of_input_neurons, number_of_neurons_in_first_layer, number_of_outputs)
    accuracy_over_time = []
    last_good_accuracy = 0
    last_good_W1, last_good_b1, last_good_W2, last_good_b2 = init_params(number_of_input_neurons, number_of_neurons_in_first_layer, number_of_outputs)
    
    for i in range(iterations):
        print("Starting iteration {i}".format(i=i))
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        predictions = get_predictions(A2);
        accuracy = get_accuracy(predictions, Y);
        if accuracy > last_good_accuracy:
            last_good_accuracy = accuracy
            last_good_W1 = W1
            last_good_b1 = b1
            last_good_W2 = W2
            last_good_b2 = b2
        accuracy_over_time.append(accuracy);        
        
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y, number_of_outputs)
        # print("For iteration " + str(i))
        # print(dW1)
        # print(db1)
        # print(dW2)
        # print(db2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # if i % 10 == 0:
        print("Iteration: ", i)
        predictions = get_predictions(A2)
        # print("Accuracy is ", get_accuracy(predictions, Y), " for prediction ", predictions)
        print("Accuracy: ", get_accuracy(predictions, Y))
            
    return last_good_W1, last_good_b1, last_good_W2, last_good_b2, last_good_accuracy, accuracy_over_time, W1, b1, W2, b2