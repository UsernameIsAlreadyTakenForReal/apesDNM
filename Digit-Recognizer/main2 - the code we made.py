"""
Neural Network warm up
Following this tutorial: https://www.youtube.com/watch?v=w8yWXqWQYmU
Useful link with code and math ecuations: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook
Put these files (onli train.csv is used) in the same folder as this code file: https://drive.google.com/drive/u/1/folders/17MvIwg0yXSr3Qno3w3xwLL_ZnYcMlbqv
To be used as reference or tool in further development
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# # ---------------------------------------------------------------------------
# # --------------------------- Intro & Explanation ---------------------------
# # ---------------------------------------------------------------------------

data = pd.read_csv('./train.csv')

# This NN has 784 inputs, 10 hidden neurons and 10 output neurons
# data.head() # to see the data
# on this particular example (but it also applies in general, i think), data comes
# in the form of a matrix of size 42000x785. Pictures are 28x28 pixels, which means
# 784 total pixels. As such, each row represents one picture, with the entire data
# having 41999 pictures (one row is the top one, with column names). It is a good
# idea to transpose this data, because in our NN each input is a pixel.

# To use this and be impressed by it, run the whole code and then run line 150 from the console line
# The first parameter picks an example for the NN to interpret, so change that if you want

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)     # shuffle before splitting into dev and training sets

# to not overfit we define a chunk of data on which we do not train
# chunk of data on which we do not train (first 1k examples)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.    # this makes the data smaller, so computation is faster

# chunk of data on which we train
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# # ---------------------------------------------------------------------------
# # -------------------------------- Functions --------------------------------
# # ---------------------------------------------------------------------------

def init_params():
    W1 = np.random.rand(10, 784) - 0.5      # this generates random values between 0 and 1
    b1 = np.random.rand(10, 1) - 0.5        # 784 inputs, 10 neurons in the hidden layer
    W2 = np.random.rand(10, 10) - 0.5       # this generates random values between 0 and 1
    b2 = np.random.rand(10, 1) - 0.5        
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

def deriv_ReLU(Z):                  # don't be scared of this. This is the derivative of
    return Z > 0                    # ReLU, but think of how ReLU looks like and boom, you lookin' for this?

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

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

def gradient_descent(X, Y, iterations, alpha): # optimization. This is literally the whole thing
    W1, b1, W2, b2 = init_params()
    accuracy_over_time = []
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        predictions = get_predictions(A2);
        accuracy = get_accuracy(predictions, Y);
        accuracy_over_time.append(accuracy);
        
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            # print("Accuracy is ", get_accuracy(predictions, Y), " for prediction ", predictions)
            print("Accuracy: ", get_accuracy(predictions, Y))
            
    return accuracy_over_time, W1, b1, W2, b2
    
# # ---------------------------------------------------------------------------
# # -------------------------------- Train NN! --------------------------------
# # ---------------------------------------------------------------------------

number_of_iterations = 2000
alpha = 0.10
accuracy_over_time, W1, b1, W2, b2 = gradient_descent(X_train, Y_train, number_of_iterations, alpha)

plt.figure(1)
plt.plot(accuracy_over_time)
plt.show()

# W1 = np.random.rand(10, 784) - 0.5      # this generates random values between 0 and 1
# b1 = np.random.rand(10, 1) - 0.5        # 784 inputs, 10 neurons in the hidden layer
# W2 = np.random.rand(10, 10) - 0.5       # this generates random values between 0 and 1
# b2 = np.random.rand(10, 1) - 0.5   

# Z1 = W1.dot(X_train) + b1
# A1 = ReLU(Z1)
# Z2 = W2.dot(A1) + b2
# A2 = softmax(Z2)

# m = Y_train.size
# one_hot_Y = one_hot(Y_train)
# dZ2 = A2 - one_hot_Y
# dW2 = 1 / m * dZ2.dot(A1.T)
# db2 = 1 / m * np.sum(dZ2)
# dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
# dW1 = 1 / m * dZ1.dot(X_train.T)
# db1 = 1 / m * np.sum(dZ1)

# alpha = 0.10
# W1 = W1 - alpha * dW1
# b1 = b1 - alpha * db1
# W2 = W2 - alpha * dW2
# b2 = b2 - alpha * db2

# predictions = get_predictions(A2);
# accuracy = get_accuracy(predictions, Y_train);

# # ---------------------------------------------------------------------------
# # ---------------------------- Use NN to predict ----------------------------
# # ---------------------------------------------------------------------------

# def make_predictions(X, W1, b1, W2, b2):
#     _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
#     predictions = get_predictions(A2)
#     return predictions

# def test_prediction(index, W1, b1, W2, b2):
#     current_image = X_train[:, index, None]
#     prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
#     label = Y_train[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)
    
#     current_image = current_image.reshape((28, 28)) * 255
#     plt.gray()
#     plt.imshow(current_image, interpolation='nearest')
#     plt.show()

# fancier to run this from the console line
# test_prediction(0, W1, b1, W2, b2)
# test_prediction(1, W1, b1, W2, b2)
# test_prediction(2, W1, b1, W2, b2)
# test_prediction(3, W1, b1, W2, b2)