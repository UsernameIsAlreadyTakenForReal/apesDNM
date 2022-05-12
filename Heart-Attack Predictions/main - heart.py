"""
Neural Network warm up as project for uni
Data taken from: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
To be used as reference or tool in further development
"""

"""
Data:    
    [0] (Age)       - Age of the patient
    [1] (Sex)       - Sex of the patient (0 / 1)
    [2] (CP)        - Chest Pain (1 - typical angina, 2 - atypical angina, 3 - non-anginal pain, 4 - asymptomatic)
    [3] (Trtbps)    - resting blood pressure (in mm Hg)
    [4] (Chol)      - cholestoral in mg/dl fetched via BMI sensor
    [5] (Fbs)       - fasting blood sugar (> 120 mg/dl -- 1 = true, 0 = false)
    [6] (Restecg)   - resting electrocardiographic results (0 - normal, 1 - ST-T wave abnormality, 2 - probable or definite left ventricular hypertrophy)
    [7] (Thalachh)  - maximum heart rate achieved
    [8] (Exng)      - exercise induced angina (1 - yes; 0 - no)
    
    [9] (Oldpeak)   - ? excluded from this example
    [10] (Slp)      - ? excluded from this example
    [11] (Caa)      - ? excluded from this example
    [12] (Thall)    - ? excluded from this example
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./heart.csv')

data = np.array(data)
data_short = np.delete(data, [9, 10, 11, 12], 1)
data = data_short
m, n = data.shape
np.random.shuffle(data)

# to not overfit we define a chunk of data on which we do not train
# chunk of data on which we do not train (first 1k examples)
data_dev = data[0:50].T
Y_dev = data_dev[9]
X_dev = data_dev[0:9]
X_dev = X_dev

# chunk of data on which we train
data_train = data[50:m].T
Y_train = data_train[9]
X_train = data_train[0:9]
X_train = X_train
_,m_train = X_train.shape

number_of_input_neurons = 9
number_of_neurons_in_first_layer = 40
number_of_outputs = 2

def init_params(): # passing variables to np.random.rand() seems to create weird looking random numbers lol (i.e. they are treated as e-01)
    W1 = np.random.rand(100, 9) - 0.5        # this generates random values between 0 and 1
    b1 = np.random.rand(100, 1) - 0.5        # 784 inputs, 10 neurons in the hidden layer
    W2 = np.random.rand(2, 100) - 0.5        # this generates random values between 0 and 1
    b2 = np.random.rand(2, 1) - 0.5        
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
    one_hot_Y = np.zeros((Y.size, 2))
    for i in range(Y.size):
        if Y[i] == 1:
            one_hot_Y[i][0] = 1
        else:
            one_hot_Y[i][1] = 1
    return one_hot_Y.T


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m_back = Y.size
    label = one_hot(Y)
    dZ2 = A2 - label
    dW2 = 1 / m_back * dZ2.dot(A1.T)
    db2 = 1 / m_back * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m_back * dZ1.dot(X.T)
    db1 = 1 / m_back * np.sum(dZ1)
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
        
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            # print("Accuracy is ", get_accuracy(predictions, Y), " for prediction ", predictions)
            print("Accuracy: ", get_accuracy(predictions, Y))
            
    return accuracy_over_time, W1, b1, W2, b2
    
# # ---------------------------------------------------------------------------
# # -------------------------------- Train NN! --------------------------------
# # ---------------------------------------------------------------------------

number_of_iterations = 500
alpha = 0.1
accuracy_over_time, W1, b1, W2, b2 = gradient_descent(X_train, Y_train, number_of_iterations, alpha)

plt.figure(1)
plt.plot(accuracy_over_time)
plt.show()

# W1 = np.random.rand(100, 9) - 0.5        # this generates random values between 0 and 1
# b1 = np.random.rand(100, 1) - 0.5        # 784 inputs, 10 neurons in the hidden layer
# W2 = np.random.rand(2, 100) - 0.5        # this generates random values between 0 and 1
# b2 = np.random.rand(2, 1) - 0.5    

# Z1 = W1.dot(X_train) + b1
# A1 = ReLU(Z1)
# Z2 = W2.dot(A1) + b2
# A2 = softmax(Z2)  

# m_back = Y_train.size
# label = one_hot(Y_train)
# dZ2 = A2 - label
# dW2 = 1 / m_back * dZ2.dot(A1.T)
# db2 = 1 / m_back * np.sum(dZ2)
# dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
# dW1 = 1 / m_back * dZ1.dot(X_train.T)
# db1 = 1 / m_back * np.sum(dZ1)