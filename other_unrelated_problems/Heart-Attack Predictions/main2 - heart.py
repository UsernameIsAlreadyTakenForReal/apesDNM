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
X_dev = X_dev / 564.

# chunk of data on which we train
data_train = data[50:m].T
Y_train = data_train[9]
X_train = data_train[0:9]
X_train = X_train / 564.
X_train = X_train.T
Y_train = Y_train.T
_,m_train = X_train.shape

number_of_input_neurons = 9
number_of_neurons_in_first_layer = 40
number_of_outputs = 2

# # ---------------------------------------------------------------------------
# # --------------------------------- Classes ---------------------------------
# # ---------------------------------------------------------------------------

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmas:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # done like this so we don't transpose every fucking fucking cufking time
                                                                    # also * 0.10 to have the values scaled to [-1, 1]
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) # batch loss
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss): 
    def forward(self, y_prediction, y_true): # y_true = target training values
        no_of_samples = len(y_prediction)
        y_pred_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1: # passed scalar class values
            correct_confidences = y_pred_clipped[range(no_of_samples), y_true.astype(int)]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    

# # ---------------------------------------------------------------------------
# # -------------------------------- Train NN! --------------------------------
# # ---------------------------------------------------------------------------

dense1 = Layer_Dense(number_of_input_neurons, number_of_neurons_in_first_layer)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(number_of_neurons_in_first_layer, number_of_outputs)
activation2 = Activation_Softmas()

loss_function = Loss_CategoricalCrossEntropy()

# take the initial values as point of reference
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

number_of_iterations = 20000
losses = np.zeros(number_of_iterations) # save all losses just to plot them at the end
accuracies = []
accuracy_threshold = 0.7

for iteration in range(number_of_iterations):
    
    # modify weights and biases in relation to the previous values, kind of like
    # you do with genetic algorithms
    dense1.weights += 0.05 * np.random.randn(number_of_input_neurons, number_of_neurons_in_first_layer)
    dense1.biases += 0.05 * np.random.randn(1, number_of_neurons_in_first_layer)
    dense2.weights += 0.05 * np.random.randn(number_of_neurons_in_first_layer, number_of_outputs)
    dense2.biases += 0.05 * np.random.randn(1, number_of_outputs)
    
    # run the NN
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output, Y_train)
    losses[iteration] = loss
    
    # np.argmax returns the indices of the maximum values along the axis. Here, 
    # maximum values for each row. Accuracy goes up when loss goes down, duh
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == Y_train)
    accuracies.append(accuracy)
    
    if loss < lowest_loss: # if we found better stuff, it becomes the new reference
        # print('New set of weights found, iteration:', iteration, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
        if accuracy > accuracy_threshold:
            break
        
    else: # if not, we revert the changes made in this iteration
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

plt.figure(3)
# plt.plot(range(number_of_iterations), losses)
plt.plot(accuracies)
plt.show()