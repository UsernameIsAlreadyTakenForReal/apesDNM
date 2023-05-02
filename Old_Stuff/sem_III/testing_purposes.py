import numpy as np
import pandas as pd
import torch
import arff
import matplotlib.pyplot as plt

from helpers_aiders_and_conveniencers import gradient_descent_functions

file_path = "../../Dataset - ECG5000/ECG5000_TRAIN.arff"

data = arff.load(file_path)

test_list = []
class_list = []
number_of_entries = 0

for row in data:
    number_of_entries = number_of_entries + 1
    temp_row = []
    for i in range(0, 141):
        temp_row.append(row[i])
    class_list.append(int(row.target))
    test_list.append(temp_row)
    

train_data_as_numpy = np.zeros((number_of_entries, 141))
labels_as_numpy = np.array(class_list)
 
for i in range(0, number_of_entries):
    something_normal_for_once_thank_you = [float(el) for el in test_list[i]]
    train_data_as_numpy[i] = np.array(something_normal_for_once_thank_you)
    
    
X = train_data_as_numpy.T
Y = labels_as_numpy.T - 1
iterations = 200
alpha = 0.1
number_of_input_neurons = train_data_as_numpy.shape[1]
number_of_neurons_in_first_layer = 250
number_of_outputs = 5


W1, b1, W2, b2 = gradient_descent_functions.init_params(number_of_input_neurons, 
                                                        number_of_neurons_in_first_layer, 
                                                        number_of_outputs)
last_good_W1, last_good_b1, last_good_W2, last_good_b2 = gradient_descent_functions.init_params(number_of_input_neurons, 
                                                                     number_of_neurons_in_first_layer, 
                                                                     number_of_outputs)
for i in range(iterations):
    print("Starting iteration {i}".format(i=i))
    Z1, A1, Z2, A2 = gradient_descent_functions.forward_prop(W1, b1, W2, b2, X)
    
    predictions = gradient_descent_functions.get_predictions(A2);
    accuracy = gradient_descent_functions.get_accuracy(predictions, Y);      
    
    dW1, db1, dW2, db2 = gradient_descent_functions.back_prop(Z1, A1, Z2, A2, W2, X, Y, number_of_outputs)
    W1, b1, W2, b2 = gradient_descent_functions.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    