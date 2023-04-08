# http://timeseriesclassification.com/description.php?Dataset=ECG5000
# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
import numpy as np
import arff
import matplotlib.pyplot as plt

from helpers_aiders_and_conveniencers import gradient_descent_functions

## THIS IS NOT FOR ANOMALY DETECTION, ONLY FOR CLASS-PREDICTION

###############################################################################
## --------------------------- Getting the datasets ---------------------------
###############################################################################

## Train data (500 entries)
file_path = "../../Dataset - ECG5000/ECG5000_TRAIN.arff"
data = arff.load(file_path)

train_list = []
train_class_list = []
train_number_of_entries = 0

for row in data:
    train_number_of_entries = train_number_of_entries + 1
    temp_row = []
    for i in range(0, 140):
        temp_row.append(row[i])
    train_class_list.append(int(row.target))
    train_list.append(temp_row)
    

train_data = np.zeros((train_number_of_entries, 140))
train_labels = np.array(train_class_list)
 
for i in range(0, train_number_of_entries):
    something_normal_for_once_thank_you = [float(el) for el in train_list[i]]
    train_data[i] = np.array(something_normal_for_once_thank_you)

    
## Test data (4500 entries)
file_path = "../../Dataset - ECG5000/ECG5000_TEST.arff"
data = arff.load(file_path)

test_list = []
test_class_list = []
test_number_of_entries = 0

for row in data:
    test_number_of_entries = test_number_of_entries + 1
    temp_row = []
    for i in range(0, 140):
        temp_row.append(row[i])
    test_class_list.append(int(row.target))
    test_list.append(temp_row)
    

test_data = np.zeros((test_number_of_entries, 140))
test_labels = np.array(test_class_list)
 
for i in range(0, test_number_of_entries):
    something_normal_for_once_thank_you = [float(el) for el in test_list[i]]
    test_data[i] = np.array(something_normal_for_once_thank_you)
    
    
###############################################################################
## --------------------------- Training the dataset ---------------------------
###############################################################################
X = train_data.T
Y = train_labels.T - 1                   # classes in dataset are in array [1, 2, 3, 4, 5]
iterations = 50
alpha = 0.1
number_of_input_neurons = train_data.shape[1]
number_of_neurons_in_first_layer = 250
number_of_outputs = 5
# number_of_outputs = len(np.unique(Y))     # this should be default. No. of output classes should be 
                                            # the same as no. of classes represented in the label array.

last_good_W1, last_good_b1, last_good_W2, last_good_b2, last_good_accuracy, \
accuracy_over_time, W1, b1, W2, b2 = gradient_descent_functions.gradient_descent(X, Y, 
                                                                                 iterations, 
                                                                                 alpha, 
                                                                                 number_of_input_neurons, 
                                                                                 number_of_neurons_in_first_layer,
                                                                                 number_of_outputs)
plt.figure(1)
plt.plot(accuracy_over_time)
plt.show()


###############################################################################
## ---------------------------- Testing the dataset ---------------------------
###############################################################################

X_test = test_data.T
Y_test = test_labels.T - 1

Z1_test, A1_test, Z2_test, A2_test = gradient_descent_functions.forward_prop(last_good_W1, 
                                                                         last_good_b1, 
                                                                         last_good_W2, 
                                                                         last_good_b2, X_test)
test_predictions = gradient_descent_functions.get_predictions(A2_test);
test_accuracy = gradient_descent_functions.get_accuracy(test_predictions, Y_test)

print(test_accuracy)
