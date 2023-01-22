# http://timeseriesclassification.com/description.php?Dataset=ECG5000
# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
import numpy as np
import pandas as pd
import torch
import arff
import matplotlib.pyplot as plt

file_path = "../../Dataset - ECG5000/ECG5000_TRAIN.arff"

data = arff.load(file_path)

test_list = []
number_of_entries = 0

for row in data:
    number_of_entries = number_of_entries + 1
    temp_row = []
    for i in range(0, 141):
        temp_row.append(row[i])
    test_list.append(temp_row)
    

train_data_as_numpy = np.zeros((number_of_entries, 141))
 
for i in range(0, number_of_entries):
    something_normal_for_once_thank_you = [float(el) for el in test_list[i]]
    as_array = np.array(something_normal_for_once_thank_you)
    # flip, because the beat seems to be reversed
    train_data_as_numpy[i] = np.flip(as_array)
    

plt.plot(train_data_as_numpy[:6].flatten())
plt.show()