import arff
import os, psutil
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pylab import rcParams
from datetime import datetime
from sklearn import metrics

import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
from apes_static_definitions import Shared_Definitions

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

all_train_data = np.c_[train_data, train_labels]

df = pd.DataFrame(all_train_data)

# df = df._append(df_2)

CLASS_NORMAL = 1
class_names = ["Normal", "R on T", "PVC", "SP", "UB"]

new_columns = list(df.columns)
new_columns[-1] = "target"
df.columns = new_columns


## Let’s get all normal heartbeats and drop the target (class) column:
normal_df = df[df.target == int(CLASS_NORMAL)].drop(labels="target", axis=1)

## We’ll merge all other classes and mark them as anomalies:
anomaly_df = df[df.target != int(CLASS_NORMAL)].drop(labels="target", axis=1)

## We’ll split the normal examples into train, validation and test sets:
train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=RANDOM_SEED)
val_df, test_df = train_test_split(val_df, test_size=0.33, random_state=RANDOM_SEED)


from type_ekg.apes_solution_ekg_1_train import solution_ekg_1
from helpers_aiders_and_conveniencers.Logger import Logger

Logger = Logger()
shared_definitions = Shared_Definitions()

test_solution = solution_ekg_1(shared_definitions, Logger)

train_dataset, seq_len, n_features = test_solution.create_dataset(train_df)
val_dataset, _, _ = test_solution.create_dataset(val_df)
test_normal_dataset, _, _ = test_solution.create_dataset(test_df)
test_anomaly_dataset, _, _ = test_solution.create_dataset(anomaly_df)

# seq_len, n_features = test_solution.create_dataset()
test_solution.create_model(seq_len, n_features)

test_solution.train(train_dataset, val_dataset, 3)
test_solution.save_model()

test_solution.test(train_dataset, test_normal_dataset, test_anomaly_dataset)
