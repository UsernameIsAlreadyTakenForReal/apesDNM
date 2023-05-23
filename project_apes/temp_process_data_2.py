import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings

warnings.filterwarnings("ignore")

from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input
from keras.models import Model

# from keras.layers.normalization import BatchNormalization
from tensorflow import keras
from keras.layers import BatchNormalization
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import resample
from datetime import datetime

for dirname, _, filenames in os.walk("../../Dataset - ECG_Heartbeat"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def add_gaussian_noise(signal):
    noise = np.random.normal(0, 0.5, 186)
    return signal + noise


time_dataset_manipulation_begin = datetime.now()
train_df = pd.read_csv("../../Dataset - ECG_Heartbeat/mitbih_train.csv", header=None)
test_df = pd.read_csv("../../Dataset - ECG_Heartbeat/mitbih_test.csv", header=None)

train_df[187] = train_df[187].astype(int)
equilibre = train_df[187].value_counts()
print(equilibre)

target_train = train_df[187]
target_test = test_df[187]
y_train = to_categorical(target_train)
y_test = to_categorical(target_test)


X_train = train_df.iloc[:, :186].values
X_test = test_df.iloc[:, :186].values
for i in range(len(X_train)):
    X_train[i, :186] = add_gaussian_noise(X_train[i, :186])
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)

temp = (X_train.shape[1], 1)
from type_ekg.apes_solution_ekg_2_train import solution_ekg_2
from helpers_aiders_and_conveniencers.Logger import Logger
from apes_static_definitions import Shared_Definitions

Logger = Logger()
shared_definitions = Shared_Definitions()

test_solution = solution_ekg_2(shared_definitions, Logger)

test_solution.create_model(temp)
test_solution.train(X_train, y_train, X_test, y_test, 1)
test_solution.save_model()
