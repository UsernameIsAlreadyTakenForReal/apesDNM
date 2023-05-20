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

files_path = "D:\Facultate\Dizertație\Dataset - ECG_Heartbeat"


for dirname, _, filenames in os.walk(files_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def add_gaussian_noise(signal):
    noise = np.random.normal(0, 0.5, 186)
    return signal + noise


time_dataset_manipulation_begin = datetime.now()
train_df = pd.read_csv(files_path + "/mitbih_train.csv", header=None)
test_df = pd.read_csv(files_path + "/mitbih_test.csv", header=None)