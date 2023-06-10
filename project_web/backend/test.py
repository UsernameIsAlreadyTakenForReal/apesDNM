import os
import numpy as np
import pandas as pd

from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns

from summarytools import dfSummary

path = r"D:\newAPES\Datasets\Dataset - ECG_Heartbeat\Dataset - ECG_Heartbeat\mitbih_test.csv"


df = pd.read_csv(path, header=None)

_, y = df.shape
column_index = y - 1

sns.boxplot(data=df.iloc[:, y - 1])
plt.xlabel("Column")
plt.ylabel("Value")
plt.title("Box Plot")
plt.show()
