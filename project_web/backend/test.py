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
print(df.head(5))


plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel("column 0")
plt.ylabel("column 1")
plt.show()
