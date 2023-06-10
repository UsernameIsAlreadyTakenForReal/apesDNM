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


# Set the figure size (optional)
plt.figure()

# Plot the heatmap using seaborn's heatmap function
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")

# Display the plot
plt.show()

# plt.figure()
# heat_map = sns.heatmap(df, linewidth=1, annot=True)
# plt.title("HeatMap using Seaborn Method")
# plt.show()


# dfSummary(df)


# numerical_cols = df.select_dtypes(include="number").columns
# df[numerical_cols].hist(bins=10)

# correlation_matrix = df[numerical_cols].corr()
# print(correlation_matrix)

# plt.plot(correlation_matrix)
# plt.show()


# print(path)
# print("file has shape " + str(df.shape))
# print("df.head \n" + str(df.head()))
# print(str(df.isnull().sum()))
