import os
import numpy as np
import pandas as pd

from datetime import datetime

path = r"C:\Users\DANIEL~1\AppData\Local\Temp\tmp9bmp4_a6"

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        full_path = os.path.join(dirname, filename)
        print(full_path)

        if "csv" in filename:
            df = pd.read_csv(full_path, header=None)
            print("-------------------------------------------------------------------")
            print(df.shape)
            print("---------------------------------")
            print(df.head())
            print("---------------------------------")
            print(df.info())
            print("---------------------------------")
            print(df.describe())
            print("--------------------------------- missing data per columns")
            print(df.isnull().sum())
            print("-------------------------------------------------------------------")
