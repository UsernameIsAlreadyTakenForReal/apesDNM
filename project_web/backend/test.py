import os
import numpy as np
import pandas as pd

from datetime import datetime


class Dataset_EDA:
    def __init__(self, Logger, dataset_path):
        self.Logger = Logger
        self.dataset_path = dataset_path

    def perform_eda(self):
        self.Logger.info(self, "EDA started for " + self.dataset_path)

        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                full_path = os.path.join(dirname, filename)
                print(full_path)

                if "csv" in filename:
                    df = pd.read_csv(full_path, header=None)
                    print("-----------------------------------------------------------")
                    print("shape: ")
                    print(df.shape)
                    print("---------------------------------")
                    print("head: ")
                    print(df.head())
                    print("---------------------------------")
                    print("info: ")
                    print(df.info())
                    print("---------------------------------")
                    print("describe: ")
                    print(df.describe())
                    print("---------------------------------")
                    print("missing data per columns")
                    print(df.isnull().sum())
                    print("-----------------------------------------------------------")

                    print(filename + " has shape " + df.shape)


path = r"C:\Users\DANIEL~1\AppData\Local\Temp\tmp9bmp4_a6"
