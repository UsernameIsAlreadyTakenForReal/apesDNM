# This file:
# Contains functions meant to bring all approached file types to a 'pandas' object.

import arff
import math
import pathlib
import numpy as np
import pandas as pd
import torch
import zipfile
import os
from scipy.io.arff import loadarff


def unarchive(path, delete_after_unarchiving=False):
    with zipfile.ZipFile(path, "r") as zip_ref:
        new_path = path.replace(".zip", "")
        zip_ref.extractall(new_path)

    if delete_after_unarchiving:
        os.remove(path)

    return new_path


def process_file_type(path, separate_train_and_test=False):
    match pathlib.Path(path).suffix:
        case ".csv":
            df = pd.read_csv(path, header=None)
            # df = df.sample(frac=1).reset_index(drop=True)  ## shuffles the rows

            if separate_train_and_test == False:
                return df
            else:
                x, y = df.shape()
                df_train = df.head(math.floor(0.2 * x))
                df_test = df.tail(x - math.floor(0.2 * x))
                return df_train, df_test
            pass
        case ".arff":
            pass
        case ".txt":
            return "txt"
        case ".npz":
            return "npz"
        case other:
            return "Could not idenfity file type."
