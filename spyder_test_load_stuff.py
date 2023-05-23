import arff
import math
import pathlib
import numpy as np
import pandas as pd
import torch
import zipfile
import os
from scipy.io.arff import loadarff


# os.system("cls" if os.name == "nt" else "clear")


def unarchive(path, delete_after_unarchiving=False):
    with zipfile.ZipFile(path, "r") as zip_ref:
        new_path = path.replace(".zip", "")
        zip_ref.extractall(new_path)

    if delete_after_unarchiving:
        os.remove(path)

    return new_path


def process_file(path, separate_train_and_test):
    match pathlib.Path(path).suffix:
        case ".csv":
            df = pd.read_csv(path)
            df = df.sample(frac=1).reset_index(drop=True)
    
