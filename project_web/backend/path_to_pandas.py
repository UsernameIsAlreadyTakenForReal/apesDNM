import arff
import math
import pathlib
import numpy as np
import pandas as pd
import torch
import zipfile
import os
from scipy.io.arff import loadarff

os.system("cls" if os.name == "nt" else "clear")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

example_path01 = "C:\\Users\\danieldum\\Desktop\\ekg datasets\\mitbih_one.zip"
example_path02 = "C:\\Users\\danieldum\\Desktop\\ekg datasets\\mitbih_two.zip"

example_path01 = (
    r"C:\Users\danieldum\Desktop\ekg datasets\Dataset - ECG5000\ECG5000_TEST.arff"
)
print(example_path01)


def unarchive(path, delete_after_unarchiving):
    with zipfile.ZipFile(path, "r") as zip_ref:
        new_path = path.replace(".zip", "")
        zip_ref.extractall(new_path)

    if delete_after_unarchiving:
        os.remove(path)

    return new_path


def filetype_processing(path, separate_train_and_test):
    match pathlib.Path(path).suffix:
        case ".csv":
            df = pd.read_csv(path)
            df = df.sample(frac=1).reset_index(drop=True)  ## shuffles the rows

            if separate_train_and_test == False:
                return df, None
            else:
                x, y = df.shape()
                df_train = df.head(math.floor(0.2 * x))
                df_test = df.tail(x - math.floor(0.2 * x))
                return df_train, df_test

        case ".arff":
            raw_data = loadarff(path)
            df = pd.DataFrame(raw_data[0])

            x, y = df.shape

            data = arff.load(path)

            train_list = []
            train_class_list = []
            train_number_of_entries = 0

            for row in data:
                train_number_of_entries = train_number_of_entries + 1
                temp_row = []
                for i in range(0, y):
                    temp_row.append(row[i])
                train_class_list.append(int(row.target))
                train_list.append(temp_row)

            train_data = np.zeros((train_number_of_entries, y))
            train_labels = np.array(train_class_list)

            return df
        case ".txt":
            return "txt"
        case ".npz":
            return "npz"
        case other:
            return "Could not idenfity file type."


def path_to_pandas(path):
    # path must be provided
    if path == "":
        return

    # if archive is provided, unarchive and get all files from dir
    if ".zip" in path:
        path = unarchive(path, False)
        paths = [path + os.sep + p for p in os.listdir(path)]
        print(paths)
    else:
        paths = path.split(";")

    print("all paths are", paths)

    if len(paths) == 1:
        full_path0 = paths[0]

        df = filetype_processing(full_path0, True)
        print(df.shape)

    elif len(paths) == 2:
        df1 = filetype_processing(paths[0], False)
        df2 = filetype_processing(paths[1], False)
        print(df1.shape)
        print(df2.shape)

    else:
        print("You have provided more than two files.")


# path_to_pandas(example_path01)
