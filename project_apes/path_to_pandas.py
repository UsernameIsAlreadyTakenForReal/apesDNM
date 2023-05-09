import arff
import pathlib
import numpy as np
import pandas as pd
import torch
import zipfile
import os


os.system("cls" if os.name == "nt" else "clear")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

example_path01 = "C:\\Users\\danieldum\\Desktop\\ekg datasets\\mitbih_one.zip"
example_path02 = "C:\\Users\\danieldum\\Desktop\\ekg datasets\\mitbih_two.zip"


def unarchive(path, delete_after_unarchiving):
    with zipfile.ZipFile(path, "r") as zip_ref:
        new_path = path.replace(".zip", "")
        zip_ref.extractall(new_path)

    if delete_after_unarchiving:
        os.remove(path)

    return new_path


def filetype_processing(path):
    match pathlib.Path(path).suffix:
        case ".csv":
            df = pd.read_csv(path)
            df = df.sample(frac=1).reset_index(drop=True)
            return df
        case ".arff":
            return "arff"
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
        # paths = os.listdir(path)
        paths = [path + "\\" + p for p in os.listdir(path)]
        print(paths)
    else:
        paths = path.split(";")

    print("all paths are", paths)

    if len(paths) == 1:
        full_path0 = paths[0]

        df = filetype_processing(full_path0)
        print(df.shape)

    elif len(paths) == 2:
        full_path0 = paths[0]
        full_path1 = paths[1]

        df1 = filetype_processing(full_path0)
        df2 = filetype_processing(full_path1)
        print(df1.shape)
        print(df2.shape)

    else:
        print("You have provided more than two files.")


path_to_pandas(example_path01)
print(" ----------------------------------- ")
path_to_pandas(example_path02)
print("end")
