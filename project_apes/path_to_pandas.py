import arff
import pathlib
import numpy as np
import pandas as pd
import torch
import zipfile
import os

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
    suffix = pathlib.Path(path).suffix

    match suffix:
        case ".csv":
            return "csv"
        case ".arff":
            return "arff"
        case ".txt":
            return "txt"
        case ".npz":
            return "npz"
        case other:
            return "Could not idenfity file type."


def path_to_pandas(path):
    print("hello from path_to_pandas()")

    # path must be provided
    if path == "":
        return

    # if archive is provided, unarchive and get all files from dir
    if ".zip" in path:
        path = unarchive(path, False)
        paths = os.listdir(path)
    else:
        paths = path.split(";")

    print(paths)

    if len(paths) == 1:
        print("One file: ", paths[0])

        pandas1 = filetype_processing(paths[0])

        print(pandas1)

    elif len(paths) == 2:
        print("Two files: ", paths[0], ", ", paths[1])

        pandas1 = filetype_processing(paths[0])
        pandas2 = filetype_processing(paths[1])

        print(pandas1, "and", pandas2)

    else:
        print("You have provided more than two files.")


path_to_pandas(example_path01)
print(" ----------------------------------- ")
path_to_pandas(example_path02)
