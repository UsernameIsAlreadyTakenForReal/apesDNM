# This file:
# Contains functions meant to bring all approached file types to a 'pandas' object.

import arff
import math
import pathlib
import pandas as pd
import os
from scipy.io.arff import loadarff


def unarchive(path, delete_after_unarchiving=False):
    import patoolib

    new_path = path.replace(pathlib.Path(path).suffix, "")
    if os.path.exists(new_path) == False:
        os.mkdir(new_path)
    patoolib.extract_archive(path, outdir=new_path)

    if delete_after_unarchiving:
        os.remove(path)

    return new_path


def process_file_type(Logger, app_instance_metadata):
    if os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path):
        ## is file, so we either have to unzip it, open it or see if there are any other available
        info_message = "app_instance_metadata.dataset_metadata.dataset_path is file."
        Logger.info("process_file_type", info_message)
        if (
            pathlib.Path(app_instance_metadata.dataset_metadata.dataset_path).suffix
            == ".zip"
            or pathlib.Path(app_instance_metadata.dataset_metadata.dataset_path).suffix
            == ".rar"
        ):
            info_message = "app_instance_metadata.dataset_metadata.dataset_path is an archive. Unarchiving."
            Logger.info("process_file_type", info_message)
            new_path = unarchive(app_instance_metadata.dataset_metadata.dataset_path)
            app_instance_metadata.dataset_metadata.dataset_path = new_path
            info_message = f"New file path is {new_path}"
            Logger.info("process_file_type", info_message)

    if os.path.isdir(app_instance_metadata.dataset_metadata.dataset_path):
        ## is folder, so we have to crawl through it
        pass

    match pathlib.Path(app_instance_metadata.dataset_metadata.dataset_path).suffix:
        case ".csv":
            df = pd.read_csv(
                app_instance_metadata.dataset_metadata.dataset_path, header=None
            )

            # df = df.sample(frac=1).reset_index(drop=True)  ## shuffles the rows

            if app_instance_metadata.dataset_metadata.separate_train_and_test == False:
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
