# This file:
# Contains functions meant to bring all approached file types to a 'pandas' object.
#
# process_file_type():
# * if archive, then unarchive, which makes the path directory
# * if directory, then crawl through files
# * * * take all files that contain keywords of interest;
# * * * see if there are multiple files with the same name but different extensions. Import on priority;
# * if single file, then import it

import arff
import math
import pathlib
import pandas as pd

from scipy.io.arff import loadarff

import re
import os

supported_file_formats = [
    ".csv",
    ".arff",
    ".txt",
    ".npz",
    ".ts",
]  # also acts as priority
rough_filename_searches = ["train", "test", "validation"]


def unarchive(path, overwrite_existing=True, delete_after_unarchiving=False):
    import patoolib

    new_path = path.replace(pathlib.Path(path).suffix, "")
    if os.path.exists(new_path) == False:
        os.mkdir(new_path)
    patoolib.extract_archive(path, outdir=new_path)

    if delete_after_unarchiving:
        os.remove(path)

    return new_path


def process_file_type(Logger, app_instance_metadata):
    list_of_dataFrames = []
    list_of_dataFramesUtilityLabels = []
    map_of_dataFrames = []

    if (
        app_instance_metadata.dataset_metadata.dataset_path[-1] == "/"
        or app_instance_metadata.dataset_metadata.dataset_path[-1] == "\\"
    ):
        app_instance_metadata.dataset_metadata.dataset_path = (
            app_instance_metadata.dataset_metadata.dataset_path[:-1]
        )

    # archive
    if os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path) and (
        pathlib.Path(app_instance_metadata.dataset_metadata.dataset_path).suffix
        == ".zip"
        or pathlib.Path(app_instance_metadata.dataset_metadata.dataset_path).suffix
        == ".rar"
    ):
        info_message = "app_instance_metadata.dataset_metadata.dataset_path is an archive. Unarchiving."
        Logger.info("process_file_type", info_message)

        app_instance_metadata.dataset_metadata.dataset_path = unarchive(
            app_instance_metadata.dataset_metadata.dataset_path
        )

        info_message = (
            f"New file path is {app_instance_metadata.dataset_metadata.dataset_path}"
        )
        Logger.info("process_file_type", info_message)

    # folder
    if os.path.isdir(app_instance_metadata.dataset_metadata.dataset_path):
        possible_files_list = []
        for dirname, _, filenames in os.walk(
            app_instance_metadata.dataset_metadata.dataset_path
        ):
            for filename in filenames:
                if pathlib.Path(filename).suffix in supported_file_formats:
                    if rough_filename_filter(filename.lower()) == True:
                        possible_files_list.append(filename)

        files_to_load = filter_files_in_folder_list(possible_files_list)

        info_message = f"Possible files to load: {possible_files_list}"
        Logger.info("process_file_type", info_message)
        info_message = f"Files that will be loaded: {files_to_load}"
        Logger.info("process_file_type", info_message)

    # file
    if (
        os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path)
        or len(files_to_load) != 0
    ):
        if len(files_to_load) == 0:
            files_to_load = app_instance_metadata.dataset_metadata.dataset_path

        for file in files_to_load:
            list_of_dataFrames.append(
                process_singular_file_type(Logger, file, app_instance_metadata)
            )
            for string_match in rough_filename_searches:
                if string_match in file.lower():
                    list_of_dataFramesUtilityLabels.append(string_match)
                    break

        info_message = f"list_of_dataFrames: {list_of_dataFrames}"
        Logger.info("process_file_type", info_message)
        info_message = (
            f"list_of_dataFramesUtilityLabels: {list_of_dataFramesUtilityLabels}"
        )
        Logger.info("process_file_type", info_message)

    map_of_dataFrames = list_of_dataFrames.append(list_of_dataFramesUtilityLabels)
    return map_of_dataFrames


def process_singular_file_type(Logger, file, app_instance_metadata):
    info_message = f"Processing file {file}"
    Logger.info("process_singular_file_type", info_message)
    match pathlib.Path(file).suffix:
        case ".csv":
            df = pd.read_csv(file, header=None)

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
    pass


def rough_filename_filter(filename):
    for string_match in rough_filename_searches:
        result = re.search(string_match, filename)
        if result != None:
            return True
    return False


def filter_files_in_folder_list(filename_list):
    list_of_stems = [pathlib.Path(x).stem for x in filename_list]
    return_list = []

    # 'seen' will be uniques, 'dupes' will be extra
    seen = set()
    dupes = []
    for x in list_of_stems:
        if x in seen:
            dupes.append(x)
        else:
            seen.add(x)

    for filename in seen:
        for extension in supported_file_formats:
            if filename + extension in filename_list:
                return_list.append(filename + extension)
                break

    if len(return_list) != 0:
        return return_list
    else:
        return filename_list
