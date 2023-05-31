# This file:
# Contains functions meant to bring all approached file types to a 'pandas' object.
#
# process_file_type():
# * if archive, then unarchive, which makes the path directory
# * if directory, then crawl through files
# * * * take all files that contain keywords of interest;
# * * * see if there are multiple files with the same name but different extensions. Import on priority;
# * if single file, then import it
#
# File // files import logic:
# single file
# * - EITHER all contents go to dataFrame_train (if % NOT defined);
# * - OF split contents between dataFrame_train and dataFrame_test by user % (if % defined);
# multiple files
# * - if the user offered delimiters, search files by those keywords
# * - if the user didn't input delimiters or they failed, take files named 'test', 'train', (maybe 'val'), use them as prescribed;
# * - if neither, all contents are collapsed into one single dataFrame and the logic of the 'single file' section is applied.

# TODO: recursive if folder (after zip?) contains folders or zip...

import arff
import math
import pathlib
import pandas as pd
import numpy as np

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
    files_to_load = []

    # checks
    if (
        app_instance_metadata.dataset_metadata.dataset_path[-1] == "/"
        or app_instance_metadata.dataset_metadata.dataset_path[-1] == "\\"
    ):
        app_instance_metadata.dataset_metadata.dataset_path = (
            app_instance_metadata.dataset_metadata.dataset_path[:-1]
        )

    # # archive
    if os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path) and (
        pathlib.Path(app_instance_metadata.dataset_metadata.dataset_path).suffix
        == ".zip"
        or pathlib.Path(app_instance_metadata.dataset_metadata.dataset_path).suffix
        == ".rar"
    ):
        return_code, return_message = treat_archive(Logger, app_instance_metadata)
        if return_code != 0:
            return (return_code, return_message)
        else:
            Logger.info("process_file_type", return_message)

    # folder
    if os.path.isdir(app_instance_metadata.dataset_metadata.dataset_path):
        return_code, return_message, files_to_load = treat_folder(
            Logger, app_instance_metadata
        )
        if return_code != 0:
            return (return_code, return_message)
        else:
            Logger.info("process_file_type", return_message)

    # file
    if (
        os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path)
        or len(files_to_load) != 0
    ):
        (
            return_code,
            return_message,
            files_to_load,
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        ) = treat_file(
            Logger,
            app_instance_metadata,
            files_to_load,
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        )
        if return_code != 0:
            return (return_code, return_message)
        else:
            Logger.info("process_file_type", return_message)

    map_of_dataFrames = (list_of_dataFrames, list_of_dataFramesUtilityLabels)
    return 0, "process_file_type exited successfully", map_of_dataFrames


def treat_file(
    Logger,
    app_instance_metadata,
    files_to_load,
    list_of_dataFrames,
    list_of_dataFramesUtilityLabels,
):
    try:
        if len(files_to_load) == 0:
            files_to_load = app_instance_metadata.dataset_metadata.dataset_path

        for file in files_to_load:
            return_code, return_message, temp_df = process_singular_file_type(
                Logger, file, app_instance_metadata
            )
            if return_code != 0:
                return return_code, return_message
            else:
                list_of_dataFrames.append(temp_df)
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

        return (
            0,
            "apes_dataset_handler.treat_file exited successfully",
            files_to_load,
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        )
    except:
        return (
            1,
            "apes_dataset_handler.treat_file exited with error",
            files_to_load,
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        )


def treat_folder(Logger, app_instance_metadata):
    try:
        no_of_files_in_folder = 0
        no_of_files_in_dir_and_subdirs = []
        possible_files_list = []

        # check if there are subdirectories
        for dirname, _, filenames in os.walk(
            app_instance_metadata.dataset_metadata.dataset_path
        ):
            no_of_files_in_dir_and_subdirs.append(len(filenames))

        # if subdirs and there is at least one file outside of the main dir
        if (
            len(no_of_files_in_dir_and_subdirs) != 1
            and sum(no_of_files_in_dir_and_subdirs) != no_of_files_in_dir_and_subdirs[0]
        ):
            info_message = f"There seem to be {len(no_of_files_in_dir_and_subdirs) - 1} subdirectories in this directory. Number of files for each: {no_of_files_in_dir_and_subdirs}"
            Logger.info("process_file_type", info_message)
            return 1, "Subdirectories present. Case not yet handled", []
        else:
            info_message = f"There seem to be {no_of_files_in_dir_and_subdirs[0]} files in this directory. No subdirectories"
            Logger.info("process_file_type", info_message)

            # search for keynames given by the user
            if len(app_instance_metadata.dataset_metadata.file_keyword_names) != 0:
                for dirname, _, filenames in os.walk(
                    app_instance_metadata.dataset_metadata.dataset_path
                ):
                    for filename in filenames:
                        if (
                            pathlib.Path(filename).suffix
                            in app_instance_metadata.dataset_metadata.file_keyword_names
                        ):
                            if (
                                rough_filename_filter(
                                    filename.lower(),
                                    app_instance_metadata.dataset_metadata.file_keyword_names,
                                )
                                == True
                            ):
                                possible_files_list.append(filename)

            # search for keynames 'test', 'train', 'val' if user didn't prescribe keywords or they failed
            if (
                len(app_instance_metadata.dataset_metadata.file_keyword_names) == 0
                or len(possible_files_list) == 0
            ):
                for dirname, _, filenames in os.walk(
                    app_instance_metadata.dataset_metadata.dataset_path
                ):
                    for filename in filenames:
                        if pathlib.Path(filename).suffix in supported_file_formats:
                            if rough_filename_filter(filename.lower()) == True:
                                possible_files_list.append(filename)

            # if that doesn't work either, just load all files and mash them all files into the same dataFrame (hoping it works)
            else:
                for dirname, _, filenames in os.walk(
                    app_instance_metadata.dataset_metadata.dataset_path
                ):
                    for filename in filenames:
                        possible_files_list.append(filename)
                pass

        files_to_load = filter_files_in_folder_list(possible_files_list)

        info_message = f"Possible files to load: {possible_files_list}"
        Logger.info("process_file_type", info_message)
        info_message = f"Files that will be loaded: {files_to_load}"
        Logger.info("process_file_type", info_message)

        return 0, "apes_data_handler.treat_folder exited successfully", files_to_load
    except:
        return 1, "apes_data_handler.treat_folder exited with error", []


def treat_archive(Logger, app_instance_metadata):
    try:
        info_message = "app_instance_metadata.dataset_metadata.dataset_path is an archive. Unarchiving."
        Logger.info("process_file_type", info_message)

        app_instance_metadata.dataset_metadata.dataset_path = unarchive(
            app_instance_metadata.dataset_metadata.dataset_path
        )

        info_message = (
            f"New file path is {app_instance_metadata.dataset_metadata.dataset_path}"
        )
        Logger.info("process_file_type", info_message)
        return 0, "apes_data_handler.treat_archive exited successfully"
    except:
        return 1, "apes_data_handler.treat_archive exited with error"


def process_singular_file_type(Logger, file, app_instance_metadata):
    if os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path) == False:
        file = app_instance_metadata.dataset_metadata.dataset_path + "/" + file
    info_message = f"Processing file {file}"
    Logger.info("process_singular_file_type", info_message)

    match pathlib.Path(file).suffix:
        case ".csv":
            df = []
            loaded_with_header = False
            try:
                info_message = f"Trying to load file {file} without header"
                Logger.info("process_singular_file_type", info_message)
                df = pd.read_csv(file, header=None, dtype=np.float64)
            except:
                info_message = f"Trying to load file {file} with header"
                Logger.info("process_singular_file_type", info_message)
                df = pd.read_csv(file, dtype=np.float64)
                loaded_with_header = True
            finally:
                if loaded_with_header == True:
                    info_message = f"Loaded file {file} with header"
                else:
                    info_message = f"Loaded file {file} without header"
                Logger.info("process_singular_file_type", info_message)
                return 0, "process_singular_file_type exited successfully", df

            # df = df.sample(frac=1).reset_index(drop=True)  ## shuffles the row
            # if app_instance_metadata.dataset_metadata.separate_train_and_test == False:
            #     return df
            # else:
            #     x, y = df.shape()
            #     df_train = df.head(math.floor(0.2 * x))
            #     df_test = df.tail(x - math.floor(0.2 * x))
            #     return df_train, df_test
        case ".arff":
            df = arff.load(file)
            return 0, "process_singular_file_type exited successfully", df
        case ".txt":
            return "txt"
        case ".npz":
            return "npz"
        case other:
            return 1, "Could not idenfity file type.", df


def rough_filename_filter(filename, keywords=rough_filename_searches):
    for string_match in keywords:
        result = re.search(string_match.lower(), filename)
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
