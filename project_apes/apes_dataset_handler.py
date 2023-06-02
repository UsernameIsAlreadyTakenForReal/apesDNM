# This file:
# Contains functions meant to bring all approached file types to 'pandas' objects. Returns a list of dataFrames and a list for their
# attributed utility (i.e. train, test, validation)
#
# handle_dataset_from_path():
# * if archive, then unarchive, which makes the path directory
# * if directory, then crawl through files
# * * * take all files that contain keywords of interest;
# * * * see if there are multiple files with the same name but different extensions. Import on priority;
# * if single file, then import it
#
# File // files import logic:
# 1. single file
# * 1.1 - EITHER all contents go to dataFrame_train (if % NOT defined);
# * 1.2 - OF split contents between dataFrame_train and dataFrame_test by user % (if % defined);
# 2. multiple files
# * 2.1 - if the user offered delimiters, search files by those keywords
# * 2.2 - if the user didn't input delimiters or they failed, take files named 'test', 'train', (maybe 'val'), use them as prescribed;
# * 2.3 - if neither, all contents are collapsed into one single dataFrame and the logic of the 'single file' section is applied.

# TODO: recursive if folder (after zip?) contains folders or zip...
# TODO: split dataset for [multiple, percetages]
# TODO: files_to_load must not contain dummy/dud files
# NON-TODO: if there are 2 "train" files, maybe user has different use-cases. Couldn't think of a scenario in which this happens.

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
rough_filename_searches = ["train", "test", "validation", "run"]


def handle_dataset_from_path(Logger, app_instance_metadata):
    list_of_dataFrames = []
    list_of_dataFramesUtilityLabels = []
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
        ) = treat_files(
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

    return (
        0,
        "process_file_type exited successfully",
        list_of_dataFrames,
        list_of_dataFramesUtilityLabels,
    )


def unarchive(path, overwrite_existing=True, delete_after_unarchiving=False):
    import patoolib

    new_path = path.replace(pathlib.Path(path).suffix, "")
    if os.path.exists(new_path) == False:
        os.mkdir(new_path)
    patoolib.extract_archive(path, outdir=new_path)

    if delete_after_unarchiving:
        os.remove(path)

    return new_path


def treat_files(
    Logger,
    app_instance_metadata,
    files_to_load,
    list_of_dataFrames,
    list_of_dataFramesUtilityLabels,
):
    flag_enter = False
    default_label = (
        app_instance_metadata.shared_definitions.default_file_to_dataset_label
    )

    if len(app_instance_metadata.dataset_metadata.file_keyword_names) != 0:
        rough_filename_searches = (
            app_instance_metadata.dataset_metadata.file_keyword_names
        )

    try:
        if len(files_to_load) == 0:
            files_to_load = app_instance_metadata.dataset_metadata.dataset_path
            flag_enter = True

        # Load files and label them. Also counts as case 2.1 and 2.2. Also 2.3, as the default label is "train" (see in apes_static_definitions)
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
                    else:
                        list_of_dataFramesUtilityLabels.append(default_label)
                        break

        # Exit if the no. of dataFrames does not match no. of labels
        if len(list_of_dataFrames) != len(list_of_dataFramesUtilityLabels):
            return (
                1,
                "apes_dataset_handler.treat_file exited with error: number of dataFrames do not match number of utility labels",
                files_to_load,
                list_of_dataFrames,
                list_of_dataFramesUtilityLabels,
            )

        # If list_of_dataFramesUtilityLabels contains only "train", make one single "train" dataFrame. We will pass this to case 2.3 // 1.1 or 1.2
        # Else if list_of_dataFramesUtilityLabels contains only one of each, go ahead
        # Else unify_dataFrames (make only one "train", only one "test" etc)
        verifier = [
            0 if x == default_label else 1 for x in list_of_dataFramesUtilityLabels
        ]
        if sum(verifier) == 0:
            temp_list_of_dataFrames = []
            temp_list_of_dataFrames.append(pd.concat(list_of_dataFrames))
            list_of_dataFramesUtilityLabels = [default_label]
            list_of_dataFrames = temp_list_of_dataFrames
            flag_enter = True
        elif are_labels_unique(list_of_dataFramesUtilityLabels) == True:
            flag_enter = False
        else:
            (
                return_code,
                return_message,
                list_of_dataFrames,
                list_of_dataFramesUtilityLabels,
            ) = unify_dataFrames(
                Logger, list_of_dataFrames, list_of_dataFramesUtilityLabels
            )
            if return_code != 0:
                return return_code, return_message

        # If single file or if all were put together
        if flag_enter == True:
            # Case 1.1 -- do not split
            if app_instance_metadata.dataset_metadata.separate_train_and_test == False:
                list_of_dataFrames = list_of_dataFrames
                list_of_dataFramesUtilityLabels = [default_label]
            # Case 1.2 -- split
            else:
                (
                    return_code,
                    return_message,
                    list_of_dataFrames,
                    list_of_dataFramesUtilityLabels,
                ) = split_dataFrames(
                    Logger,
                    app_instance_metadata,
                    list_of_dataFramesUtilityLabels,
                    list_of_dataFramesUtilityLabels,
                )

        info_message = f"list_of_dataFrames: {list_of_dataFrames}"
        Logger.info("treat_files", info_message)
        info_message = (
            f"list_of_dataFramesUtilityLabels: {list_of_dataFramesUtilityLabels}"
        )
        Logger.info("treat_files", info_message)

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
                        if pathlib.Path(filename).suffix in supported_file_formats:
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

                if app_instance_metadata.dataset_metadata.shuffle_rows == True:
                    info_message = "Shuffling rows"
                    Logger.info("process_singular_file_type", info_message)
                    df = df.sample(frac=1).reset_index(drop=True)
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
    # TODO: treat .txt files that do not contain tables or something of the sort
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


def unify_dataFrames(Logger, list_of_dataFrames, list_of_dataFramesUtilityLabels):
    temp_list_of_dataFrames = []

    # 'seen' will be uniques, 'dupes' will be extra
    seen = []
    dupes = []
    for x in list_of_dataFramesUtilityLabels:
        if x in seen:
            dupes.append(x)
        else:
            seen.append(x)

    temp_list = []
    for element in seen:
        # get all dataFrames of one label in a single list
        for i in range(0, len(list_of_dataFramesUtilityLabels)):
            if element == list_of_dataFramesUtilityLabels[i]:
                temp_list.append(list_of_dataFrames[i])

        # mash those dataFrames into one single dataFrame
        first_dataFrame_shape = temp_list[0].shape
        for i in range(1, len(temp_list)):
            if temp_list[i].shape != first_dataFrame_shape:
                info_message = f"dataFrame.shape at index {i} does not match {first_dataFrame_shape}"
                Logger.info("unify_dataFrames", info_message)
        temp_list_of_dataFrames.append(pd.concat(temp_list))

    return (
        0,
        "apes_data_handler.unify_dataFrames exited successfully",
        temp_list_of_dataFrames,
        seen,
    )


def are_labels_unique(list_of_dataFramesUtilityLabels):
    seen = []
    dupes = []
    for x in list_of_dataFramesUtilityLabels:
        if x in seen:
            dupes.append(x)
        else:
            seen.append(x)

    if seen == list_of_dataFramesUtilityLabels:
        return True
    return False


def split_dataFrames(
    Logger, app_instance_metadata, list_of_dataFrames, list_of_dataFramesUtilityLabels
):
    # [%]
    if len(app_instance_metadata.dataset_metadata.percentage_of_split) == 1:
        if len(list_of_dataFrames) != 1:
            info_message = "Only one split % value was given, but there are multiple dataFrames. If this error appears, there's something extremely wrong with Python."
            Logger.info("split_dataFrames", info_message)

        rows = list_of_dataFrames[0].shape[0]
        number_of_rows_to_train = round(
            (app_instance_metadata.dataset_metadata.percentage_of_split[0] / 100) * rows
        )

        dataFrame_train = list_of_dataFrames[0].iloc[:number_of_rows_to_train, :]
        dataFrame_test = list_of_dataFrames[0].iloc[number_of_rows_to_train:, :]

        list_of_dataFrames = []
        list_of_dataFrames.append(dataFrame_train)
        list_of_dataFramesUtilityLabels.append("train")
        list_of_dataFrames.append(dataFrame_test)
        list_of_dataFramesUtilityLabels.append("test")

        return (
            0,
            "split_dataFrames exited successfully",
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        )
    # [%, %]
    elif len(app_instance_metadata.dataset_metadata.percentage_of_split) == 2:
        if len(list_of_dataFrames) != 1:
            info_message = "Two split % value was given, but there are multiple dataFrames. If this error appears, there's something extremely wrong with Python."
            Logger.info("split_dataFrames", info_message)

        return (
            0,
            "split_dataFrames exited successfully. Let's not do this right now.",
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        )
    return (
        0,
        "split_dataFrames exited successfully",
        list_of_dataFrames,
        list_of_dataFramesUtilityLabels,
    )
