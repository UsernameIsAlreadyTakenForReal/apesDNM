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
# TODO: fuck knows how, but make sure the label/target is in the last column

import sys

import arff
import math
import pathlib
import pandas as pd
import numpy as np

from scipy.io.arff import loadarff
from helpers_aiders_and_conveniencers.misc_functions import (
    how_many_different_values_in_list,
)

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
    files_to_load = []
    list_of_dataFrames = []
    list_of_dataFramesUtilityLabels = []

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
            Logger.info("handle_dataset_from_path", return_message)

    # folder
    if os.path.isdir(app_instance_metadata.dataset_metadata.dataset_path):
        return_code, return_message, files_to_load = treat_folder(
            Logger, app_instance_metadata
        )
        if return_code != 0:
            return (return_code, return_message)
        else:
            Logger.info("handle_dataset_from_path", return_message)

    # file
    if (
        os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path)
        or len(files_to_load) != 0
    ):
        (
            return_code,
            return_message,
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        ) = treat_files(
            Logger,
            app_instance_metadata,
            files_to_load,
        )
        if return_code != 0:
            return return_code, return_message, [], []
        else:
            Logger.info("handle_dataset_from_path", return_message)

    return (
        0,
        "handle_dataset_from_path exited successfully",
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
):
    list_of_dataFrames = []
    list_of_dataFramesUtilityLabels = []
    flag_enter = False
    default_label = (
        app_instance_metadata.shared_definitions.default_file_to_dataset_label
    )

    rough_filename_searches = (
        app_instance_metadata.shared_definitions.rough_filename_searches
    )

    if len(app_instance_metadata.dataset_metadata.file_keyword_names) != 0:
        info_message = "Switched file-to-dataFrames labels from default to user-defined"
        Logger.info("apes_dataset_handler.treat_files", info_message)
        rough_filename_searches = (
            app_instance_metadata.dataset_metadata.file_keyword_names
        )

    # try:
    if True:
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
                    matching_file = file.lower()
                    if string_match in matching_file:
                        list_of_dataFramesUtilityLabels.append(string_match)
                        break
                    elif string_match == rough_filename_searches[-1]:
                        list_of_dataFramesUtilityLabels.append(default_label)
                        break

        # Exit if the no. of dataFrames does not match no. of labels
        if len(list_of_dataFrames) != len(list_of_dataFramesUtilityLabels):
            return (
                1,
                "apes_dataset_handler.treat_file exited with error: number of dataFrames do not match number of utility labels",
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
            info_message = "list_of_dataFramesUtilityLabels contains only 'train'. Collapsing into a single dataFrame"
            Logger.info("apes_dataset_handler.treat_files", info_message)
            temp_list_of_dataFrames = []
            temp_list_of_dataFrames.append(pd.concat(list_of_dataFrames))
            list_of_dataFramesUtilityLabels = [default_label]
            list_of_dataFrames = temp_list_of_dataFrames
            flag_enter = True
        elif are_labels_unique(list_of_dataFramesUtilityLabels) == True:
            info_message = "list_of_dataFramesUtilityLabels contains only unique labels"
            Logger.info("apes_dataset_handler.treat_files", info_message)
            flag_enter = False
        else:
            info_message = "list_of_dataFramesUtilityLabels contains duplicates. Collapsing duplicates"
            Logger.info("apes_dataset_handler.treat_files", info_message)
            (
                return_code,
                return_message,
                list_of_dataFrames,
                list_of_dataFramesUtilityLabels,
            ) = unify_dataFrames(
                Logger, list_of_dataFrames, list_of_dataFramesUtilityLabels
            )
            if return_code != 0:
                return return_code, return_message, [], []

        # If single file or if all were put together
        if flag_enter == True:
            # Case 1.1 -- do not split
            if app_instance_metadata.dataset_metadata.separate_train_and_test == False:
                info_message = "One file or all files put together, and user doesn't want them split"
                Logger.info("apes_dataset_handler.treat_files", info_message)
                list_of_dataFrames = list_of_dataFrames
                list_of_dataFramesUtilityLabels = [default_label]
            # Case 1.2 -- split
            else:
                info_message = (
                    "One file or all files put together, and user wants them split"
                )
                Logger.info("apes_dataset_handler.treat_files", info_message)
                (
                    return_code,
                    return_message,
                    list_of_dataFrames,
                    list_of_dataFramesUtilityLabels,
                ) = split_dataFrames(
                    Logger,
                    app_instance_metadata,
                    list_of_dataFrames,
                    list_of_dataFramesUtilityLabels,
                )
                if return_code != 0:
                    return return_code, return_message, [], []

        if app_instance_metadata.display_dataFrames == True:
            info_message = f"list_of_dataFrames: {list_of_dataFrames}"
            Logger.info("apes_dataset_handler.treat_files", info_message)
        info_message = (
            f"list_of_dataFramesUtilityLabels: {list_of_dataFramesUtilityLabels}"
        )
        Logger.info("apes_dataset_handler.treat_files", info_message)

        return (
            0,
            "apes_dataset_handler.treat_file exited successfully",
            list_of_dataFrames,
            list_of_dataFramesUtilityLabels,
        )
    # except:
    #     return (
    #         1,
    #         "apes_dataset_handler.treat_file exited with error",
    #         list_of_dataFrames,
    #         list_of_dataFramesUtilityLabels,
    #     )


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
            Logger.info("apes_dataset_handler.treat_folder", info_message)
            return 1, "Subdirectories present. Case not yet handled", []
        else:
            info_message = f"There seem to be {no_of_files_in_dir_and_subdirs[0]} files in this directory. No subdirectories"
            Logger.info("apes_dataset_handler.treat_folder", info_message)

            # search for keynames given by the user
            if len(app_instance_metadata.dataset_metadata.file_keyword_names) != 0:
                for dirname, _, filenames in os.walk(
                    app_instance_metadata.dataset_metadata.dataset_path
                ):
                    for filename in filenames:
                        if (
                            pathlib.Path(filename).suffix
                            in app_instance_metadata.shared_definitions.supported_file_formats
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
            elif (
                len(app_instance_metadata.dataset_metadata.file_keyword_names) == 0
                or len(possible_files_list) == 0
            ):
                for dirname, _, filenames in os.walk(
                    app_instance_metadata.dataset_metadata.dataset_path
                ):
                    for filename in filenames:
                        if (
                            pathlib.Path(filename).suffix
                            in app_instance_metadata.shared_definitions.supported_file_formats
                        ):
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

        files_to_load = filter_files_in_folder_list(
            app_instance_metadata, possible_files_list
        )

        info_message = f"Possible files to load: {possible_files_list}"
        Logger.info("apes_dataset_handler.treat_folder", info_message)
        info_message = f"Files that will be loaded: {files_to_load}"
        Logger.info("apes_dataset_handler.treat_folder", info_message)

        return 0, "apes_dataset_handler.treat_folder exited successfully", files_to_load
    except:
        return 1, "apes_dataset_handler.treat_folder exited with error", []


def treat_archive(Logger, app_instance_metadata):
    try:
        info_message = "app_instance_metadata.dataset_metadata.dataset_path is an archive. Unarchiving."
        Logger.info("apes_dataset_handler.treat_archive", info_message)

        app_instance_metadata.dataset_metadata.dataset_path = unarchive(
            app_instance_metadata.dataset_metadata.dataset_path
        )

        info_message = (
            f"New file path is {app_instance_metadata.dataset_metadata.dataset_path}"
        )
        Logger.info("apes_dataset_handler.treat_archive", info_message)
        return 0, "apes_dataset_handler.treat_archive exited successfully"
    except:
        return 1, "apes_dataset_handler.treat_archive exited with error"


def process_singular_file_type(Logger, file, app_instance_metadata):
    if os.path.isfile(app_instance_metadata.dataset_metadata.dataset_path) == False:
        file = app_instance_metadata.dataset_metadata.dataset_path + "/" + file
    info_message = "## ----------------- New File ----------------- ## "
    Logger.info("apes_dataset_handler.process_singular_file_type", info_message)
    info_message = f"Processing file {file}"
    Logger.info("apes_dataset_handler.process_singular_file_type", info_message)

    match pathlib.Path(file).suffix:
        case ".csv":
            df = []
            loaded_with_header = False
            try:
                info_message = f"Trying to load file {file} without header"
                Logger.info(
                    "apes_dataset_handler.process_singular_file_type", info_message
                )
                df = pd.read_csv(file, header=None, dtype=np.float64)
            except:
                info_message = f"Trying to load file {file} with header"
                Logger.info(
                    "apes_dataset_handler.process_singular_file_type", info_message
                )
                df = pd.read_csv(file, dtype=np.float64)
                loaded_with_header = True
            finally:
                if loaded_with_header == True:
                    info_message = (
                        f"Loaded file {file} with header and then defaulted the header"
                    )
                else:
                    info_message = f"Loaded file {file} without header"
                Logger.info(
                    "apes_dataset_handler.process_singular_file_type", info_message
                )
                info_message = f"DataFrame is of shape {df.shape}"
                Logger.info(
                    "apes_dataset_handler.process_singular_file_type", info_message
                )

                if app_instance_metadata.dataset_metadata.shuffle_rows == True:
                    info_message = f"Shuffling rows for file {file}"
                    Logger.info(
                        "apes_dataset_handler.process_singular_file_type", info_message
                    )
                    df = df.sample(frac=1).reset_index(drop=True)
                    df.columns = list(range(len(df.columns)))
                    if app_instance_metadata.dataset_metadata.is_labeled == True:
                        # assuming label resides in the last column
                        classes_list = [
                            int(x) for x in df.iloc[:, df.shape[1] - 1].values
                        ]
                        if (
                            app_instance_metadata.dataset_metadata.number_of_classes
                            != how_many_different_values_in_list(classes_list)
                        ):
                            return (
                                1,
                                "Number of classes in dataFrame does not correspond to number of classes inputted",
                                df,
                            )
                        df = normalize_label_column(Logger, app_instance_metadata, df)
                return (
                    0,
                    "apes_dataset_handler.process_singular_file_type exited successfully",
                    df,
                )
        case ".arff":
            data = loadarff(file)
            df = pd.DataFrame(data[0])
            info_message = f"Loaded file {file}. DataFrame is of shape {df.shape}"
            Logger.info("apes_dataset_handler.process_singular_file_type", info_message)
            df.columns = list(range(len(df.columns)))
            if app_instance_metadata.dataset_metadata.is_labeled == True:
                # assuming label resides in the last column
                classes_list = [int(x) for x in df.iloc[:, df.shape[1] - 1].values]
                if (
                    app_instance_metadata.dataset_metadata.number_of_classes
                    != how_many_different_values_in_list(classes_list)
                ):
                    return (
                        1,
                        "Number of classes in dataFrame does not correspond to number of classes inputted",
                        df,
                    )
                df = normalize_label_column(Logger, app_instance_metadata, df)
            return (
                0,
                "apes_dataset_handler.process_singular_file_type exited successfully",
                df,
            )
        case ".txt":
            return "txt"
        case ".npz":
            return "npz"
        case other:
            return 1, "Could not idenfity file type.", df


# [1, 2, 3, 4] -> [0, 1, 2, 3]
def normalize_label_column(Logger, app_instance_metadata, dataFrame):
    df = []
    cols = dataFrame.shape[1] - 1

    if app_instance_metadata.dataset_metadata.is_labeled == True:
        # assuming label resides in the last column
        target_list = [int(x) for x in dataFrame.iloc[:, cols].values]
        print(
            "these many unique values "
            + str(how_many_different_values_in_list(target_list))
        )
        # info_message = f"Old targets are {target_list}"
        # Logger.info("apes_dataset_handler.normalize_label_column", info_message)

        if min(target_list) > 0:
            minimum = min(target_list)
            target_list = [x - minimum for x in target_list]

        # info_message = f"New targets are {target_list}"
        # Logger.info("apes_dataset_handler.normalize_label_column", info_message)

        df = dataFrame.drop(labels=cols, axis=1)
        new_targets = pd.DataFrame(target_list)
        new_targets.columns = ["temp_labels_targets"]
        df = df.join(new_targets)
        df.columns = list(range(len(df.columns)))
        return df
    return dataFrame


def rough_filename_filter(filename, keywords=rough_filename_searches):
    for string_match in keywords:
        result = re.search(string_match.lower(), filename)
        if result != None:
            return True
    return False


def filter_files_in_folder_list(app_instance_metadata, filename_list):
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
        for (
            extension
        ) in app_instance_metadata.shared_definitions.supported_file_formats:
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

    for element in seen:
        temp_list = []
        # get all dataFrames of one label in a single list
        for i in range(0, len(list_of_dataFramesUtilityLabels)):
            if element == list_of_dataFramesUtilityLabels[i]:
                temp_list.append(list_of_dataFrames[i])

        # mash those dataFrames into one single dataFrame
        first_dataFrame_shape = temp_list[0].shape
        for i in range(1, len(temp_list)):
            if temp_list[i].shape[1] != first_dataFrame_shape[1]:
                info_message = f"dataFrame.shape {temp_list[i].shape} at index {i} does not match {first_dataFrame_shape}"
                Logger.info("apes_dataset_handler.unify_dataFrames", info_message)

        if len(temp_list) == 1:
            info_message = f"1 dataFrame(s) for '{element}' in {list_of_dataFramesUtilityLabels}. Nothing to collapse"
        else:
            info_message = f"{len(temp_list)} dataFrame(s) for '{element}' in {list_of_dataFramesUtilityLabels}. Collapsing"
        Logger.info("apes_dataset_handler.unify_dataFrames", info_message)
        temp_list_of_dataFrames.append(pd.concat(temp_list))

    return (
        0,
        "apes_dataset_handler.unify_dataFrames exited successfully",
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
