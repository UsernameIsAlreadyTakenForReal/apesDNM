from matplotlib import pyplot as plt
from flask.json import jsonify
from datetime import datetime
import seaborn as sns
import uuid

import random
import os
import pandas as pd

from apes_dataset_handler import load_file_type
from helpers_aiders_and_conveniencers.misc_functions import (
    how_many_different_values_in_list,
)


def plot():
    number_of_points = random.randint(5, 15)
    points_x = []
    points_y = []

    for _ in range(number_of_points):
        points_x.append(random.randint(0, 100))
        points_y.append(random.randint(0, 100))

    plt.clf()

    plt.plot(points_x, points_y)

    plt.ylabel(str(number_of_points) + " random numbers")
    plt.xlabel(str("just as many random numbers"))

    plot_title = "a plot with " + str(number_of_points) + " random numbers"
    plt.title(plot_title)

    filename = str(uuid.uuid4()) + ".png"

    figure_path_png = os.path.join("images", filename)

    plt.savefig(figure_path_png)
    return figure_path_png, plot_title


class Dataset_EDA:
    def __init__(
        self,
        Logger,
        dataset_path,
        shared_definitions,
    ):
        self.Logger = Logger
        self.shared_definitions = shared_definitions
        self.dataset_path = dataset_path

    def perform_eda(self):
        import pathlib

        results = []

        self.Logger.info(self, "EDA started for " + self.dataset_path)

        return_code, return_message, verifier = self.discriminate_dataset_category()
        if return_code != 0:
            return return_code, return_message, [], []

        if verifier[0]:
            self.dataset_category = "ekg"
        if verifier[1]:
            self.dataset_category = "img"
        if verifier[2]:
            self.dataset_category = "ral"

        for dirname, _, filenames in os.walk(self.dataset_path):
            for index, filename in enumerate(filenames):
                if pathlib.Path(filename).suffix != ".json":
                    dict = {}

                    full_path = os.path.join(dirname, filename)
                    self.Logger.info(self, full_path)

                    return_code, return_message, df = load_file_type(
                        self.Logger, full_path
                    )
                    # self.Logger.info(self, str(type(df)))

                    dict["index"] = index
                    dict["filename"] = filename

                    x, y = df.shape

                    dict["rows"] = str(x)
                    dict["columns"] = str(y)
                    dict["head"] = str(df.head(5)).split("\n")

                    missing_data = 0
                    nulls = df.isnull().sum().to_frame()
                    for _, row in nulls.iterrows():
                        if row[0] != 0:
                            missing_data = missing_data + 1

                    dict["columns_with_missing_data"] = str(missing_data)

                    plots = []

                    # #############################################################

                    # self.Logger.info(
                    #     self,
                    #     "creating heatmap for " + filename + "... this might take a while",
                    # )
                    # beginning_time = datetime.now()

                    # plt.clf()
                    # plt.figure()
                    # sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
                    # caption = "heatmap for file #" + str(index) + ": " + filename
                    # plt.title(caption)
                    # full_path = os.path.join("images", str(uuid.uuid4()) + ".png")
                    # plt.savefig(full_path)
                    # plots.append(
                    #     {
                    #         "path": full_path,
                    #         "caption": caption,
                    #     }
                    # )

                    # difference_in_seconds = (datetime.now() - beginning_time).seconds
                    # self.Logger.info(
                    #     self,
                    #     "took " + str(difference_in_seconds) + " seconds to create...",
                    # )

                    # #############################################################

                    plt.clf()
                    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
                    plt.xlabel("column 0")
                    plt.ylabel("column 1")
                    caption = "scatter plot for file #" + str(index) + ": " + filename
                    plt.title(caption)
                    full_path = os.path.join("images", str(uuid.uuid4()) + ".png")
                    plt.savefig(full_path)
                    plots.append({"path": full_path, "caption": caption})

                    self.Logger.info(self, "creating histogram for " + filename + "...")
                    plt.clf()
                    plt.figure()
                    plt.hist(df.iloc[:, y - 1])
                    plt.xlabel("value")
                    plt.ylabel("frequency")
                    caption = "histogram for file #" + str(index) + ": " + filename
                    plt.title(caption)
                    full_path = os.path.join("images", str(uuid.uuid4()) + ".png")
                    plt.savefig(full_path)
                    plots.append({"path": full_path, "caption": caption})

                    # for _ in range(random.randint(1, 3)):
                    #     _full_path, _caption = plot()
                    #     plots.append({"path": _full_path, "caption": _caption})

                    dict["plots"] = plots

                    dict["info"] = str(df.info())
                    dict["describe"] = str(df.describe()).split("\n")

                    possible_number_of_classes = how_many_different_values_in_list(
                        df.iloc[:, y - 1]
                    )
                    dict["possible_number_of_classes"] = str(possible_number_of_classes)

                    results.append(dict)

        self.Logger.info(
            self, "eda performed successfully for files in " + self.dataset_path
        )

        return (
            0,
            "Dataset_EDA.perform_eda() exited successfully",
            results,
            self.dataset_category,
        )

    def discriminate_dataset_category(self):
        import os
        import pathlib

        possible_extensions_for_numerical = (
            self.shared_definitions.possible_extensions_for_numerical_datasets
        )
        possible_extensions_for_image = (
            self.shared_definitions.possible_extensions_for_image_datasets
        )
        possible_extensions_for_audio = (
            self.shared_definitions.possible_extensions_for_audio_datasets
        )

        contains_only_numerical = True
        contains_only_image = True
        contains_only_audio = True

        file_extensions = (
            possible_extensions_for_numerical
            + possible_extensions_for_image
            + possible_extensions_for_audio
        )

        file_candidates = []

        for dirname, _, filenames in os.walk(self.dataset_path):
            for index, filename in enumerate(filenames):
                if pathlib.Path(filename).suffix in file_extensions:
                    file_candidates.append(filename)

        ## for numerical
        for file in file_candidates:
            if pathlib.Path(file).suffix not in possible_extensions_for_numerical:
                contains_only_numerical = False
                break

        ## for image
        for file in file_candidates:
            if pathlib.Path(file).suffix not in possible_extensions_for_image:
                contains_only_image = False
                break

        ## for audio
        for file in file_candidates:
            if pathlib.Path(file).suffix not in possible_extensions_for_audio:
                contains_only_audio = False
                break

        # info_message = f"discriminate_dataset_category() -- Given dataset contains only numerical: {contains_only_numerical}, contains only images: {contains_only_image}, contains only audio: {contains_only_audio}"
        # self.Logger(self, info_message)

        verifier = [contains_only_numerical, contains_only_image, contains_only_audio]
        if sum(verifier) != 1:
            return (
                1,
                "discriminate_dataset_category() -- can't have multiple dataset categories at the same time",
            )
        else:
            return 0, "discriminate_dataset_category() -- exited successfully", verifier
