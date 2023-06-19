from matplotlib import pyplot as plt
from flask.json import jsonify
from datetime import datetime
import seaborn as sns
import uuid

import random
import os
import pandas as pd

from apes_dataset_handler import load_file_type


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
    ):
        self.Logger = Logger
        self.dataset_path = dataset_path

    def perform_eda(self):
        results = []

        self.Logger.info(self, "EDA started for " + self.dataset_path)

        for dirname, _, filenames in os.walk(self.dataset_path):
            for index, filename in enumerate(filenames):
                dict = {}

                full_path = os.path.join(dirname, filename)
                self.Logger.info(self, full_path)

                return_code, return_message, df = load_file_type(self.Logger, full_path)
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

                results.append(dict)

        self.Logger.info(
            self, "eda performed successfully for files in " + self.dataset_path
        )

        return results
