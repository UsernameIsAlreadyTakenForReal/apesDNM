# This file:
# Contains two classes meant to hold metadata about the running application (apes project) instance and the dataset.

# ############ Dataset metada:
# is_labeled: True/False
# file_keyword_names: [] (files usually come as either 'something_train.csv' and 'something_test.csv'. User can give keywords to search by)
#
# * For labeled datasets:
# class_names: list (e.g. ['normal', 'Q on F'] etc)
# label_column_name: string (e.g. 'target' or 'label'); not vital
# numerical_value_of_desired_label: int (e.g. 0)
# separate_train_and_test: True / False (possibly redundant)
# percentage_of_split: [value] / [value1, value2] (where value2 is % to split for validation)
# shuffle_rows: True / False

from apes_static_definitions import Shared_Definitions
import pandas as pd
import os


class Dataset_Metadata:
    def __init__(
        self,
        Logger,
        dataset_path="",
        is_labeled=True,
        file_keyword_names=[],
        class_names=[],
        label_column_name="target",
        numerical_value_of_desired_label=0,
        separate_train_and_test=False,
        percentage_of_split=0.7,
        shuffle_rows=False,
    ):
        self.Logger = Logger

        self.dataset_path = dataset_path
        self.is_labeled = is_labeled
        self.file_keyword_names = file_keyword_names
        self.class_names = class_names
        self.label_column_name = label_column_name
        self.numerical_value_of_desired_label = numerical_value_of_desired_label
        self.separate_train_and_test = separate_train_and_test
        self.percentage_of_split = percentage_of_split
        self.shuffle_rows = shuffle_rows

        info_message = "Created object of type Dataset_Metadata"
        self.Logger.info(self, info_message)

    def getMetadataAsString(self):
        return (
            "\nDataset_metadata"
            + "\n|___dataset_path = "
            + str(self.dataset_path)
            + "\n|___is_labeled = "
            + str(self.is_labeled)
            + "\n|___file_keyword_names = "
            + str(self.file_keyword_names)
            + "\n|___class_names = "
            + str(self.class_names)
            + "\n|___label_column_name = "
            + str(self.label_column_name)
            + "\n|___numerical_value_of_desired_label = "
            + str(self.numerical_value_of_desired_label)
            + "\n|___separate_train_and_test = "
            + str(self.separate_train_and_test)
            + "\n|___percentage_of_split = "
            + str(self.percentage_of_split)
            + "\n|___shuffle_rows = "
            + str(self.shuffle_rows)
        )


# ############ Application instance metada:
# Dataset_metadata: an object of type dataset_metadata. TODO: multiple such objects for a type of application_mode == benchmark_solution
#                                                             (same solution, multiple datasets)?
#
# * Development / Maintenance parameters
# display_dataFrames: True / False (to not clog logs)
#
# * Functional parameters
# application_mode: compare_solutions / run_one_solution (redunant, obtainable from solution_index)
# dataset_origin: new_dataset / (user chose) existing_dataset
# dataset_category: ekg / img / ral / N/A
# solution_category: ekg / img / ral (redundant)
# solution_nature: supervised / unsupervised
# solution_index: 1/2/3 OR [1,2,3] OR [1,2] OR [1,3] OR [2,3] for ekg, 1 for img, 1 for ral
# {FOR DATASET_ORIGIN=='new_data'} model_origin: train_new_model / use_existing_model
# model_train_epochs: (int)


class Application_Instance_Metadata:
    def __init__(
        self,
        Logger,
        dataset_metadata,
        display_dataFrames=False,
        application_mode="compare_solutions",
        dataset_origin="new_dataset",
        dataset_category="ekg",
        solution_category="ekg",
        solution_nature="supervised",
        solution_index=1,
        model_origin="train_new_model",
        model_train_epochs=40,
    ):
        self.Logger = Logger
        self.dataset_metadata = dataset_metadata

        self.display_dataFrames = (display_dataFrames,)

        self.application_mode = application_mode
        self.dataset_origin = dataset_origin
        self.dataset_category = dataset_category
        self.solution_category = solution_category
        self.solution_nature = solution_nature
        self.solution_index = solution_index
        self.model_origin = model_origin
        self.model_train_epochs = model_train_epochs

        self.shared_definitions = Shared_Definitions()

        info_message = "Created object of type Application_Instance_Metadata"
        self.Logger.info(self, info_message)

    def getMetadataAsString(self):
        return (
            "\nApplication_metadata"
            + "\n----display_dataFrames = "
            + str(self.display_dataFrames)
            + "\n----application_mode = "
            + str(self.application_mode)
            + "\n----dataset_origin = "
            + str(self.dataset_origin)
            + "\n----dataset_category = "
            + str(self.dataset_category)
            + "\n----solution_category = "
            + str(self.solution_category)
            + "\n----solution_nature = "
            + str(self.solution_nature)
            + "\n----solution_index = "
            + str(self.solution_index)
            + "\n----model_origin = "
            + str(self.model_origin)
            + "\n----model_train_epochs = "
            + str(self.model_train_epochs)
        )

    def printMetadata(self):
        info_message = (
            "Running with this configuration: "
            + self.getMetadataAsString()
            + self.dataset_metadata.getMetadataAsString()
        )
        self.Logger.info(self, info_message)


from matplotlib import pyplot as plt
from flask.json import jsonify
from datetime import datetime
import seaborn as sns
import uuid


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

                df = pd.read_csv(full_path, header=None)

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

                # self.Logger.info(
                #     self, "creating heatmap... this bitch might take a while"
                # )
                # beginning_time = datetime.now()

                # plt.clf()
                # plt.figure()
                # sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
                # plt.title("heatmap for file #" + str(index) + ": " + filename)
                # full_path = os.path.join("images", str(uuid.uuid4()) + ".png")
                # plt.savefig(full_path)
                # plots.append(
                #     {
                #         "path": full_path,
                #         "caption": "heatmap for file #" + str(index) + ": " + filename,
                #     }
                # )

                # difference_in_seconds = (datetime.now() - beginning_time).seconds
                # self.Logger.info(
                #     self,
                #     "motherfucker took "
                #     + str(difference_in_seconds)
                #     + " seconds to create one fucking plot...",
                # )

                self.Logger.info(self, "creating histogram...")
                plt.clf()
                plt.figure()
                plt.hist(df.iloc[:, y - 1])
                plt.xlabel("value")
                plt.ylabel("frequency")
                plt.title("histogram for file #" + str(index) + ": " + filename)
                full_path = os.path.join("images", str(uuid.uuid4()) + ".png")
                plt.savefig(full_path)
                plots.append(
                    {
                        "path": full_path,
                        "caption": "histogram for file #"
                        + str(index)
                        + ": "
                        + filename,
                    }
                )

                dict["plots"] = plots

                # dict["info"] = str(df.info())
                # dict["describe"] = str(df.describe())

                results.append(dict)

        self.Logger.info(self, "eda performed successfully")

        return results
