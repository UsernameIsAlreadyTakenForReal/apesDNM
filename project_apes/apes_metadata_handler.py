# This file:
# Contains two classes meant to hold metadata about the running application (apes project) instance and the dataset.

# ############ Dataset metada:
# is_labeled: True/False
# file_keyword_names: [] (files usually come as either 'something_train.csv' and 'something_test.csv'. User can give keywords to search by)
# dataset_name_stub: string (short name that comes from user, but since we're not doing that, well...)
#
# * For labeled datasets:
# class_names: list (e.g. ['normal', 'Q on F'] etc)
# label_column_name: string (e.g. 'target' or 'label'); not vital
# numerical_value_of_desired_label: int (e.g. 0)
# separate_train_and_test: True / False (possibly redundant)
# percentage_of_split: [value] / [value1, value2] (where value2 is % to split for validation)
# shuffle_rows: True / False

from apes_static_definitions import Shared_Definitions


class Dataset_Metadata:
    def __init__(
        self,
        Logger,
        dataset_path="",
        dataset_name_stub="",
        is_labeled=True,
        file_keyword_names=[],
        number_of_classes=5,
        class_names=[],
        label_column_name="target",
        numerical_value_of_desired_label=0,
        separate_train_and_test=False,
        percentage_of_split=0.7,
        shuffle_rows=False,
    ):
        self.Logger = Logger

        self.dataset_path = dataset_path
        self.dataset_name_stub = dataset_name_stub
        self.is_labeled = is_labeled
        self.file_keyword_names = file_keyword_names
        self.number_of_classes = number_of_classes
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
            + "\n|___dataset_name_stub = "
            + str(self.dataset_name_stub)
            + "\n|___is_labeled = "
            + str(self.is_labeled)
            + "\n|___number_of_classes = "
            + str(self.number_of_classes)
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
        app_instance_ID,
        display_dataFrames=False,
        application_mode="compare_solutions",
        dataset_origin="new_dataset",
        dataset_category="ekg",
        solution_category="ekg",
        solution_nature="supervised",
        solution_index=[1],
        model_origin="train_new_model",
        model_train_epochs=40,
    ):
        self.Logger = Logger
        self.dataset_metadata = dataset_metadata

        self.app_instance_ID = app_instance_ID

        self.display_dataFrames = display_dataFrames

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
            + "\n|___display_dataFrames = "
            + str(self.display_dataFrames)
            + "\n|___application_mode = "
            + str(self.application_mode)
            + "\n|___dataset_origin = "
            + str(self.dataset_origin)
            + "\n|___dataset_category = "
            + str(self.dataset_category)
            + "\n|___solution_category = "
            + str(self.solution_category)
            + "\n|___solution_nature = "
            + str(self.solution_nature)
            + "\n|___solution_index = "
            + str(self.solution_index)
            + "\n|___model_origin = "
            + str(self.model_origin)
            + "\n|___model_train_epochs = "
            + str(self.model_train_epochs)
        )

    def printMetadata(self):
        info_message = (
            "Running with this configuration: "
            + self.getMetadataAsString()
            + self.dataset_metadata.getMetadataAsString()
        )
        self.Logger.info(self, info_message)
