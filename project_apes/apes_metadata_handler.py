# This file:
# Contains two classes meant to hold metadata about the running application (apes project) instance and the dataset.

# ############ Dataset metada:
# is_labeled: True/False
#
# * For labeled datasets:
# class_names: list (e.g. ['normal', 'Q on F'] etc)
# label_column_name: string (e.g. 'target' or 'label'); not vital
# numerical_value_of_desired_label: int (e.g. 0)
# percentage_of_split: value / [value1, value2] (where value2 is % to split for validation)


class Dataset_metadata:
    def __init__(
        self,
        Logger,
        dataset_path="",
        is_labeled=True,
        class_names=[],
        label_column_name="target",
        numerical_value_of_desired_label=0,
        separate_train_and_test=False,
        percentage_of_split=0.7,
    ):
        self.Logger = Logger

        self.dataset_path = dataset_path
        self.is_labeled = is_labeled
        self.class_names = class_names
        self.label_column_name = label_column_name
        self.numerical_value_of_desired_label = numerical_value_of_desired_label
        self.separate_train_and_test = separate_train_and_test
        self.percentage_of_split = percentage_of_split

        info_message = "Created object of type Dataset_metadata"
        self.Logger.info(self, info_message)

    def getMetadataAsString(self):
        return (
            "\nDataset_metadata"
            + "\n dataset_path = "
            + str(self.dataset_path)
            + "\n is_labeled = "
            + str(self.is_labeled)
            + "\n class_names = "
            + str(self.class_names)
            + "\n label_column_name = "
            + str(self.label_column_name)
            + "\n numerical_value_of_desired_label = "
            + str(self.numerical_value_of_desired_label)
            + "\n separate_train_and_test = "
            + str(self.separate_train_and_test)
            + "\n percentage_of_split = "
            + str(self.percentage_of_split)
        )


# ############ Application instance metada:
# Dataset_metadata: an object of type dataset_metadata. TODO: multiple such objects for a type of application_mode == benchmark_solution
#                                                             (same solution, multiple datasets)?
# application mode: compare solutions / run one solution only
# dataset_origin: new_dataset / (user chose) existing_dataset
# dataset_category: ekg / img / ral / N/A
# solution_category: ekg / img / ral (redundant)
# solution_nature: supervised / unsupervised
# solution_index: 1/2/3 OR [1,2,3] OR [1,2] OR [1,3] OR [2,3] for ekg, 1 for img, 1 for ral
# {FOR DATASET_ORIGIN=='new_data'} model_origin: train_new_model / use_existing_model


class Application_instance_metadata:
    def __init__(
        self,
        Logger,
        dataset_metadata,
        application_mode="compare_solutions",
        dataset_origin="new_dataset",
        dataset_category="ekg",
        solution_category="ekg",
        solution_nature="supervised",
        solution_index=1,
        model_origin="train_new_model",
    ):
        self.Logger = Logger
        self.dataset_metadata = dataset_metadata

        self.application_mode = application_mode
        self.dataset_origin = dataset_origin
        self.dataset_category = dataset_category
        self.solution_category = solution_category
        self.solution_nature = solution_nature
        self.solution_index = solution_index
        self.model_origin = model_origin

        info_message = "Created object of type Application_instance_metadata"
        self.Logger.info(self, info_message)

    def getMetadataAsString(self):
        return (
            "\nApplication_metadata"
            + "\n application_mode = "
            + str(self.application_mode)
            + "\n dataset_origin = "
            + str(self.dataset_origin)
            + "\n dataset_category = "
            + str(self.dataset_category)
            + "\n solution_category = "
            + str(self.solution_category)
            + "\n solution_nature = "
            + str(self.solution_nature)
            + "\n solution_index = "
            + str(self.solution_index)
            + "\n model_origin = "
            + str(self.model_origin)
        )

    def printMetadata(self):
        info_message = (
            "Running with this configuration: "
            + self.getMetadataAsString()
            + self.dataset_metadata.getMetadataAsString()
        )
        self.Logger.info(self, info_message)
