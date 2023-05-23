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


class dataset_metadata:
    def __init__(
        self,
        is_labeled,
        class_names,
        label_column_name,
        numerical_value_of_desired_label,
        percentage_of_split,
    ):
        self.is_labeled = is_labeled
        self.class_names = class_names
        self.label_column_name = label_column_name
        self.numerical_value_of_desired_label = numerical_value_of_desired_label
        self.percentage_of_split = percentage_of_split


# ############ Application instance metada:
# application mode: compare solutions / run one solution only
# dataset_origin: new_data / user chose existing data
# dataset_category: ekg / img / ral
# solution_category: ekg / img / ral (redundant)
# solution_nature: supervised / unsupervised
# solution_index: 1/2/3 OR [1,2,3] OR [1,2] OR [1,3] OR [2,3] for ekg, 1 for img, 1 for ral
# {FOR DATASET_ORIGIN=='new_data'} model_origin: train_model / use_existing_model


class application_instance_metadata:
    def __init__(
        self,
        application_mode="compare_solutions",
        dataset_origin="new_dataset",
        dataset_category="ekg",
        solution_category="ekg",
        solution_nature="supervised",
        solution_index=1,
        model_origin="train_model",
    ):
        self.application_mode = application_mode
        self.dataset_origin = dataset_origin
        self.dataset_category = dataset_category
        self.solution_category = solution_category
        self.solution_nature = solution_nature
        self.solution_index = solution_index
        self.model_origin = model_origin
