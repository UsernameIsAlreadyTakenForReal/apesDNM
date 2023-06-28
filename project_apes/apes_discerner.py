# This file:
# Defines a class meant to figure out what kind of data the given dataset contains.

import sys
import os
import numpy as np
import pathlib


class Discerner:
    def __init__(self, Logger, application_instance_metadata):
        self.Logger = Logger

        info_message = "Created object of type Discerner."
        self.Logger.info(self, info_message)

        self.application_instance_metadata = application_instance_metadata
        self.shared_definitions = application_instance_metadata.shared_definitions

        self.predicted_dataset_type = "ekg"

    def get_dataset_type(self):

        self.path = self.application_instance_metadata.dataset_metadata.dataset_path
        self.number_of_accepted_datasets = len(self.application_instance_metadata.shared_definitions.accepted_dataset_types)
        self.accepted_dataset_types = self.application_instance_metadata.shared_definitions.accepted_dataset_types

        number_of_files = self.get_number_of_files_to_handle(self.path)
        self.dataset_type_probabilities_matrix = np.zeros([number_of_files, self.number_of_accepted_datasets], dtype=int)
        self.dataset_type_probability_list = np.zeros(self.number_of_accepted_datasets, dtype=int)

        info_message = f"Found {number_of_files} files to be analyzed by Discerner."
        self.Logger.info(self, info_message)

        self.get_1weight()

        self.predicted_dataset_type = self.interpret_probability_matrix()

        info_message = f"Determined dataset type to be {self.predicted_dataset_type}"
        self.Logger.info(self, info_message)

        return 0, "Discerner.get_dataset_type() exited successfully", self.predicted_dataset_type

    def interpret_probability_matrix(self):
        for row in self.dataset_type_probabilities_matrix:
            if np.where(row == max(row))[0].size == 1: # if there's a single column with that maximum value
                self.dataset_type_probability_list[np.where(row == max(row))[0][0]] += 1
            else:
                pass

        self.predicted_dataset_type = self.accepted_dataset_types[int(np.where(self.dataset_type_probability_list == max(self.dataset_type_probability_list))[0][0])]
        return self.predicted_dataset_type

    def get_1weight(self):
        ## Weight 1 -- File extensions
        possible_extensions_for_numerical = (
            self.shared_definitions.possible_extensions_for_numerical_datasets
        )
        possible_extensions_for_image = (
            self.shared_definitions.possible_extensions_for_image_datasets
        )
        possible_extensions_for_audio = (
            self.shared_definitions.possible_extensions_for_audio_datasets
        )

        for dirname, _, filenames in os.walk(self.path):
            number_of_files_in_this_dir = len(filenames)
            info_message = f"{number_of_files_in_this_dir} files found in this directory"
            self.Logger.info(self, info_message)

            for index, filename in enumerate(filenames):
                if pathlib.Path(filename).suffix in possible_extensions_for_numerical:
                    self.dataset_type_probabilities_matrix[index, 0] += 1
                elif pathlib.Path(filename).suffix in possible_extensions_for_image:
                    self.dataset_type_probabilities_matrix[index, 1] += 1
                elif pathlib.Path(filename).suffix in possible_extensions_for_audio:
                    self.dataset_type_probabilities_matrix[index, 2] += 1
                pass


    def get_number_of_files_to_handle(self, path):
        number_of_files = 0
        for dirname, _, filenames in os.walk(path):
            number_of_files += len(filenames)

        return number_of_files