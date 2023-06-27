# This file:
# Contains global / static definitions.
# Use:
#
# * from apes_static_definitions import Shared_Definitions
# * shared_definitions = Shared_Definitions()
# * new_object = class_creator(shared_definitions, other, arguments)

import os
import json
import platform


class Shared_Definitions:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    ## Dataset-related parameters
    default_file_to_dataset_label = "train"
    rough_filename_searches = ["train", "test", "validation", "run"]
    supported_file_formats = [
        ".csv",
        ".arff",
        ".txt",
        ".npz",
        ".ts",
    ]  # also acts as priority when loading files

    ## Solutions-related parameters
    supported_model_extensions = [".pth", ".h5"]
    project_model_root_path = "./models"  # "./models" is relative to where temp_main_mockup.py (or apes_static_definitions??)
    project_solution_runs_path = "./runs"

    project_solution_ekg1_model_filename = "apes_solution_ekg_1_model.pth"
    project_solution_ekg1_d_ekg1_best_model_filename = (
        "----s_ekg1_d_ekg2_20230606_233300.pth"
    )

    project_solution_ekg2_model_filename = "apes_solution_ekg_2_model.h5"
    project_solution_ekg2_model_filename_last_good_one = (
        "----s_ekg2_d_ekg1_20230611_221423.h5"
    )
    plot_savefile_format = "png"
    plot_show_at_runtime = False
    plot_savefile_location = "./../project_web/backend/images/"  # '.' is proejct_apes

    def __init__(self):
        # TODO: check centralized information (S3?)

        if platform.system() == "Windows":
            file = open("apes_static_definitions.json")
        else:
            file = open(
                "/ebs_data/project_home/apesDNM/project_apes/apes_static_definitions.json"
            )

        data = json.load(file)

        ## DATASETS
        self.accepted_dataset_types = data["accepted_dataset_types"]
        self.default_file_to_dataset_label = data["default_file_to_dataset_label"]
        self.rough_filename_searches = data["rough_filename_searches"]
        self.supported_file_formats = data["supported_file_formats"]

        self.possible_extensions_for_numerical_datasets = data[
            "possible_extensions_for_numerical_datasets"
        ]
        self.possible_extensions_for_image_datasets = data[
            "possible_extensions_for_image_datasets"
        ]
        self.possible_extensions_for_audio_datasets = data[
            "possible_extensions_for_audio_datasets"
        ]

        ## MODELS
        self.supported_model_extensions = data["supported_model_extensions"]
        self.project_model_root_path = (
            data["project_model_root_path_development"]
            if platform.system() == "Windows"
            else data["project_model_root_path"]
        )
        self.project_solution_runs_path = (
            data["project_solution_runs_path_development"]
            if platform.system() == "Windows"
            else data["project_solution_runs_path"]
        )

        ## S_EKG1
        self.project_solution_ekg1_class_filename = data[
            "project_solution_ekg1_class_filename"
        ]
        self.project_solution_ekg1_model_filename = data[
            "project_solution_ekg1_model_filename"
        ]
        self.project_solution_ekg1_d_ekg1_best_model_filename = data[
            "project_solution_ekg1_d_ekg1_best_model_filename"
        ]
        self.project_solution_ekg1_d_ekg2_best_model_filename = data[
            "project_solution_ekg1_d_ekg2_best_model_filename"
        ]

        ## S_EKG2
        self.project_solution_ekg2_class_filename = data[
            "project_solution_ekg2_class_filename"
        ]
        self.project_solution_ekg2_model_filename = data[
            "project_solution_ekg2_model_filename"
        ]
        self.project_solution_ekg2_d_ekg1_best_model_filename = data[
            "project_solution_ekg2_d_ekg1_best_model_filename"
        ]
        self.project_solution_ekg2_d_ekg2_best_model_filename = data[
            "project_solution_ekg2_d_ekg2_best_model_filename"
        ]

        ## S_IMG1
        self.project_solution_img1_class_filename = data[
            "project_solution_img1_class_filename"
        ]
        self.project_solution_img1_model_filename = data[
            "project_solution_img1_model_filename"
        ]
        self.project_solution_img1_d_img1_best_model_filename = data[
            "project_solution_img1_d_img1_best_model_filename"
        ]

        ## PLOTS
        self.plot_savefile_format = data["plot_savefile_format"]
        self.plot_savefile_location = (
            data["plot_savefile_location_development"]
            if platform.system() == "Windows"
            else data["plot_savefile_location"]
        )

        print(self.plot_savefile_location)

        ## DONE
