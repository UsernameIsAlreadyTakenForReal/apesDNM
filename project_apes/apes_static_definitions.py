# This file:
# Contains global / static definitions.
# Use:
#
# * from apes_static_definitions import Shared_Definitions
# * shared_definitions = Shared_Definitions()
# * new_object = class_creator(shared_definitions, other, arguments)

import os


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
    supported_model_extensions = [".pth", "h5"]
    project_model_root_path = "./models"  # "./models" is relative to where temp_main_mockup.py (or apes_static_definitions??)

    project_solution_ekg_1_model_filename = "apes_solution_ekg_1_model.pth"
    project_solution_ekg_1_training_script = "apes_solution_ekg_1_train.py"
    project_solution_ekg_1_model_filename_last_good_one = (
        "s_ekg1_d_ekg2_20230606_233300.pth"
    )

    project_solution_ekg_2_model_filename = "apes_solution_ekg_2_model.h5"
    project_solution_ekg_2_training_script = "apes_solution_ekg_2_train.py"
    project_solution_ekg_2_model_filename_last_good_one = ""

    plot_savefile_format = "png"
    plot_show_at_runtime = False
    plot_savefile_location = "./../project_web/backend/images/"  # '.' is proejct_apes

    def __init__(self):
        # TODO: check centralized information (S3?)
        pass
