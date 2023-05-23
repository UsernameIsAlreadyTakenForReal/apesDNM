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

    project_solution_ekg_1_model_filename = "apes_solution_ekg_1_model.pth"
    project_solution_ekg_1_training_script = "apes_solution_ekg_1_train.py"
    project_solution_ekg_1_model_filename_last_good_one = (
        "apes_solution_ekg_1_model.pth"
    )

    project_solution_ekg_2_model_filename = "apes_solution_ekg_2_model.h5"
    project_solution_ekg_2_training_script = "apes_solution_ekg_2_train.py"
    project_solution_ekg_2_model_filename_last_good_one = "apes_solution_ekg_2_model.h5"
