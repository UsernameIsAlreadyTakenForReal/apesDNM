# This file:
# Contains the main application class.

from apes_discerner import Discerner
from apes_dataset_handler import *

from apes_static_definitions import Shared_Definitions

from type_ekg.apes_solution_ekg_1_train import Solution_ekg_1
from type_ekg.apes_solution_ekg_2_train import Solution_ekg_2

# TODO:
# * fail if compare_solutions + solution_type==ral or solution_type==img
# * check solution_nature against dataset_metadata.is_labeled and process dataset if necessary


class APES_Application:
    def __init__(self, Logger, application_instance_metadata):
        self.Logger = Logger
        self.application_instance_metadata = application_instance_metadata

        info_message = "Created object of type APES_Application."
        self.Logger.info(self, info_message)
        info_message = "Starting run. This is a mock-up run."
        self.Logger.info(self, info_message)

    def run(self):
        self.application_instance_metadata.printMetadata()

        ## Part 1 -- Get the desired dataset as a pandas dataFrame
        if self.application_instance_metadata.dataset_origin == "new_dataset":
            info_message = "Dataset is not coming from our database."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.dataset_category == "N/A":
                info_message = "We do not seem to know what kind of dataset this is. Calling the discerner."
                self.Logger.info(self, info_message)

                discerner = Discerner(self.Logger)

            else:
                info_message = "We know this dataset is of type " + str(
                    self.application_instance_metadata.dataset_category
                )
                self.Logger.info(self, info_message)

                self.dataFrame = process_file_type(self.application_instance_metadata)
                pass

        else:
            info_message = "Dataset is coming from our database."
            self.Logger.info(self, info_message)

            # We don't have unidentified databases at the moment of writing this code, but just in case.
            if self.application_instance_metadata.dataset_category == "N/A":
                info_message = "We do not seem to know what kind of dataset this is. Calling the discerner."
                self.Logger.info(self, info_message)

                discerner = Discerner(self.Logger)

            else:
                info_message = "We know this dataset is of type " + str(
                    self.application_instance_metadata.dataset_category
                )
                self.Logger.info(self, info_message)

                self.dataFrame = process_file_type(self.application_instance_metadata)
                pass

        ## Part 2 -- Get the desired solution(s) runners
        if self.application_instance_metadata.application_mode == "run_one_solution":
            info_message = "Only one solution to load."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.solution_category == "ral":
                info_message = "Loading solution ral_1"
                self.Logger.info(self, info_message)
                pass
            elif self.application_instance_metadata.solution_category == "img":
                info_message = "Loading solution img_1"
                self.Logger.info(self, info_message)
                pass
            else:
                match int(self.application_instance_metadata.solution_index):
                    case 1:
                        info_message = "Loading supervised solution ekg_1"
                        self.Logger.info(self, info_message)

                        self.solution = Solution_ekg_1(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        pass
                    case 2:
                        info_message = "Loading solution ekg_2"
                        self.Logger.info(self, info_message)

                        self.solution = Solution_ekg_2(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        pass

        else:
            solution_indexes = list(self.application_instance_metadata.solution_index)
            info_message = "Multiple solutions to load."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.solution_category == "ral":
                info_message = "Beep beep, I'm a sheep. There's only 1 ral solution. Why is there a directive to load more?"
                self.Logger.info(self, info_message)
            elif self.application_instance_metadata.solution_category == "img":
                info_message = "Beep beep, I'm a sheep. There's only 1 img solution. Why is there a directive to load more?"
                self.Logger.info(self, info_message)
            else:
                match solution_indexes:
                    case [1, 2]:
                        info_message = (
                            "Multiple solutions to load. Loading ekg_1 and ekg_2."
                        )
                        self.Logger.info(self, info_message)

                        self.solution_1 = Solution_ekg_1(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        self.solution_2 = Solution_ekg_2(
                            self.application_instance_metadata.shared_definitions,
                            self.Logger,
                        )
                        pass
