# This file:
# Contains the main application class.

from apes_discerner import Discerner
from apes_dataset_handler import *


class Apes_application:
    def __init__(self, Logger, application_instance_metadata):
        self.Logger = Logger
        self.application_instance_metadata = application_instance_metadata

        info_message = "Created object of type Apes_application."
        self.Logger.info(self, info_message)
        info_message = "Starting run. This is a mock-up run."
        self.Logger.info(self, info_message)

    def run(self):
        self.application_instance_metadata.printMetadata()

        if self.application_instance_metadata.dataset_origin == "new_dataset":
            info_message = "Dataset is not coming from our database."
            self.Logger.info(self, info_message)

            if self.application_instance_metadata.dataset_category == "N/A":
                info_message = "We do not seem to know what kind of dataset this is. Calling the discerner."
                self.Logger.info(self, info_message)

                discerner = Discerner(self.Logger)

            else:
                info_message = "User defined this dataset as type " + str(
                    self.application_instance_metadata.dataset_category
                )
                self.Logger.info(self, info_message)

                dataFrame = process_file_type(self.application_instance_metadata)
                print(dataFrame[0][0])
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

                dataFrame = process_file_type(self.application_instance_metadata)
                print(dataFrame[0][0])
                pass
