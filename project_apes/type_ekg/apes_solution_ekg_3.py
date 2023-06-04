# This file:
# Contains the solution_ekg_3. ?
#
# Main methods to be called from apes_application:
# create_model(), load_model()
#       -- purpose: self.model exists
# save_model()
#       -- purpose: self.model save
# train(epochs), test()
#       -- purpose: train and use self.model
# adapt_dataset(self, application_instance_metadata, list_of_dataFrames, list_of_dataFramesUtilityLabels)
#       -- purpose:


class Solution_ekg_2:
    def __init__(self, shared_definitions, Logger):
        self.Logger = Logger

        self.Logger.info(self, "Creating object of type solution_ekg_3")
        self.shared_definitions = shared_definitions
        self.project_solution_model_filename = (
            shared_definitions.project_solution_ekg_3_model_filename
        )
        self.project_solution_training_script = (
            shared_definitions.project_solution_ekg_3_training_script
        )

    ## -------------- General methods --------------
    ## Model methods
    def create_model(self):  # im_shape = (X_train.shape[1], 1)
        pass

    def save_model(self, filename="", path=""):
        pass

    def load_model(self, filename="", path=""):
        pass

    ## Functionality methods
    def train(self, epochs=40):
        pass

    def test(self):
        info_message = "Begining testing"
        self.Logger.info(self, info_message)
        info_message = f"Testing - it took "
        self.Logger.info(self, info_message)
        pass

    ## Dataset methods
    def adapt_dataset(
        self,
        application_instance_metadata,
        list_of_dataFrames,
        list_of_dataFramesUtilityLabels,
    ):
        pass

    ## -------------- Particular methods --------------
