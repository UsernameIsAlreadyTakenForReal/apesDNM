from apes_application import Apes_application
from apes_metadata_handler import *
from helpers_aiders_and_conveniencers.Logger import Logger


Logger = Logger()

dataset_metadata = Dataset_metadata(Logger, True, [], "target", 0, False, 0.7)
application_instance_metadata = Application_instance_metadata(
    Logger,
    dataset_metadata,
    "compare_solutions",
    "new_dataset",
    "ekg",
    "ekg",
    "supervised",
    1,
    "train_new_model",
)

apes_application_instance = Apes_application(Logger, application_instance_metadata)

apes_application_instance.run()
