from apes_application import APES_Application
from apes_metadata_handler import *
from helpers_aiders_and_conveniencers.Logger import Logger


Logger = Logger()

example_path = "../../Dataset - ECG_Heartbeat/mitbih_test.csv"

dataset_metadata = Dataset_Metadata(
    Logger,
    example_path,
    True,
    ["Normal", "R on T", "PVC", "SP", "UB"],
    "target",
    0,
    False,
    0.7,
)
application_instance_metadata = Application_Instance_Metadata(
    Logger,
    dataset_metadata,
    "run_one_solution",
    "existing_dataset",
    "ekg",
    "ekg",
    "supervised",
    1,
    "train_new_model",
)

apes_application_instance = APES_Application(Logger, application_instance_metadata)

apes_application_instance.run()
