import sys

n = len(sys.argv)
print("Total number of arguments passed: ", n)

for i in range(0, n):
    print(f"Parameter {i}: {sys.argv[i]}")

example_path = ""

if int(sys.argv[1]) == 1:
    print("Running with dataset egk1")
    example_path = "../../Datasets/Dataset - ECG5000/Dataset - ECG5000"
elif int(sys.argv[1]) == 2:
    print("Running with dataset ekg2")
    example_path = "../../Datasets/Dataset - ECG_Heartbeat/Dataset - ECG_Heartbeat.zip"
else:
    print("whatev")

from helpers_aiders_and_conveniencers.logger import Logger

Logger = Logger()

init_message = "Begin"
Logger.info("Program", init_message)

from apes_application import APES_Application

init_message = "Imported APES_Application"
Logger.info("Program", init_message)


from apes_metadata_handler import *

init_message = "Imported apes_metadata_handler.*"
Logger.info("Program", init_message)


# example_path = "../../Datasets/Dataset - ECG_Heartbeat/Dataset - ECG_Heartbeat.zip"
# # example_path = "../../Datasets/Dataset - ECG5000/Dataset - ECG5000"


dataset_metadata = Dataset_Metadata(
    Logger,
    example_path,
    True,
    [],
    ["Normal", "R on T", "PVC", "SP", "UB"],
    "target",
    0,
    False,
    [0.7],
    True,
)
application_instance_metadata = Application_Instance_Metadata(
    Logger,
    dataset_metadata,
    False,
    "compare_solution",
    "existing_dataset",
    "ekg",
    "ekg",
    "supervised",
    [2],
    "train_new_model",
    5,
)

apes_application_instance = APES_Application(Logger, application_instance_metadata)

return_code, return_message = apes_application_instance.run()
Logger.info("Program", return_message)
