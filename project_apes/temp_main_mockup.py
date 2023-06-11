import sys

n = len(sys.argv)
print("Total number of arguments passed: ", n)
if n != 3:
    sys.exit(
        "Test correctly, please. Only 2 parameters (dataset, solution) at the moment"
    )

example_path = ""
dataset_metadata__dataset_name_stub = ""
solution_index = []

if int(sys.argv[1]) == 1:
    print("Running with dataset egk1")
    example_path = "../../Datasets/Dataset - ECG5000/Dataset - ECG5000"
    dataset_metadata__dataset_name_stub = "ekg1"
elif int(sys.argv[1]) == 2:
    print("Running with dataset ekg2")
    example_path = "../../Datasets/Dataset - ECG_Heartbeat/Dataset - ECG_Heartbeat.zip"
    dataset_metadata__dataset_name_stub = "ekg2"
else:
    print("whatev")
    sys.exit("Test correctly, please. Selection is [1, 2] at the moment")

if int(sys.argv[2]) == 1:
    solution_index = [1]
elif int(sys.argv[2]) == 2:
    solution_index = [2]
elif int(sys.argv[2]) == 3:
    solution_index = [1, 2]

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

from datetime import datetime

dataset_metadata__dataset_path = example_path
dataset_metadata__dataset_name_stub = dataset_metadata__dataset_name_stub
dataset_metadata__is_labeled = True
dataset_metadata__file_keyword_names = []
dataset_metadata__number_of_classes = 5
dataset_metadata__class_names = ["Normal", "R on T", "PVC", "SP", "UB"]
dataset_metadata__label_column_name = "target"
dataset_metadata__numerical_value_of_desired_label = 0
dataset_metadata__separate_train_and_test = False
dataset_metadata__percentage_of_split = [0.7]
dataset_metadata__shuffle_rows = True

application_metadata__app_instance_ID = datetime.now().strftime("%Y%m%d_%H%M%S")[2:]
application_metadata__display_dataFrames = False
application_metadata__application_mode = "run_one_solution"
application_metadata__dataset_origin = "existing_dataset"
application_metadata__dataset_category = "ekg"
application_metadata__solution_category = "ekg"
application_metadata__solution_nature = "supervised"
application_metadata__solution_index = solution_index
# application_metadata__model_origin = "train_new_model"
application_metadata__model_origin = "use_existing_model"
application_metadata__model_train_epochs = 5

dataset_metadata = Dataset_Metadata(
    Logger,
    dataset_metadata__dataset_path,
    dataset_metadata__dataset_name_stub,
    dataset_metadata__is_labeled,
    dataset_metadata__file_keyword_names,
    dataset_metadata__number_of_classes,
    dataset_metadata__class_names,
    dataset_metadata__label_column_name,
    dataset_metadata__numerical_value_of_desired_label,
    dataset_metadata__separate_train_and_test,
    dataset_metadata__percentage_of_split,
    dataset_metadata__shuffle_rows,
)
application_instance_metadata = Application_Instance_Metadata(
    Logger,
    dataset_metadata,
    application_metadata__app_instance_ID,
    application_metadata__display_dataFrames,
    application_metadata__application_mode,
    application_metadata__dataset_origin,
    application_metadata__dataset_category,
    application_metadata__solution_category,
    application_metadata__solution_nature,
    application_metadata__solution_index,
    application_metadata__model_origin,
    application_metadata__model_train_epochs,
)

apes_application_instance = APES_Application(Logger, application_instance_metadata)

return_code, return_message = apes_application_instance.run()
Logger.info("Program", return_message)
