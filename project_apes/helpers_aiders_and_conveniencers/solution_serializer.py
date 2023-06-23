import json
import platform


class Solution_Serializer:
    def __init__(self):
        _app_instance_ID = ""
        _solution_name = ""
        _time_object_creation = ""
        _platform = platform.system()

        _used_test_function = ""
        _used_train_function = ""

        time_train_start = ""
        time_train_end = ""
        time_train_total = ""
        train_epochs = ""

        time_test_start = ""
        time_test_end = ""
        time_test_total = ""

        dataset_name = ""
        dataset_full_size = ""
        dataset_train_size = ""
        dataset_test_size = ""
        dataset_val_size = ""
        dataset_run_size = ""

        device_used = ""
        accuracy = ""
        correct_normal_predictions = ""
        correct_abnormal_predictions = ""

        model_filename = ""

        plots_filenames = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
