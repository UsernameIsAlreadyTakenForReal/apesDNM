import json


class Solution_Serializer:
    def __init__(self):
        _solution_name = ""
        _time_object_creation = ""

        time_train_start = ""
        time_train_end = ""
        time_train_total = ""

        time_test_start = ""
        time_test_end = ""
        time_test_total = ""

        dataset_name = ""
        dataset_full_size = ""
        dataset_train_size = ""
        dataset_test_size = ""
        dataset_val_size = ""
        dataset_run_size = ""

        used_test_function = ""
        used_train_function = ""

        device_used = ""
        accuracy = ""

        model_filename = ""

        plots_filenames = []

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
