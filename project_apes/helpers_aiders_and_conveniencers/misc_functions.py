def how_many_different_values_in_list(incoming_list):
    seen = []
    dupes = []
    for x in incoming_list:
        if x in seen:
            dupes.append(x)
        else:
            seen.append(x)

    return len(seen)


def get_last_model(Logger, solution_name, app_instance_metadata):
    import os
    import pathlib
    import re

    possible_models = []
    list_of_finds = []
    file_extension = ""

    match solution_name:
        case "ekg1":
            file_extension = ".pth"
        case "ekg2":
            file_extension = ".h5"

    for dirname, _, filenames in os.walk(
        app_instance_metadata.shared_definitions.project_model_root_path
    ):
        for filename in filenames:
            if (
                pathlib.Path(filename).suffix
                in app_instance_metadata.shared_definitions.supported_model_extensions
            ):
                possible_models.append(filename)

    search_expression = (
        "s_{solution_name}_d_{name_stub}_.{{6,8}}_.{{4,6}}{extension}".format(
            solution_name=solution_name,
            name_stub=app_instance_metadata.dataset_metadata.dataset_name_stub,
            extension=file_extension,
        )
    )

    for file in possible_models:
        x = re.search(search_expression, file)
        if x != None:
            list_of_finds.append(x.string)

    list_of_finds.sort()
    list_of_finds.reverse()

    if len(list_of_finds) != 0:
        new_path = ""
        new_path = get_full_path_of_given_model(
            list_of_finds[0],
            app_instance_metadata.shared_definitions.project_model_root_path,
        )

        if new_path == "":
            return (
                2,
                "misc_functions.get_last_model encountered a problem when creating absolute path for module",
                [],
            )

        info_message = f"List of model matches: {list_of_finds}"
        Logger.info("misc_functions.get_last_model", info_message)
        info_message = f"Returning the latest one as absolute path {new_path}."
        Logger.info("misc_functions.get_last_model", info_message)

        return (
            0,
            "misc_functions.get_last_model exited successfully",
            new_path,
        )
    else:
        return (
            1,
            "misc_functions.get_last_model found no suitable model. Please train a model first, or check the path to the models directory",
            [],
        )


def get_full_path_of_given_model(model_filename, project_model_root_path):
    import os

    incomplete_full_path = os.path.abspath(model_filename)
    path_elements = incomplete_full_path.split(os.sep)
    folder_structure_from_project_root_to_model_dir = project_model_root_path[2:]
    new_path = ""

    for i in range(len(path_elements)):
        if i != len(path_elements) - 1:
            new_path += path_elements[i] + "/"
        else:
            new_path += (
                folder_structure_from_project_root_to_model_dir + "/" + path_elements[i]
            )
            break
    return new_path


def get_full_path_of_given_plot_savefile(plot_filename, backend_images):
    import os

    incomplete_full_path = os.path.abspath(plot_filename)
    path_elements = incomplete_full_path.split(os.sep)
    folder_structure_from_project_root_to_model_dir = backend_images[2:]
    new_path = ""

    for i in range(len(path_elements)):
        if i != len(path_elements) - 1:
            new_path += path_elements[i] + "/"
        else:
            new_path += (
                folder_structure_from_project_root_to_model_dir + "/" + path_elements[i]
            )
            break
    return new_path


def get_plot_save_filename(plot_caption, solution, app_instance_metadata):
    from datetime import datetime
    import os

    filename = (
        "s_"
        + solution
        + "_d_"
        + app_instance_metadata.dataset_metadata.dataset_name_stub
    )
    filename += "__" + plot_caption + "__"
    filename += datetime.now().strftime("%Y%m%d_%H%M%S")
    filename += "." + app_instance_metadata.shared_definitions.plot_savefile_format

    file_absolute_path = (
        os.path.abspath(app_instance_metadata.shared_definitions.plot_savefile_location)
        + os.sep
        + filename
    )
    return filename, file_absolute_path


def model_filename_fits_expected_name(
    solution_name, app_instance_metadata, model_filename
):
    import re

    match solution_name:
        case "ekg1":
            file_extension = ".pth"
        case "ekg2":
            file_extension = ".h5"

    search_expression = (
        "^s_{solution_name}_d_{name_stub}_.{{6,8}}_.{{4,6}}{extension}".format(
            solution_name=solution_name,
            name_stub=app_instance_metadata.dataset_metadata.dataset_name_stub,
            extension=file_extension,
        )
    )

    if re.search(search_expression, model_filename) == None:
        return False
    else:
        return True


def write_to_solutions_runs_json_file(
    Logger, solution, serializer, app_instance_metadata
):
    import json
    import os

    json_file = f"s_{solution}.json"
    json_file_path = (
        app_instance_metadata.shared_definitions.project_solution_runs_path
        + os.sep
        + json_file
    )

    # because the json serializer doesn't do a good job (that or I'm too much of an idiot to figure out
    # why it writes literal string in a file in which \n should be treated as an \n, not as a "\n", but whatever)
    serialized_data = serializer.toJSON()
    serialized_data += "\n"
    serialized_data = serialized_data.replace("{", "    {")
    serialized_data = serialized_data.replace("}", "    }")
    serialized_data = serialized_data.replace('    "', '      "')

    if os.path.exists(json_file_path) == False:
        info_message = f"Runs file for solution {solution} does not exist. Creating"
        Logger.info("write_to_solutions_runs_json_file", info_message)

        initial_file_content = '{{\n  "solution": "{solution}",\n  "runs": [\n'.format(
            solution=solution
        )
        all_lines = initial_file_content + serialized_data
        all_lines += "  ]\n}\n"

    else:
        info_message = f"Runs file for solution {solution} exists. Appending"
        Logger.info("write_to_solutions_runs_json_file", info_message)
        file = open(json_file_path, "r")
        all_lines = file.readlines()
        file.close()

        all_lines[-3] = all_lines[-3].replace("}\n", "},\n")
        all_lines.insert(-2, serialized_data)

    file = open(json_file_path, "w")
    file.writelines(all_lines)
    file.close()


def get_accuracies_from_confusion_matrix(
    Logger, confusion_matrix, THRESHOLD_PER_CLASS, THRESHOLD_PER_TOTAL
):
    accuracy_per_class = []

    for i in range(len(confusion_matrix)):
        x = (confusion_matrix[i][i] * 100) / sum(confusion_matrix[i])
        accuracy_per_class.append(x)
        if x > THRESHOLD_PER_CLASS:
            info_message = f"Class {i} was predicted with good enough accuracy (> {THRESHOLD_PER_CLASS}%)"
            Logger.info(
                "misc_functions.get_accuracies_from_confusion_matrix", info_message
            )

    mean_total_accuracy = sum(accuracy_per_class) / len(accuracy_per_class)

    return accuracy_per_class, mean_total_accuracy


def assess_whether_to_save_model_as_best(
    Logger,
    app_instance_metadata,
    solution_name,
    model_filename,
    accuracy_per_class,
    mean_total_accuracy,
):
    import json
    from jsonpath_ng.ext import parse

    ### Section 1 -- get runs data and instances where the best model appears
    # first, the current model
    file_to_load = f"{app_instance_metadata.shared_definitions.project_solution_runs_path}/s_{solution_name}.json"
    file = open(file_to_load)
    shared_data = json.load(file)
    file.close()

    list_of_current_model_appInstance_matches = []
    jsonpath_expression_get_appInstances_of_current_model = parse(
        f"$.runs[?(@.model_filename=='{model_filename}')]._app_instance_ID"
    )

    for match in jsonpath_expression_get_appInstances_of_current_model.find(
        shared_data
    ):
        list_of_current_model_appInstance_matches.append(match.value)

    info_message = f"list_of_current_model_appInstance_matches: {list_of_current_model_appInstance_matches}"
    Logger.info("misc_functions.asses_whether_to_save_model_as_best", info_message)
    list_of_current_model_meanTotalAccuracy = []
    list_of_current_model_accuracyPerClass = []

    for occurence in list_of_current_model_appInstance_matches:
        jsonpath_expression_get_meanTotalAccuracy_of_current_model = parse(
            f"$.runs[?(@._app_instance_ID=='{occurence}')].mean_total_accuracy"
        )
        for match in jsonpath_expression_get_meanTotalAccuracy_of_current_model.find(
            shared_data
        ):
            list_of_current_model_meanTotalAccuracy.append(match.value)

        jsonpath_expression_get_accuracyPerClass_of_current_model = parse(
            f"$.runs[?(@._app_instance_ID=='{occurence}')].accuracy_per_class"
        )
        for match in jsonpath_expression_get_accuracyPerClass_of_current_model.find(
            shared_data
        ):
            list_of_current_model_accuracyPerClass.append(match.value)

    info_message = f"list_of_current_model_meanTotalAccuracy: {list_of_current_model_meanTotalAccuracy}"
    Logger.info("misc_functions.asses_whether_to_save_model_as_best", info_message)
    info_message = f"list_of_current_model_accuracyPerClass: {list_of_current_model_accuracyPerClass}"
    Logger.info("misc_functions.asses_whether_to_save_model_as_best", info_message)

    ## then, the best model to date. Taken directly from the static_definitions json to avoid a huge match case statement
    file_to_load = "apes_static_definitions.json"
    file = open(file_to_load)
    static_definitions = json.load(file)
    file.close()

    field_search = "project_solution_ekg1_d_ekg1_best_model_filename"
    field_search = f"project_solution_{solution_name}_d_{app_instance_metadata.dataset_metadata.dataset_name_stub}_best_model_filename"
    model_filename = static_definitions[field_search]

    list_of_best_model_appInstance_matches = []

    jsonpath_expression_get_appInstances_of_best_model = parse(
        f"$.runs[?(@.model_filename=='{model_filename}')]._app_instance_ID"
    )

    for match in jsonpath_expression_get_appInstances_of_best_model.find(shared_data):
        list_of_best_model_appInstance_matches.append(match.value)

    info_message = f"list_of_best_model_appInstance_matches: {list_of_best_model_appInstance_matches}"
    Logger.info("misc_functions.asses_whether_to_save_model_as_best", info_message)
    list_of_best_model_meanTotalAccuracy = []
    list_of_best_model_accuracyPerClass = []

    for occurence in list_of_best_model_appInstance_matches:
        print(occurence)
        jsonpath_expression_get_meanTotalAccuracy_of_best_model = parse(
            f"$.runs[?(@._app_instance_ID=='{occurence}')].mean_total_accuracy"
        )
        for match in jsonpath_expression_get_meanTotalAccuracy_of_best_model.find(
            shared_data
        ):
            list_of_best_model_meanTotalAccuracy.append(match.value)

        jsonpath_expression_get_accuracyPerClass_of_best_model = parse(
            f"$.runs[?(@._app_instance_ID=='{occurence}')].accuracy_per_class"
        )
        for match in jsonpath_expression_get_accuracyPerClass_of_best_model.find(
            shared_data
        ):
            list_of_best_model_accuracyPerClass.append(match.value)

    info_message = (
        f"list_of_best_model_meanTotalAccuracy: {list_of_best_model_meanTotalAccuracy}"
    )
    Logger.info("misc_functions.asses_whether_to_save_model_as_best", info_message)
    info_message = (
        f"list_of_best_model_accuracyPerClass: {list_of_best_model_accuracyPerClass}"
    )
    Logger.info("misc_functions.asses_whether_to_save_model_as_best", info_message)

    ### Section 2 -- compare with the current model
    if (
        compare_model_performance(
            Logger,
            solution_name,
            list_of_best_model_meanTotalAccuracy,
            list_of_best_model_accuracyPerClass,
            list_of_current_model_meanTotalAccuracy,
            list_of_current_model_accuracyPerClass,
        )
        == False
    ):
        return

    ### Section 3 -- save model as best model
    info_message = f"Saving model {model_filename} as best model"
    Logger.info("misc_functions.asses_whether_to_save_model_as_best", info_message)
    file_to_load = "apes_static_definitions.json"
    file = open(file_to_load)
    static_definitions = json.load(file)
    file.close()
    static_definitions[field_search] = model_filename
    file = open(file_to_load, "w")
    json.dump(static_definitions, file)


def compare_model_performance(
    Logger,
    solution,
    list_of_best_model_meanTotalAccuracy,
    list_of_best_model_accuracyPerClass,
    list_of_current_model_meanTotalAccuracy,
    list_of_current_model_accuracyPerClass,
):
    for best_model_meanTotalAccuracy in list_of_best_model_meanTotalAccuracy:
        if list_of_current_model_meanTotalAccuracy[0] < best_model_meanTotalAccuracy:
            info_message = "mean_total_accuracy of current model is lower than a previous instance of the best model. Exiting"
            Logger.info(
                "misc_functions.asses_whether_to_save_model_as_best", info_message
            )
            return False

    # this is pretty shitty and also good. It forces EVERY class to be good
    for best_model_accuracyPerClass in list_of_best_model_accuracyPerClass:
        for i in range(len(best_model_accuracyPerClass)):
            if (
                list_of_current_model_accuracyPerClass[0][i]
                < best_model_accuracyPerClass[i]
            ):
                info_message = "class accuracy of current model for a certain class is lower than a previous instance of the best model. Exiting"
                Logger.info(
                    "misc_functions.asses_whether_to_save_model_as_best",
                    info_message,
                )
                return False

    return True
