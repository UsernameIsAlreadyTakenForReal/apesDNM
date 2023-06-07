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
        return 1, "misc_functions.get_last_model found no suitable model", []


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
        "s_{solution_name}_d_{name_stub}_.{{6,8}}_.{{4,6}}{extension}".format(
            solution_name=solution_name,
            name_stub=app_instance_metadata.dataset_metadata.dataset_name_stub,
            extension=file_extension,
        )
    )

    if re.search(search_expression, model_filename) == None:
        return False
    else:
        return True
