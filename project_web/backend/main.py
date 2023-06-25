from gevent import monkey

monkey.patch_all()

from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
from flask.json import jsonify
from datetime import datetime

from matplotlib import pyplot as plt
import uuid

plt.switch_backend("agg")

import os, os.path
import tempfile
import random

import sys

sys.path.append("../../project_apes/helpers_aiders_and_conveniencers")
sys.path.append("../../project_apes")

from logger import Logger
from apes_application import APES_Application
from apes_static_definitions import Shared_Definitions

from apes_metadata_handler import *


app = Flask(__name__)
app.config["SECRET_KEY"] = "apesDNM"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")
CORS(app)

logger = Logger(socketio)
apes_application_instance = APES_Application(logger)
shared_definitions = Shared_Definitions()


def cls():
    os.system("cls" if os.name == "nt" else "clear")


def plot():
    number_of_points = random.randint(5, 15)
    points_x = []
    points_y = []

    for _ in range(number_of_points):
        points_x.append(random.randint(0, 100))
        points_y.append(random.randint(0, 100))

    plt.clf()

    plt.plot(points_x, points_y)

    plt.ylabel(str(number_of_points) + " random numbers")
    plt.xlabel(str("just as many random numbers"))

    plot_title = "a plot with " + str(number_of_points) + " random numbers"
    plt.title(plot_title)

    filename = str(uuid.uuid4()) + ".png"

    info_message = "plot filename is " + filename
    logger.info("matplotlib", info_message)

    figure_path_png = os.path.join("images", filename)

    plt.savefig(figure_path_png)

    info_message = (
        "Created picture in folder"
        + " || "
        + str(figure_path_png)
        + " || "
        + str(plot_title)
    )
    logger.info("upload_file", info_message)


@app.route("/datasets", methods=["GET", "POST"])
def get_datasets():
    print("get_datasets() function called")

    import data

    return jsonify(data.datasets)


@app.route("/perform_eda", methods=["GET", "POST"])
def perform_eda():
    try:
        print("perform_eda() function called")

        import shutil
        import platform

        shutil.rmtree("images")
        os.makedirs("images")

        files = []
        temp_dir = ""
        path = ""

        logger.info("perform_eda", "# of files sent is " + str(len(request.files)))

        for i in range(len(request.files)):
            if request.files.get(f"file{i}"):
                file = request.files.get(f"file{i}")

                files.append(file)

                logger.info("request", f"file{i}: " + file.filename)

        if len(files) == 0:
            logger.info("file_save", "files processing done. no files sent by user")
            logger.info(
                "file_save", "checking if existing dataset category has been selected"
            )

            dataset_name_stub = request.form.get("dataset_name_stub", None)

            if dataset_name_stub is None:
                logger.info("file_save", "no dataset selected. exiting eda")
                return jsonify({"results": "there was nothing to perform eda on"}), 400
            else:
                if dataset_name_stub == "ekg1":
                    logger.info("file_save", "ekg1 dataset selected")
                    if platform.system() == "Windows":
                        path = "../../../Datasets/Dataset - ECG5000"
                    else:
                        path = "/ebs_data/project_datasets/d_ekg1"

                if dataset_name_stub == "ekg2":
                    logger.info("file_save", "ekg2 dataset selected")
                    if platform.system() == "Windows":
                        path = "../../../Datasets/Dataset - ECG_Heartbeat/Dataset - ECG_Heartbeat"
                    else:
                        path = "/ebs_data/project_datasets/d_ekg2"

        else:
            temp_dir = tempfile.mkdtemp()
            for file in files:
                file.save(os.path.join(temp_dir, file.filename))
            logger.info(
                "file_save", "files processing done. files saved at " + temp_dir
            )

        if path == "":
            path = temp_dir

        if len(files) == 1 and (
            "zip" in files[0].filename or "rar" in files[0].filename
        ):
            # unarchive
            import patoolib, pathlib

            archive_path = os.path.join(temp_dir, files[0].filename)

            new_path = archive_path.replace(pathlib.Path(archive_path).suffix, "")
            if os.path.exists(new_path) == False:
                os.mkdir(new_path)
            patoolib.extract_archive(archive_path, outdir=new_path)

            os.remove(archive_path)

            path = new_path
            logger.info("unarchive", "new path is " + path)

        results = "above you will find some results about how the processing went. check below for individual files' details and plots"

        # from apes_EDA_handler import Dataset_EDA

        return_code, return_message, files_dicts, dataset_category = apes_application_instance.run_EDA(
            path,
            shared_definitions
        )

        # dataset_EDA = Dataset_EDA(logger, path)
        # files_dicts = dataset_EDA.perform_eda()

        print("dataset category issssssssssssssssssss10: " + dataset_category)

        return jsonify({"results": return_message, "path": path, "eda": files_dicts, "dataset_category": dataset_category})
    except Exception as e:
        logger.info(
            "perform_eda",
            "there was an error when running the function, please check input and try again\n"
            + "error was: "
            + str(e),
        )

        return (
            jsonify(
                {
                    "results": "the program ran into an error. please check the console and try again."
                },
            ),
            400,
        )


@app.route("/upload", methods=["GET", "POST"])
def run_apesdnm():
    test_run = request.form.get("test_run", "")
    if test_run:
        import shutil

        shutil.rmtree("images")
        os.makedirs("images")

        dataset_path = request.form.get("dataset_path", "")
        print(dataset_path)

        return_code, return_message = mockup_test_run(logger, dataset_path)

        results = return_message

        result = {"results": results}
        return jsonify(result)

    try:
        print("run_apesdnm() function called")

        # ###################### cleaning-up images folder #######################
        clear_images = request.form.get("clear_images", True)
        if clear_images:
            import shutil

            shutil.rmtree("images")
            os.makedirs("images")

        # ########################### files processing ###########################

        # ########################## gathering metadata ##########################

        from apes_metadata_handler import (
            Dataset_Metadata,
            Application_Instance_Metadata,
        )

        # 0 -- See what the system flow is (comes from existing dataset or it is a new one)
        system_flow = request.form.get("system_flow", "using_a_previously_existing_dataset")

        # 1 -- Fill dataset metadata (+ dataset_category)
        # If using an existing dataset, get the dataset_metadata from _desc.json
        if system_flow == "using_a_previously_existing_dataset":
            logger.info("run_apesdnm()", "using_a_previously_existing_dataset")
            import json
            import platform

            dataset_name_stub = request.form.get("dataset_name_stub", "")

            if platform.system() == "Windows":
                file_path = f"../../project_apes/datasets/d_{dataset_name_stub}_description.json"
                info_message = f"Importing known dataset descriptor from {file_path}"
                logger.info("run_apesdnm()", info_message)
                file = open(f"../../project_apes/datasets/d_{dataset_name_stub}_description.json")
            else:
                file_path = f"/ebs_data/project_datasets/d_{dataset_name_stub}/d_{dataset_name_stub}_description.json"
                info_message = f"Importing known dataset descriptor from {file_path}"
                logger.info("run_apesdnm()", info_message)
                file = open(f"/ebs_data/project_datasets/d_{dataset_name_stub}/d_{dataset_name_stub}_description.json")

            data = json.load(file)

            if platform.system() == "Windows":
                dataset_metadata__dataset_path = data["dataset_path_windows"]
            else:
                dataset_metadata__dataset_path = data["dataset_path_linux"]
            dataset_metadata__dataset_name_stub = dataset_name_stub # repetition to have all of them here in order
            dataset_metadata__is_labeled = True if data["is_labeled"] == "True" else False
            dataset_metadata__file_keyword_names = data["file_keyword_names"]
            dataset_metadata__number_of_classes = int(data["number_of_classes"])
            dataset_metadata__class_names = data["class_names"]
            dataset_metadata__label_column_name = data["label_column_name"]
            dataset_metadata__numerical_value_of_desired_label = int(data["numerical_value_of_desired_label"])
            dataset_metadata__separate_train_and_test = True if data["separate_train_and_test"] == "True" else False
            dataset_metadata__percentage_of_split = [float(data["percentage_of_split"])]
            dataset_metadata__shuffle_rows = True if data["shuffle_rows"] == "True" else False

            application_metadata__dataset_category = data["dataset_category"]
            application_metadata__dataset_origin = "existing_dataset"

        # If this is a new dataset, fill from the frontend form and, if requested, save the dataset description + files
        else:
            # we need these first few ones to keep a nice order when defining the metadata
            logger.info("run_apesdnm()", "this_is_an_absolutely_new_dataset")
            dataset_identifier = request.form.get("dataset_identifier", "1108")
            dataset_category = request.form.get("dataset_category", "ekg")
            print(dataset_category)
            print(dataset_identifier)
            class_names = request.form.get("class_names", []).split(",")
            number_of_classes = len(class_names)

            dataset_metadata__dataset_path = request.form.get("dataset_path", "")
            dataset_metadata__dataset_name_stub = dataset_category + dataset_identifier
            dataset_metadata__is_labeled = True if request.form.get("is_labeled", True) == "yes" else False
            dataset_metadata__file_keyword_names = request.form.get("file_keyword_names", [])
            dataset_metadata__number_of_classes = number_of_classes
            dataset_metadata__class_names = class_names
            dataset_metadata__label_column_name = request.form.get("label_column_name", "target")
            dataset_metadata__numerical_value_of_desired_label = int(request.form.get("numerical_value_of_desired_label", 0))
            dataset_metadata__separate_train_and_test = True if request.form.get("separate_train_and_test", False) == "True" else False
            dataset_metadata__percentage_of_split = [float(request.form.get("percentage_of_split", 0.7))]
            dataset_metadata__shuffle_rows = True if request.form.get("shuffle_rows", False) == "True" else False

            application_metadata__dataset_category = dataset_category
            if request.form.get("is_labeled") == "idk": ## discerner
                logger.info("run_apesdnm()", "Discerner will be called")
                application_metadata__dataset_category = "N/A"
            application_metadata__dataset_origin = "new_dataset"

            print(request.form.get("save_data"))
            print(type(request.form.get("save_data")))
            save_data = True if request.form.get("save_data", False) == "True" else False

            if save_data == True:
                logger.info("run_apesdnm()", "saving dataset for later use")
                dataset_metadata_dict = {
                    "dataset_name_stub": dataset_metadata__dataset_name_stub,
                    "dataset_path_linux": dataset_metadata__dataset_path if platform.system() == "Linux" else "",
                    "dataset_path_windows": dataset_metadata__dataset_path if platform.system() == "Windows" else "",

                    "is_labeled": dataset_metadata__is_labeled,
                    "number_of_classes": dataset_metadata__number_of_classes,
                    "class_names": dataset_metadata__class_names,
                    "label_column_name": dataset_metadata__label_column_name,
                    "numerical_value_of_desired_label": dataset_metadata__numerical_value_of_desired_label,
                    "separate_train_and_test": dataset_metadata__separate_train_and_test,
                    "percentage_of_split": dataset_metadata__percentage_of_split,
                    "shuffle_rows": dataset_metadata__shuffle_rows,
                }

                print(jsonify(dataset_metadata_dict))
                # TODO: save all of above in json, and save files at path in the ./files subdirectory
                pass


        # 1 -- Fill application instance metadata (+ dataset_category)
        solution_indexes = request.form.get("solution_index", [1])
        solution_indexes_list = solution_indexes.split(",")

        application_metadata__app_instance_ID = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )[2:]
        application_metadata__display_dataFrames = request.form.get("display_dataFrames", False)
        # print(display_dataFrames)  # displays False, not (False,) # ????
        application_metadata__application_mode = request.form.get("application_mode", "compare_solutions")
        application_metadata__dataset_origin = application_metadata__dataset_origin
        application_metadata__dataset_category = application_metadata__dataset_category
        application_metadata__solution_category = request.form.get("solution_category", "ekg")
        application_metadata__solution_nature = request.form.get("solution_nature", "supervised")
        application_metadata__solution_index = [int(x) for x in solution_indexes_list]
        application_metadata__model_origin = request.form.get("model_origin", "train_new_model")
        application_metadata__model_train_epochs = request.form.get("model_train_epochs", 5)


        dataset_metadata = Dataset_Metadata(
            logger,
            dataset_metadata__dataset_path.strip(),
            dataset_metadata__dataset_name_stub.strip(),
            dataset_metadata__is_labeled,
            dataset_metadata__file_keyword_names,
            dataset_metadata__number_of_classes,
            dataset_metadata__class_names,
            dataset_metadata__label_column_name.strip(),
            dataset_metadata__numerical_value_of_desired_label,
            dataset_metadata__separate_train_and_test,
            dataset_metadata__percentage_of_split,
            dataset_metadata__shuffle_rows,
        )

        application_instance_metadata = Application_Instance_Metadata(
            logger,
            shared_definitions,
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

        # apes_application_instance = APES_Application(
        #     logger, application_instance_metadata
        # )

        apes_application_instance.update_app_instance_metadata(
            application_instance_metadata
        )

        # ########################### application run ############################

        return_code, return_message = apes_application_instance.run()
        logger.info("apesdnm", return_message)

        # ########################### plot processing ############################

        # number_of_plots = random.randint(1, 5)
        # logger.info("run_apesdnm", "there are " + str(number_of_plots) + " plots")

        # for _ in range(number_of_plots):
        #     plot()

        # ########################### creating results ###########################

        results = return_message

        result = {"results": results}
        return jsonify(result)

    except Exception as e:
        logger.info(
            "run_apesdnm",
            "there was an error when running the function, please check input and try again\n"
            + "error was: "
            + str(e),
        )

        return (
            jsonify(
                {
                    "results": "the program ran into an error. please check the console and try again."
                }
            ),
            400,
        )


@app.route("/images/<path:filename>")
def get_image(filename):
    print("file " + filename + " requested")
    return send_from_directory("images", filename)


@app.route("/testing", methods=["GET", "POST"])
def testing():
    return str(datetime.now()) + " OK"


if __name__ == "__main__":
    cls()
    socketio.run(app, debug=True, port=5000)


def mockup_test_run(Logger, path):
    dataset_metadata__dataset_name_stub = "ekg1"
    solution_index = [2]

    init_message = "Begin"
    Logger.info("Program", init_message)

    # from apes_application import APES_Application

    init_message = "Imported APES_Application"
    Logger.info("Program", init_message)

    init_message = "Imported apes_metadata_handler.*"
    Logger.info("Program", init_message)

    from datetime import datetime

    dataset_metadata__dataset_path = path
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
    application_metadata__model_origin = "train_new_model"
    # application_metadata__model_origin = "use_existing_model"
    application_metadata__model_train_epochs = 50

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
        shared_definitions,
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

    apes_application_instance = APES_Application(Logger)

    # return_code, return_message = apes_application_instance.run_EDA()
    # Logger.info("Program.run_EDA()", return_message)

    apes_application_instance.update_app_instance_metadata(
        application_instance_metadata
    )

    return_code, return_message = apes_application_instance.run()
    Logger.info("Program.run()", return_message)

    return return_code, return_message


# ############################################################################


# ################## in case of routing the std output #######################
# output = StringIO()
# sys.stdout = output
# print("...")
# console_info = output.getvalue()
# sys.stdout = sys.__stdout__
# result = {"results": results , "plots": plots, "console": console_info}

# METHODS

def jsonify_dataset_metadata(dataset_metadata_dict): {
}