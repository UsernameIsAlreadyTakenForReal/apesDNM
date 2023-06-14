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

app = Flask(__name__)
app.config["SECRET_KEY"] = "apesDNM"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")
CORS(app)

logger = Logger(socketio)

# TODO
# make sure to interrupt any perform_eda requests if upload_file is triggered
# add all necessary thing for archives
# add some error handling for frontend requests
# add some error handling for backend? if something happens, how to ensure backend runs for next requests?


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

        shutil.rmtree("images")
        os.makedirs("images")

        files = []

        logger.info("upload_file", "# of files sent is " + str(len(request.files)))

        for i in range(len(request.files)):
            if request.files.get(f"file{i}"):
                files.append(request.files.get(f"file{i}"))

                file = request.files.get(f"file{i}")

                logger.info("request", f"file{i}: " + file.filename)

        temp_dir = ""
        path = ""  # equivalent to len(files) == 0

        if len(files) == 0:
            logger.info("file_save", "files processing done. no files to save")
        else:
            temp_dir = tempfile.mkdtemp()
            for file in files:
                file.save(os.path.join(temp_dir, file.filename))
            logger.info(
                "file_save", "files processing done. files saved at " + temp_dir
            )

        # no files => np path
        if len(files) == 0:
            path = ""

        # more than one file => folder
        if len(files) > 1:
            path = temp_dir

        if len(files) == 1:
            for file in files:
                if "zip" in file.filename or "rar" in file.filename:
                    path = os.path.join(temp_dir, file.filename)
                else:
                    path = temp_dir

        results = "these are some results about the files you uploaded. there's also some plots"

        from apes_metadata_handler import Dataset_EDA

        dataset_EDA = Dataset_EDA(logger, path)
        files_dicts = dataset_EDA.perform_eda()

        return jsonify({"results": results, "path": path, "eda": files_dicts})
    except Exception as e:
        logger.info(
            "upload_file",
            "there was an error when running the function, please check input and try again\n"
            + "error was: "
            + str(e),
        )

        return (
            jsonify(
                {
                    "results": "the program ran into an eror. please check the console and try again."
                },
            ),
            400,
        )


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    try:
        print("upload_file() function called")

        logger.info("upload_file", "upload has been triggered")

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
        from apes_application import APES_Application

        # dataset metadata
        dataset_path = request.form.get("dataset_path", "")
        is_labeled = request.form.get("is_labeled", True)
        file_keyword_names = request.form.get("file_keyword_names", [])
        class_names = request.form.get("class_names", [])
        label_column_name = request.form.get("label_column_name", "target")
        numerical_value_of_desired_label = request.form.get(
            "numerical_value_of_desired_label", 0
        )
        desired_label = request.form.get("desired_label", "")  # ????
        separate_train_and_test = request.form.get("separate_train_and_test", False)
        percentage_of_split = request.form.get("percentage_of_split", 0.7)
        shuffle_rows = request.form.get("shuffle_rows", False)

        # application instance metadata
        display_dataFrames = request.form.get("display_dataFrames", False)
        print(display_dataFrames)  # displays False, not (False,) # ????
        application_mode = request.form.get("application_mode", "compare_solutions")
        dataset_origin = request.form.get("dataset_origin", "new_dataset")
        dataset_category = request.form.get("dataset_category", "ekg")
        solution_category = request.form.get("solution_category", "ekg")
        solution_nature = request.form.get("solution_nature", "supervised")
        solution_index = request.form.get("solution_index", [1])
        model_origin = request.form.get("model_origin", "train_new_model")
        model_train_epochs = request.form.get("model_train_epochs", 40)

        save_data = request.form.get("save_data", False)

        dataset_metadata = Dataset_Metadata(
            logger,
            dataset_path,
            is_labeled,
            file_keyword_names,
            class_names,
            label_column_name,
            numerical_value_of_desired_label,
            separate_train_and_test,
            percentage_of_split,
            shuffle_rows,
        )

        application_instance_metadata = Application_Instance_Metadata(
            logger,
            dataset_metadata,
            display_dataFrames,
            application_mode,
            dataset_origin,
            dataset_category,
            solution_category,
            solution_nature,
            solution_index,
            model_origin,
            model_train_epochs,
        )

        apes_application_instance = APES_Application(
            logger, application_instance_metadata
        )

        # ########################### application run ############################

        return_code, return_message = apes_application_instance.run()
        logger.info("Program", return_message)

        # ########################### plot processing ############################

        number_of_plots = random.randint(1, 5)
        logger.info("upload_file", "there are " + str(number_of_plots) + " plots")

        for _ in range(number_of_plots):
            plot()

        # ########################### creating results ###########################

        results = return_message

        result = {"results": results}
        return jsonify(result)

    except Exception as e:
        logger.info(
            "upload_file",
            "there was an error when running the function, please check input and try again\n"
            + "error was: "
            + str(e),
        )

        return (
            jsonify(
                {
                    "results": "the program ran into an eror. please check the console and try again."
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


# ############################################################################


# ################## in case of routing the std output #######################
# output = StringIO()
# sys.stdout = output
# print("...")
# console_info = output.getvalue()
# sys.stdout = sys.__stdout__
# result = {"results": results , "plots": plots, "console": console_info}
