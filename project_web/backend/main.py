from gevent import monkey

monkey.patch_all()

from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
from flask.json import jsonify
from datetime import datetime
from io import StringIO

import os, os.path
import tempfile
import gevent

import sys

sys.path.insert(1, "../../project_apes/helpers_aiders_and_conveniencers")
from logger import Logger

app = Flask(__name__)
app.config["SECRET_KEY"] = "apesDNM"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")
CORS(app)

logger = Logger(socketio)

# ############################################################################


class Output:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if self._x != value:
            self._x = value
            self.trigger_action()

    def trigger_action(self):
        print("value changed!")
        gevent.spawn(socketio.emit("console", str(self._x), broadcast=True))


# ############################################################################


# class Logger:
#     def _init_(self):
#         self.message_in_queue = (
#             "Nothing to print in Logger. This may mean something is wrong."
#         )
#         pass

#     def info(self, sender, text_to_log):
#         message = str(datetime.now()) + " -- " + str(sender) + " -- " + text_to_log
#         gevent.spawn(socketio.emit("console", str(message), broadcast=True))
#         print(message)
#         # time.sleep(2)

#     def print_info(self):
#         print(self.message_in_queue)


# ############################################################################


def cls():
    os.system("cls" if os.name == "nt" else "clear")


@app.route("/datasets", methods=["GET", "POST"])
def getDatasets():
    import data

    print("getDatasets() function called")
    return jsonify(data.datasets)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    print("upload_file() function called")

    # ########################################################################
    # ########################### changing output ############################
    # ########################################################################

    output = StringIO()
    sys.stdout = output

    # ########################################################################
    # ########################### init-ing logger ############################
    # ########################################################################

    # logger = Logger()

    info_message = "upload has been triggered"
    logger.info("upload_file", info_message)

    sys.path.insert(1, "../../project_apes/helpers_aiders_and_conveniencers")
    from testing_broadcast import test_singleton

    test_singleton()

    # return "ok"

    # ########################################################################
    # ###################### cleaning-up images folder #######################
    # ########################################################################

    import shutil

    shutil.rmtree("images")
    os.makedirs("images")

    # ########################################################################
    # ########################### files processing ###########################
    # ########################################################################

    files = []

    info_message = "# of files sent is " + str(len(request.files))
    logger.info("upload_file", info_message)

    for i in range(len(request.files)):
        if request.files.get(f"file{i}"):
            files.append(request.files.get(f"file{i}"))

            file = request.files.get(f"file{i}")

            info_message = f"file{i}: " + file.filename
            logger.info("request", info_message)

    temp_dir = ""

    if len(files) == 0:
        info_message = "files processing done. no files to save"
        logger.info("file_save", info_message)
    else:
        temp_dir = tempfile.mkdtemp()
        for file in files:
            file.save(os.path.join(temp_dir, file.filename))
        info_message = "files processing done. files saved at " + temp_dir
        logger.info("file_save", info_message)

    # ########################################################################
    # ########################## gathering metadata ##########################
    # ########################################################################

    #
    dataset_path = temp_dir
    is_labeled = request.form.get("input_name", True)
    file_keyword_names = request.form.get("file_keyword_names", [])
    class_names = request.form.get("class_names", [])
    label_column_name = request.form.get("class_names", "target")
    numerical_value_of_desired_label = request.form.get(
        "numerical_value_of_desired_label", 0
    )
    desired_label = request.form.get("desired_label", "")  # ????
    separate_train_and_test = request.form.get("separate_train_and_test", False)
    percentage_of_split = request.form.get("percentage_of_split", 0.7)
    shuffle_rows = request.form.get("shuffle_rows", False)

    # application instance metadata
    display_dataFrames = request.form.get("display_dataFrames", False)
    application_mode = request.form.get("application_mode", "compare_solutions")
    dataset_origin = request.form.get("dataset_origin", "new_dataset")
    dataset_category = request.form.get("dataset_category", "ekg")
    solution_category = request.form.get("solution_category", "ekg")
    solution_nature = request.form.get("solution_nature", "supervised")
    solution_index = request.form.get("solution_index", [1])
    model_origin = request.form.get("model_origin", "train_new_model")
    model_train_epochs = request.form.get("model_train_epochs", 40)

    saveData = request.form.get("saveData", False)

    # ########################################################################
    # ########################### plot processing ############################
    # ########################################################################

    from matplotlib import pyplot as plt
    import uuid

    plt.switch_backend("agg")
    plots = []

    # clears the current plot
    plt.clf()

    # create plot
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.ylabel("more numbers")
    # create filename
    filename = str(uuid.uuid4()) + ".png"
    # log info
    info_message = "plot filename is " + filename
    logger.info("matplotlib", info_message)
    # create path and save
    figure_path_png = os.path.join("images", filename)
    plt.savefig(figure_path_png)
    # add to array of plots
    plots.append(figure_path_png)

    info_message = "exiting endpoint"
    logger.info("upload_file", info_message)

    console_info = output.getvalue()
    sys.stdout = sys.__stdout__

    # ########################################################################
    # ########################### creating results ###########################
    # ########################################################################

    results = "run has been successful. for detailed steps, check the console."

    result = {"results": results, "plots": plots, "console": console_info}
    return jsonify(result)


@app.route("/images/<path:filename>")
def get_image(filename):
    return send_from_directory("images", filename)


@app.route("/testing", methods=["GET", "POST"])
def testing():
    return str(datetime.now()) + " OK"


if __name__ == "__main__":
    # app.run(debug=True, port=5000)

    cls()
    socketio.run(app, debug=True, port=5000)
