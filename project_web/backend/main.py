# ############################################################################

# class Output:
#     def __init__(self):
#         self._x = None

#     @property
#     def x(self):
#         return self._x

#     @x.setter
#     def x(self, value):
#         if self._x != value:
#             self._x = value
#             self.trigger_action()

#     def trigger_action(self):
#         print("variable x has changed!")

# https://maxhalford.github.io/blog/flask-sse-no-deps/

# ############################################################################

from datetime import datetime


class Logger:
    def _init_(self):
        self.message_in_queue = (
            "Nothing to print in Logger. This may mean something is wrong."
        )
        pass

    def info(self, sender, text_to_log):
        print(str(datetime.now()) + " -- " + str(sender) + " -- " + text_to_log)

    def print_info(self):
        print(self.message_in_queue)


############################################################################


from flask import Flask, request, send_from_directory
from flask_cors import CORS
from flask.json import jsonify

import os, os.path
import tempfile


app = Flask(__name__)
CORS(app)


def cls():
    os.system("cls" if os.name == "nt" else "clear")


@app.route("/datasets", methods=["GET", "POST"])
def getDatasets():
    import data

    return jsonify(data.datasets)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if (
        len(
            [
                name
                for name in os.listdir("images")
                if os.path.isfile(os.path.join("images", name))
            ]
        )
        > 10
    ):
        import shutil

        shutil.rmtree("images")
        os.makedirs("images")

    import sys
    from io import StringIO

    import uuid

    # changing output to variable so we can show it in the frontend
    output = StringIO()
    sys.stdout = output

    logger = Logger()

    info_message = "upload has been triggered"
    logger.info("upload_file", info_message)

    files = []

    info_message = "# of files sent is " + str(len(request.files))
    logger.info("upload_file", info_message)

    for i in range(len(request.files)):
        if request.files.get(f"file{i}"):
            files.append(request.files.get(f"file{i}"))

            file = request.files.get(f"file{i}")

            info_message = f"file{i}: " + file.filename
            logger.info("request", info_message)

    temp_dir = tempfile.mkdtemp()

    info_message = "temp file has been created: " + temp_dir
    logger.info("tempfile", info_message)

    from matplotlib import pyplot as plt

    plt.switch_backend("agg")
    plots = []

    # create plot
    plt.plot([1, 2, 3, 4])
    plt.ylabel("numbers")
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

    if len(files) == 0:
        info_message = "request ok. no files to save"
        logger.info("file_save", info_message)

    for file in files:
        file.save(os.path.join(temp_dir, file.filename))

    info_message = "request ok. files saved at " + temp_dir
    logger.info("file_save", info_message)

    console_info = output.getvalue()
    sys.stdout = sys.__stdout__

    results = "Run has been successful. Accuracy at 95% over 85 epochs."

    result = {"results": results, "plots": plots, "console": console_info}
    return jsonify(result)


@app.route("/images/<path:filename>")
def get_image(filename):
    return send_from_directory("images", filename)


@app.route("/testing", methods=["GET", "POST"])
def testing():
    return str(datetime.now()) + " OK"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
