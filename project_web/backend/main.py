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


from flask import Flask, request, send_file
from flask_cors import CORS
from flask.json import jsonify

import os
import tempfile

from datetime import datetime

app = Flask(__name__)
CORS(app)


def cls():
    os.system("cls" if os.name == "nt" else "clear")


def print_data():
    print("hello there")


@app.route("/datasets", methods=["GET", "POST"])
def getDatasets():
    import data

    return jsonify(data.datasets)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    import sys
    from io import StringIO

    # changing output to variable so we can show it in the frontend
    output = StringIO()
    sys.stdout = output

    logger = Logger()

    logger.info("upload_file", "hello from logger")

    files = []

    print("len of files sent is " + str(len(request.files)))

    for i in range(len(request.files)):
        if request.files.get(f"file{i}"):
            files.append(request.files.get(f"file{i}"))

    temp_dir = tempfile.mkdtemp()

    # from matplotlib import pyplot as plt

    # plt.switch_backend("agg")
    plots = []

    # plt.plot([1, 2, 3, 4])
    # plt.ylabel("some numbers")
    # figure_path_png = os.path.join(temp_dir, "figure" + str(len(plots)) + ".png")
    # plt.savefig(figure_path_png)
    # plots.append(figure_path_png)

    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    # figure_path_png = os.path.join(temp_dir, "figure" + str(len(plots)) + ".png")
    # plt.savefig(figure_path_png)
    # plots.append(figure_path_png)

    console_info = output.getvalue()
    sys.stdout = sys.__stdout__

    result = {"plots": plots, "results": console_info}
    return jsonify(result)

    if len(files) == 0:
        return "request ok. no files to save"

    for file in files:
        file.save(os.path.join(temp_dir, file.filename))

    return "request ok. files saved at " + temp_dir


@app.route("/testing", methods=["GET", "POST"])
def testing():
    return str(datetime.now()) + " OK"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
