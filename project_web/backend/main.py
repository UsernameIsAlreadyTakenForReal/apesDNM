import json
from random import randint
import threading
from time import sleep
from flask import Flask, Response, request
from flask_cors import CORS
from flask.json import jsonify

import os
import tempfile
import time

import sys
from io import StringIO

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
    sleep(5)

    cls()

    files = []

    for i in range(len(request.files)):
        if request.files.get(f"file{i}"):
            files.append(request.files.get(f"file{i}"))

    temp_dir = tempfile.mkdtemp()

    if len(files) == 0:
        return "request ok. no files to save"

    for file in files:
        file.save(os.path.join(temp_dir, file.filename))

    return "request ok. files saved at " + temp_dir


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
        print("variable x has changed!")


@app.route("/events")
def send_event(event_id):
    # print("data is", event_id)

    def generate_data():
        print("data is", event_id)
        yield f"data: {event_id}\n\n"

    return Response(generate_data(), mimetype="text/event-stream")


@app.route("/testingSSEs", methods=["GET", "POST"])
def mock_endpoint():
    data = "SOMETHING HERE MAN"
    send_event(data)
    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
