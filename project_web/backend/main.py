from time import sleep

from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify

import os
import tempfile

from datetime import datetime


from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


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


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("message", "Connected to the WebSocket")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


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


# https://maxhalford.github.io/blog/flask-sse-no-deps/


@app.route("/testing", methods=["GET", "POST"])
def testing():
    socketio.emit("message", "Hello from /testing endpoint")
    socketio.emit("message", "Hello again from /testing endpoint")

    return str(datetime.now()) + " OK"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
