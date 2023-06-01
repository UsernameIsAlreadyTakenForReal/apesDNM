from time import sleep

from flask import Flask, Response, request
from flask_cors import CORS
from flask.json import jsonify
from flask_sse import sse

import os
import tempfile


app = Flask(__name__)
app.register_blueprint(sse, url_prefix="/stream")
# CORS(app)


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


@app.route("/stream")
def send_event(event_id):
    print("from send_event()")

    def generate_data():
        print("from generate_data()")
        yield f"data: {event_id}\n\n"

    yield Response(generate_data(), mimetype="text/event-stream")


# https://maxhalford.github.io/blog/flask-sse-no-deps/


# @app.route("/stream")
# def stream(data):
#     print("from stream()")

#     def event_stream():
#         print("from event_stream()")
#         yield "data: Update 1\n\n"
#         sleep(1)

#         yield "data: Update 2\n\n"
#         sleep(1)

#         yield "data: Update 3\n\n"
#         sleep(1)

#         # After sending all messages, return the final response
#         yield "data: 200 OK\n\n"

#     return Response(event_stream(), mimetype="text/event-stream")


@app.route("/testingSSEs", methods=["GET", "POST"])
def mock_endpoint():
    send_event("testing...")
    sleep(5)

    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
