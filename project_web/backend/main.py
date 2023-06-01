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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
