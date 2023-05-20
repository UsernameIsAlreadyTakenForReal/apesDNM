from time import sleep
from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify
from io import BytesIO

import pandas as pd
import data
import patoolib
import os

import tempfile

app = Flask(__name__)
CORS(app)


@app.route("/datasets", methods=["GET", "POST"])
def getDatasets():
    return jsonify(data.datasets)


@app.route("/unarchive", methods=["GET", "POST"])
def unarchive():
    file = request.files["file"]

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.filename)
    file.save(temp_file_path)

    patoolib.extract_archive(temp_file_path, outdir="extracted")
    extracted_items = os.listdir("extracted")

    return len(extracted_items)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    file0 = request.files.get("file0", None)
    file1 = request.files.get("file1", None)

    if ".csv" in file0.filename:
        df = pd.read_csv(file0)
        x, y = df.shape
        string0 = "shape of " + file0.filename + " is (" + str(x) + ", " + str(y) + ")"

    if file1 is None:
        return string0, 201

    if ".csv" in file1.filename:
        df = pd.read_csv(file1)
        x, y = df.shape
        string1 = "shape of " + file1.filename + " is (" + str(x) + ", " + str(y) + ")"

    return string0 + ", " + string1, 201


if __name__ == "__main__":
    app.run(debug=True, port=5000)
