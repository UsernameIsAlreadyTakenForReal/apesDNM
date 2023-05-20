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

    temp_dir = tempfile.gettempdir()  # Get the system's temporary directory
    temp_file_path = os.path.join(temp_dir, file.filename)
    file.save(temp_file_path)
    # return temp_file_path

    # temp_file = tempfile.NamedTemporaryFile(delete=False)
    # temp_file.write(file.read())
    # temp_file.close()

    # temp_folder = tempfile.mkdtemp()

    patoolib.extract_archive(temp_file_path, outdir="extracted")
    extracted_items = os.listdir("extracted")

    return len(extracted_items)


@app.route("/upload", methods=["POST"])
def upload_file():
    sleep(10)
    return

    file1 = request.files["file1"]
    file2 = request.files["file2"] or ""
    label_column = request.form["labelColumn"]
    string1 = ""
    string2 = ""

    print(file1.filename)
    if ".csv" in file1.filename:
        df = pd.read_csv(file1)
        x, y = df.shape
        string1 = "shape of file1 is " + str(x) + " and " + str(y)

    if ".csv" in file2.filename:
        df = pd.read_csv(file2)
        x, y = df.shape
        string2 = "shape of file2 is " + str(x) + " and " + str(y)

    return string1 + ", " + string2


if __name__ == "__main__":
    app.run(debug=True, port=5000)
