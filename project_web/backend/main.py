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


@app.route("/datatypes", methods=["GET", "POST"])
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

    file = request.files["file"]
    label_column = request.form["labelColumn"]
    print(file.filename)
    if ".csv" in file.filename:
        df = pd.read_csv(file)

        x, y = df.shape

        return (
            "shape of file is "
            + str(x)
            + " and "
            + str(y)
            + " and label is "
            + label_column
        )
    else:
        return "file type not yet supported"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
