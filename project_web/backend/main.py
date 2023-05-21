from time import sleep
from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify

import pandas as pd
import numpy as np
from scipy.io import arff

import data
import patoolib
import os

import tempfile

app = Flask(__name__)
CORS(app)


def cls():
    os.system("cls" if os.name == "nt" else "clear")


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


def create_dataframe_from_file(file):
    return pd.read_csv(file)
    # file_extension = file.filename.rsplit(".", 1)[-1].lower()

    # if file_extension == "csv":
    #     return pd.read_csv(file)
    # elif file_extension == "arff":
    #     from scipy.io import arff

    #     data, meta = arff.loadarff(file)
    #     return pd.DataFrame(data)
    # elif file_extension == "npz":
    #     import numpy as np

    #     npz_file = np.load(file)
    #     data = {name: npz_file[name] for name in npz_file.files}
    #     return pd.DataFrame(data)
    # elif file_extension in ["xls", "xlsx"]:
    #     return pd.read_excel(file)
    # elif file_extension == "json":
    #     return pd.read_json(file)
    # elif file_extension == "h5":
    #     return pd.read_hdf(file)
    # elif file_extension == "parquet":
    #     return pd.read_parquet(file)
    # elif file_extension == "feather":
    #     return pd.read_feather(file)
    # elif file_extension == "pkl":
    #     return pd.read_pickle(file)
    # elif file_extension in ["txt", "dat"]:
    #     return pd.read_fwf(file)
    # elif file_extension in ["html", "htm"]:
    #     return pd.read_html(file)[0]
    # else:
    #     raise ValueError("Unsupported file type.")


def create_dataframe_from_file(file):
    return pd.read_csv(file)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    cls()

    file0 = request.files.get("file0", None)
    file1 = request.files.get("file1", None)

    print("file0 is " + file0.filename)

    # if ".csv" in file0.filename:
    df = pd.read_csv(file0)
    x, y = df.shape
    string0 = "shape of " + file0.filename + " is (" + str(x) + ", " + str(y) + ")"
    print("from me: " + string0)

    df0 = create_dataframe_from_file(file0)
    x, y = df0.shape
    string0 = "shape of " + file0.filename + " is (" + str(x) + ", " + str(y) + ")"
    print("from chatgpt: " + string0)

    # if ".zip" in file0.filename or ".rar" in file0.filename:
    #     return "you sent an archive", 400

    # if file1 is None:
    #     return string0, 201

    # if ".csv" in file1.filename:
    #     df = pd.read_csv(file1)
    #     x, y = df.shape
    #     string1 = "shape of " + file1.filename + " is (" + str(x) + ", " + str(y) + ")"

    # return string0 + ", " + string1, 201
    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
