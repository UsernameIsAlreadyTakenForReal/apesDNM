from time import sleep
from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify

import pandas as pd
import numpy as np
from scipy.io import arff
import rarfile

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
    file_extension = file.filename.rsplit(".", 1)[-1].lower()

    if file_extension == "csv":
        return pd.read_csv(file)

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


def unarchive(file):
    rar = rarfile.RarFile(file)

    folder_name = ""

    for file_info in rar.infolist():
        if file_info.isdir():
            folder_name = file_info.filename

    folder_files = []
    for file_info in rar.infolist():
        if not file_info.isdir() and file_info.filename.startswith(folder_name):
            folder_files.append(file_info.filename)

    return folder_files


# https://medium.com/geekculture/how-to-a-build-real-time-react-app-with-server-sent-events-9fbb83374f90
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    cls()

    file0 = request.files.get("file0", None)
    file1 = request.files.get("file1", None)

    print("file0: " + file0.filename if file0 else "file0: None")
    print("file1: " + file1.filename if file1 else "file1: None")

    if file0 and not file1 and file0.filename.endswith(".rar"):
        files = unarchive(file0)
    elif file0 and file1:
        files = [file0, file1]
    else:
        files = [file0]

    print(files)

    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
