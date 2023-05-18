from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify
import pandas as pd

import data
import path_to_pandas as ptp
import pathlib

app = Flask(__name__)
CORS(app)


@app.route("/datatypes", methods=["GET", "POST"])
def getDatasets():
    return jsonify(data.datasets)


@app.route("/unarchive", methods=["GET", "POST"])
def unarchive():
    file = request.files["file"]

    result = ptp.unarchive(file, False)
    print(result)

    return "ok"


@app.route("/upload", methods=["POST"])
def upload_file():
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
