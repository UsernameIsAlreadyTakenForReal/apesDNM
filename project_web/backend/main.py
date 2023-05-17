from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify
import pandas as pd

import data

app = Flask(__name__)
CORS(app)


@app.route("/datatypes", methods=["GET", "POST"])
def getDatasets():
    return jsonify(data.datasets)


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    label_column = request.form["labelColumn"]
    df = pd.read_csv(file)

    x, y = df.shape

    return (
        "shape of file is " + str(x) + " and " + str(y) + "and label is " + label_column
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
