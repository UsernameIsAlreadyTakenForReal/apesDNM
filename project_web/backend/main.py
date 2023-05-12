from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify
import pandas as pd

app = Flask(__name__)
CORS(app)


@app.route("/datatypes", methods=["GET", "POST"])
def getDatasets():
    datasets = [
        {"id": 1, "method": "EKG"},
        {"id": 2, "method": "RAL"},
        {"id": 3, "method": "IMG"},
        {"id": 4, "method": "I don't know my data type..."},
    ]
    datasets = jsonify(datasets)
    return datasets


@app.route("/datatype", methods=["GET", "POST"])
def getSingularDataset():
    id = request.args["id"]
    method = ""

    datasets = [
        {"id": 1, "method": "NN de mana ca saracii"},
        {"id": 2, "method": "LSTM"},
        {"id": 3, "method": "CUDA"},
        {"id": 4, "method": "Some other method we copied from some other place"},
    ]

    for item in datasets:
        if str(item["id"]) == str(id):
            found = True
            method = item["method"]

    if found == False:
        response = {"error": "could not find method"}
    else:
        # get data about method from database
        response = {
            "id": id,
            "method": method,
            "best_suited_for": "EKG",
            "best_accuracy": 92.1,
            "time_elapsed_per_epoch_in_secs": 15.2,
            "epochs": 40,
            "error": "",
        }

    return response


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    df = pd.read_csv(file)

    print(df.shape)

    return "File uploaded successfully"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
