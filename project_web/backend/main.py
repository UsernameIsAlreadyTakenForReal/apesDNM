from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify
import os

app = Flask(__name__)
CORS(app)


def cls():
    os.system("cls" if os.name == "nt" else "clear")


@app.route("/datasets", methods=["GET", "POST"])
def getDatasets():
    import data

    return jsonify(data.datasets)


# https://medium.com/geekculture/how-to-a-build-real-time-react-app-with-server-sent-events-9fbb83374f90
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    cls()

    file0 = request.files.get("file0", None)
    file1 = request.files.get("file1", None)

    dataset = request.form.get("dataset", None)
    methods = request.form.get("methods", None)

    print("file0: " + file0.filename if file0 else "file0: None")
    print("file1: " + file1.filename if file1 else "file1: None")

    return "request ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
