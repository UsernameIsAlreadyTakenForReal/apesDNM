from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify

import os
import tempfile

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

    files = []

    for i in range(len(request.files)):
        file_key = f"file{i}"
        file = request.files.get(file_key)
        if file:
            files.append(file)

    # print(files)

    temp_dir = tempfile.mkdtemp()

    for file in files:
        file.save(os.path.join(temp_dir, file.filename))

    return "request ok, files saved at " + temp_dir


if __name__ == "__main__":
    app.run(debug=True, port=5000)
