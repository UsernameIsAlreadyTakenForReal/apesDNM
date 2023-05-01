from flask import Flask, request
from flask_cors import CORS
from flask.json import jsonify

app = Flask(__name__)
CORS(app)

@app.route('/datasets', methods=['GET'])
def getDatasets():
    datasets = [
        { "id": 1, "method": "NN de mana ca saracii" },
        { "id": 2, "method": "LSTM" },
        { "id": 3, "method": "CUDA" },
        { "id": 4, "method": "Some other method we copied from some other place" },
    ]
    datasets = jsonify(datasets)
    return datasets

@app.route('/datasets/<id>', methods=['GET', 'POST'])
def getDataset(id):
    id = request.args.get('id')
    print(id)
    return "ok"


if __name__ == "__main__":
    app.run(debug=True, port=5000)