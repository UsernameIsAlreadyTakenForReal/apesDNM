import json
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

@app.route('/datasets', methods=['GET'])
def getDatasets():
    return [
        { id: 1, "method": "NN de mana ca saracii" },
        { id: 2, "method": "LTSM" },
        { id: 3, "method": "CUDA" },
        { id: 4, "method": "Some other method we copied from some other place" },
    ]


if __name__ == "__main__":
    app.run(debug=True, port=5000)