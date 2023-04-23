import json
from flask import Flask, request
from flask_cors import CORS
import time

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

@app.route('/morbin')
def its_morbin_time():
    return "it's morbin time"

@app.route('/double', methods=['POST'])
def get_double():
    json1 = json.loads(json.dumps(request.get_json()))
    print(json1["value1"] + "; " +  json1["value2"])
    return "sum is " + str(int(json1["value1"]) + int(json1["value2"]))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print("hello from upload_files()")

    for _ in range(25):
        for x in range(10000):
            print(x)

    file = request.form['file']

    return "ok"

if __name__ == "__main__":
    app.run(debug=True, port=5000)